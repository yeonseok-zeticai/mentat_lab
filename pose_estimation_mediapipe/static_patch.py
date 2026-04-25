"""Replace onnx2torch dynamic-shape modules with static equivalents.

`torch.export` runs in fake-tensor mode and trips on calls like `tensor.tolist()`
or `torch.any(shape == 0)` because they create unbacked symints. The pose
ONNX graph passes shape/pad/size tensors as ordinary inputs even though they
come from constant initializers — replacing them with python-int constants
baked into the module makes the graph fully static.

Patched op types (from onnx2torch):
- OnnxPadDynamic     -> F.pad with fixed pad list
- OnnxReshape        -> torch.reshape with fixed shape
- OnnxResize         -> F.interpolate with fixed size
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.fx import GraphModule


def _resolve_constant(gm: GraphModule, name: str) -> Optional[torch.Tensor]:
    """Walk dotted attribute path on `gm` and return the tensor (or None)."""
    obj = gm
    for part in name.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj if isinstance(obj, torch.Tensor) else None


def _onnx_pads_to_torch(onnx_pads: List[int]) -> List[int]:
    # ONNX layout: [b0, b1, ..., bN-1, e0, e1, ..., eN-1] over input dims 0..N-1.
    # torch F.pad layout: (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right).
    rank = len(onnx_pads) // 2
    begins = onnx_pads[:rank]
    ends = onnx_pads[rank:]
    out: List[int] = []
    for i in range(rank - 1, -1, -1):
        out.extend([begins[i], ends[i]])
    return out


class StaticPad(torch.nn.Module):
    def __init__(self, torch_pads: Iterable[int], mode: str = "constant", value: float = 0.0):
        super().__init__()
        self.torch_pads = list(torch_pads)
        self.mode = mode
        self.value = value

    def forward(self, x: torch.Tensor, *unused) -> torch.Tensor:
        return F.pad(x, pad=self.torch_pads, mode=self.mode, value=self.value)


class StaticReshape(torch.nn.Module):
    def __init__(self, shape: Iterable[int]):
        super().__init__()
        self.shape: Tuple[int, ...] = tuple(int(s) for s in shape)

    def forward(self, x: torch.Tensor, *unused) -> torch.Tensor:
        return torch.reshape(x, self.shape)


class StaticResize(torch.nn.Module):
    """ONNX Resize with constant `sizes` input — torch.nn.functional.interpolate."""

    def __init__(
        self,
        sizes: Iterable[int],
        onnx_mode: str = "nearest",
        align_corners: Optional[bool] = None,
    ):
        super().__init__()
        sizes = list(int(s) for s in sizes)
        # Drop batch/channel dims (NCHW)
        self.spatial_size: Tuple[int, ...] = tuple(sizes[2:])
        # Map ONNX mode → torch mode for 4D inputs (assume 2D spatial).
        if onnx_mode == "nearest":
            self.torch_mode = "nearest"
        elif onnx_mode in ("linear", "bilinear"):
            self.torch_mode = "bilinear"
        elif onnx_mode == "cubic":
            self.torch_mode = "bicubic"
        else:
            raise ValueError(f"unsupported resize mode: {onnx_mode!r}")
        self.align_corners = align_corners if self.torch_mode != "nearest" else None

    def forward(self, x: torch.Tensor, *unused) -> torch.Tensor:
        return F.interpolate(
            x,
            size=self.spatial_size,
            mode=self.torch_mode,
            align_corners=self.align_corners,
        )


def _safe_name(raw: str, used: set[str]) -> str:
    out = []
    for ch in raw:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    base = "".join(out).strip("_") or "node"
    if base[0].isdigit():
        base = "n_" + base
    name = base
    i = 0
    while name in used:
        i += 1
        name = f"{base}_{i}"
    used.add(name)
    return name


def sanitize_module_names(gm: GraphModule) -> GraphModule:
    """Rename submodules whose dotted target paths contain `:`, `;`, `/`, etc.

    `torch.export` serializes the FX graph as text, and certain separator
    characters break its metadata parser. Pure-identifier module paths
    survive the round-trip cleanly.
    """
    used: set[str] = set()
    rename: dict[str, str] = {}

    # Pre-reserve top-level names that already are valid identifiers.
    for name, _ in gm.named_children():
        if name.replace("_", "").isalnum() and not name[0].isdigit():
            used.add(name)

    for node in list(gm.graph.nodes):
        if node.op != "call_module" or not isinstance(node.target, str):
            continue
        target = node.target
        if "." in target or any(c in target for c in ":;/"):
            if target not in rename:
                rename[target] = _safe_name(target, used)

    if not rename:
        return gm

    for old, new in rename.items():
        try:
            mod = gm.get_submodule(old)
        except AttributeError:
            continue
        # Move the submodule under the sanitized name.
        gm.add_module(new, mod)
        # Remove the old attribute (it may have been added via setattr with
        # a literal name containing weird chars).
        try:
            delattr(gm, old)
        except AttributeError:
            pass

    for node in list(gm.graph.nodes):
        if node.op == "call_module" and isinstance(node.target, str) and node.target in rename:
            node.target = rename[node.target]

    gm.graph.lint()
    gm.recompile()
    return gm


def patch_dynamic_ops(gm: GraphModule) -> GraphModule:
    """In-place replace dynamic onnx2torch modules with static ones."""
    # Late import — keep static_patch importable without onnx2torch installed.
    from onnx2torch.node_converters.pad import OnnxPadDynamic
    from onnx2torch.node_converters.reshape import OnnxReshape
    from onnx2torch.node_converters.resize import OnnxResize

    replacements: List[Tuple[str, torch.nn.Module]] = []

    for node in list(gm.graph.nodes):
        if node.op != "call_module":
            continue
        target = node.target
        if not isinstance(target, str):
            continue
        try:
            mod = gm.get_submodule(target)
        except AttributeError:
            continue

        # Helper: pull constant tensors for trailing inputs.
        const_inputs = []
        for arg in node.args[1:]:
            if hasattr(arg, "op") and arg.op == "get_attr":
                const_inputs.append(_resolve_constant(gm, arg.target))
            else:
                const_inputs.append(None)

        if isinstance(mod, OnnxPadDynamic):
            pads_t = const_inputs[0] if const_inputs else None
            if pads_t is None:
                raise RuntimeError(f"Pad node {target} has non-constant pads")
            torch_pads = _onnx_pads_to_torch(pads_t.tolist())
            value = 0.0
            if len(const_inputs) > 1 and isinstance(const_inputs[1], torch.Tensor):
                value = float(const_inputs[1].item())
            replacements.append((target, StaticPad(torch_pads, mode=mod.mode, value=value)))

        elif isinstance(mod, OnnxReshape):
            shape_t = const_inputs[0] if const_inputs else None
            if shape_t is None:
                raise RuntimeError(f"Reshape node {target} has non-constant shape")
            replacements.append((target, StaticReshape(shape_t.tolist())))

        elif isinstance(mod, OnnxResize):
            # OnnxResize args = (x, roi, scales, sizes) — pick whichever is non-empty.
            sizes_t = const_inputs[2] if len(const_inputs) > 2 else None
            scales_t = const_inputs[1] if len(const_inputs) > 1 else None
            if sizes_t is not None and sizes_t.numel() > 0:
                replacements.append(
                    (
                        target,
                        StaticResize(
                            sizes_t.tolist(),
                            onnx_mode=mod.onnx_mode,
                            align_corners=mod.align_corners,
                        ),
                    )
                )
            elif scales_t is not None and scales_t.numel() > 0:
                # Fall back to scale-based sizes derived from input shape on first call.
                raise RuntimeError(f"Resize node {target} only has scales — not implemented")
            else:
                raise RuntimeError(f"Resize node {target} has no constant sizes/scales")

    if not replacements:
        return gm

    for target, new_mod in replacements:
        # Replace submodule via dotted-path setattr.
        parent = gm
        parts = target.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_mod)

    # Drop the now-unused initializer args from each replaced call_module.
    for node in list(gm.graph.nodes):
        if node.op != "call_module":
            continue
        if not isinstance(node.target, str):
            continue
        try:
            mod = gm.get_submodule(node.target)
        except AttributeError:
            continue
        if isinstance(mod, (StaticPad, StaticReshape, StaticResize)):
            # Keep only the first arg (the data tensor).
            if len(node.args) > 1:
                node.args = (node.args[0],)

    gm.graph.eliminate_dead_code()
    gm.recompile()

    # Drop dead `get_attr` targets so they don't show up in the state_dict
    # of the ExportedProgram (otherwise torch.export.save chokes on int64
    # initializers that are no longer referenced).
    referenced: set[str] = set()
    for node in gm.graph.nodes:
        if node.op == "get_attr" and isinstance(node.target, str):
            referenced.add(node.target)
    init_mod = getattr(gm, "initializers", None)
    if init_mod is not None:
        # Strip referenced "initializers." prefix to compare names locally.
        ref_local = {n.split(".", 1)[1] for n in referenced if n.startswith("initializers.")}
        for name in list(init_mod._parameters.keys()):
            if name not in ref_local:
                del init_mod._parameters[name]
        for name in list(init_mod._buffers.keys()):
            if name not in ref_local:
                del init_mod._buffers[name]
    return gm
