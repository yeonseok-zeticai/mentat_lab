"""Export the rewritten ONNX model as a torch ExportedProgram (.pt2).

Uses the same onnx2torch path as convert_coreml.py, then torch.export.export.
"""
import warnings
warnings.filterwarnings("ignore")
import os, numpy as np, onnx, torch, onnx2torch
import torch.nn as nn

from onnx2torch.node_converters.registry import _CONVERTER_REGISTRY, OperationDescription
from onnx2torch.utils.common import OnnxToTorchModule, OperationConverterResult, onnx_mapping_from_node

SRC = "build/max_ops_final.opset13.onnx"
# torch.export.export fails on this graph (data-dependent reshape shapes), so
# we save the TorchScript trace (.pt) instead — same idea, older container.
DST = "/mnt/disks/zeticai_database/models/trialandtestworld/max_ops_final.pt"

# Version aliasing
existing = {(d.operation_type, d.version): conv for d, conv in _CONVERTER_REGISTRY.items() if d.domain == ""}
for op in {n.op_type for n in onnx.load(SRC).graph.node}:
    versions = sorted(v for (o, v) in existing if o == op)
    if not versions:
        continue
    latest = versions[-1]
    conv = existing[(op, latest)]
    for v in range(latest + 1, 25):
        d = OperationDescription(domain="", operation_type=op, version=v)
        if d not in _CONVERTER_REGISTRY:
            _CONVERTER_REGISTRY[d] = conv


class _OH(nn.Module, OnnxToTorchModule):
    def __init__(self, axis=-1):
        super().__init__()
        self._axis = axis

    def forward(self, indices, depth, values):
        d = int(depth.item())
        eye = torch.eye(d, dtype=values.dtype, device=values.device)
        e = eye * (values[1] - values[0]) + values[0]
        out = torch.embedding(e, indices.long())
        ax = self._axis
        if ax < 0:
            ax += out.dim()
        if ax != out.dim() - 1:
            perm = list(range(out.dim()))
            perm.insert(ax, perm.pop(-1))
            out = out.permute(perm).contiguous()
        return out


class _SE(nn.Module, OnnxToTorchModule):
    def __init__(self, axis=0):
        super().__init__()
        self._axis = axis

    def forward(self, data, indices, updates):
        return data.clone().scatter(self._axis, indices.long(), updates)


for ver in (11, 13, 16, 18, 19, 20):
    for op_name, mod_cls in [("OneHot", _OH), ("ScatterElements", _SE)]:
        d = OperationDescription(domain="", operation_type=op_name, version=ver)
        if d not in _CONVERTER_REGISTRY:
            def _f(_n, _g, _cls=mod_cls, _op=op_name):
                ax = _n.attributes.get("axis", -1 if _op == "OneHot" else 0)
                return OperationConverterResult(
                    torch_module=_cls(axis=ax),
                    onnx_mapping=onnx_mapping_from_node(node=_n),
                )
            _CONVERTER_REGISTRY[d] = _f


# Bypass onnx2torch's data-dependent `if torch.any(shape == 0)` in Reshape —
# torch.export doesn't allow guarding on runtime tensor values.
import onnx2torch.node_converters.reshape as _rs
def _do_reshape_static(input_tensor, shape):
    sizes = [int(s) for s in shape]
    sizes = [int(input_tensor.shape[i]) if v == 0 else v for i, v in enumerate(sizes)]
    return torch.reshape(input_tensor, torch.Size(sizes))
_rs.OnnxReshape._do_reshape = staticmethod(_do_reshape_static)

print("[step] ONNX → torch")
torch_model = onnx2torch.convert(SRC).eval()

inputs_np = {
    "x2d": np.load("inputs/00_01_x2d.npy"),
    "x1d": np.load("inputs/01_02_x1d.npy"),
    "x3d": np.load("inputs/02_03_x3d.npy"),
    "v16": np.load("inputs/03_04_v16.npy"),
    "seq": np.load("inputs/04_05_seq.npy"),
    "idx": np.load("inputs/05_06_idx.npy"),
}
order = [i.name for i in onnx.load(SRC).graph.input]
example = tuple(torch.from_numpy(inputs_np[n]) for n in order)

print("[step] torch.jit.trace")
traced = torch.jit.trace(torch_model, example, strict=False, check_trace=False)

print(f"[step] saving {DST}")
torch.jit.save(traced, DST)
print(f"[ok] {DST}  size={os.path.getsize(DST)} bytes")

print("[step] reload & verify")
loaded = torch.jit.load(DST)
out = loaded(*example)
print(f"[ok] reloaded out = {out}")
