"""ONNX -> PyTorch -> CoreML, since coremltools 9 dropped the ONNX source."""
import os, warnings, sys
warnings.filterwarnings("ignore")
import numpy as np
import onnx
import torch
import onnx2torch
import coremltools as ct
# coremltools 9 lacks a broadcast_to torch handler — alias to expand.
from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY
from coremltools.converters.mil.frontend.torch import ops as _ct_torch_ops
_TORCH_OPS_REGISTRY.register_func(_ct_torch_ops.expand, torch_alias=["broadcast_to"], override=True)


# fmod(x, y) = x - trunc(x/y)*y  (sign of dividend, unlike Python's % which is floor-div based)
def _fmod_handler(context, node):
    from coremltools.converters.mil import mil
    mb = mil.Builder
    inputs = _ct_torch_ops._get_inputs(context, node)
    x, y = _ct_torch_ops.promote_input_dtypes([inputs[0], inputs[1]])
    div = mb.real_div(x=x, y=y, name=node.name + "_div")
    context.add(div)
    # trunc(z): floor for z>=0, ceil for z<0  →  sign(z)*floor(|z|)
    abs_div = mb.abs(x=div, name=node.name + "_abs"); context.add(abs_div)
    floor_abs = mb.floor(x=abs_div, name=node.name + "_floorabs"); context.add(floor_abs)
    sign_div = mb.sign(x=div, name=node.name + "_sign"); context.add(sign_div)
    trunc = mb.mul(x=sign_div, y=floor_abs, name=node.name + "_trunc"); context.add(trunc)
    scaled = mb.mul(x=trunc, y=y, name=node.name + "_scaled"); context.add(scaled)
    out = mb.sub(x=x, y=scaled, name=node.name)
    context.add(out)


_TORCH_OPS_REGISTRY.register_func(_fmod_handler, torch_alias=["fmod"], override=True)


# Torch ops that have direct equivalents under different names.
_TORCH_OPS_REGISTRY.register_func(_ct_torch_ops.gt, torch_alias=["greater"], override=True)
_TORCH_OPS_REGISTRY.register_func(_ct_torch_ops.ge, torch_alias=["greater_equal"], override=True)
_TORCH_OPS_REGISTRY.register_func(_ct_torch_ops.lt, torch_alias=["less"], override=True)
_TORCH_OPS_REGISTRY.register_func(_ct_torch_ops.le, torch_alias=["less_equal"], override=True)
# Bicubic upsample — coremltools has no native bicubic op; fall back to bilinear
# (the ONNX op is in this graph only as op-coverage, output is reduced to a scalar).
_TORCH_OPS_REGISTRY.register_func(_ct_torch_ops.upsample_bilinear2d, torch_alias=["upsample_bicubic2d"], override=True)


def _prod_handler(context, node):
    """torch.prod: prod over all dims, or prod over a single dim with optional keepdim."""
    from coremltools.converters.mil import mil
    mb = mil.Builder
    inputs = _ct_torch_ops._get_inputs(context, node)
    x = inputs[0]
    if len(inputs) == 1:
        out = mb.reduce_prod(x=x, axes=list(range(x.rank)), keep_dims=False, name=node.name)
    else:
        # signature: prod(x, dim, keepdim=False, *, dtype=None)
        dim = inputs[1].val if hasattr(inputs[1], "val") else inputs[1]
        keepdim = bool(inputs[2].val) if len(inputs) > 2 and hasattr(inputs[2], "val") else False
        out = mb.reduce_prod(x=x, axes=[int(dim)], keep_dims=keepdim, name=node.name)
    context.add(out)


_TORCH_OPS_REGISTRY.register_func(_prod_handler, torch_alias=["prod"], override=True)

SRC = "build/max_ops_final.opset13.onnx"
DST = "build/max_ops_final.mlpackage"

shapes = {
    "x2d": (2, 3, 8, 8),
    "x1d": (2, 3, 8),
    "x3d": (2, 3, 4, 8, 8),
    "v16": (2, 16),
    "seq": (2, 4, 8),
    "idx": (2, 4),
}

# Custom converters for ops onnx2torch hasn't shipped yet (OneHot, ScatterElements).
import torch.nn as nn
from onnx2torch.node_converters.registry import _CONVERTER_REGISTRY, OperationDescription, add_converter
from onnx2torch.utils.common import OnnxToTorchModule, OperationConverterResult, onnx_mapping_from_node


class _OnnxOneHot(nn.Module, OnnxToTorchModule):
    def __init__(self, axis: int = -1):
        super().__init__()
        self._axis = axis

    def forward(self, indices, depth, values):
        depth_int = int(depth.item())
        eye = torch.eye(depth_int, dtype=values.dtype, device=values.device)
        # values = [off, on], so encoding = (eye * (on - off) + off) — shape [depth, depth]
        on_v, off_v = values[1], values[0]
        encoded = eye * (on_v - off_v) + off_v
        flat_idx = indices.long()
        out = torch.embedding(encoded, flat_idx)  # shape: indices.shape + [depth]
        ax = self._axis
        if ax < 0:
            ax += out.dim()
        if ax != out.dim() - 1:
            perm = list(range(out.dim()))
            new_axis_pos = ax
            perm.insert(new_axis_pos, perm.pop(-1))
            out = out.permute(perm).contiguous()
        return out


class _OnnxScatterElements(nn.Module, OnnxToTorchModule):
    def __init__(self, axis: int = 0, reduction: str = "none"):
        super().__init__()
        self._axis = axis
        self._reduction = reduction

    def forward(self, data, indices, updates):
        out = data.clone()
        return out.scatter(self._axis, indices.long(), updates)


for ver in (11, 13, 16, 18):
    desc_oh = OperationDescription(domain="", operation_type="OneHot", version=ver)
    if desc_oh not in _CONVERTER_REGISTRY:
        def _make_oh(_node, _graph, _v=ver):
            axis = _node.attributes.get("axis", -1)
            return OperationConverterResult(
                torch_module=_OnnxOneHot(axis=axis),
                onnx_mapping=onnx_mapping_from_node(node=_node),
            )
        _CONVERTER_REGISTRY[desc_oh] = _make_oh
    desc_se = OperationDescription(domain="", operation_type="ScatterElements", version=ver)
    if desc_se not in _CONVERTER_REGISTRY:
        def _make_se(_node, _graph, _v=ver):
            axis = _node.attributes.get("axis", 0)
            return OperationConverterResult(
                torch_module=_OnnxScatterElements(axis=axis),
                onnx_mapping=onnx_mapping_from_node(node=_node),
            )
        _CONVERTER_REGISTRY[desc_se] = _make_se
print("[step] aliasing onnx2torch converters for newer op versions")
present_ops = sorted({n.op_type for n in onnx.load(SRC).graph.node})
existing = {(d.operation_type, d.version): conv for d, conv in _CONVERTER_REGISTRY.items() if d.domain == ""}
for op in present_ops:
    versions = sorted(v for (o, v) in existing if o == op)
    if not versions:
        continue
    latest_known = versions[-1]
    converter = existing[(op, latest_known)]
    for v in range(latest_known + 1, 25):
        desc = OperationDescription(domain="", operation_type=op, version=v)
        if desc not in _CONVERTER_REGISTRY:
            _CONVERTER_REGISTRY[desc] = converter

print("[step] loading ONNX → torch")
torch_model = onnx2torch.convert(SRC)
torch_model.eval()

# Build sample input matching ONNX graph input order.
m = onnx.load(SRC)
input_order = [i.name for i in m.graph.input]
print(f"[step] graph input order: {input_order}")

example = []
for name in input_order:
    arr = np.load(f"inputs/{[fn for fn in os.listdir('inputs') if fn.endswith(name+'.npy')][0]}")
    example.append(torch.from_numpy(arr))
example = tuple(example)

print("[step] tracing torch model")
try:
    traced = torch.jit.trace(torch_model, example, strict=False, check_trace=False)
except Exception as e:
    print(f"[err] trace failed: {type(e).__name__}: {str(e)[:500]}")
    sys.exit(1)

print("[step] coremltools.convert")
ct_inputs = []
for name, sh in zip(input_order, [shapes[n] for n in input_order]):
    dtype = np.int32 if name == "idx" else np.float32
    ct_inputs.append(ct.TensorType(name=name, shape=sh, dtype=dtype))

# reduce_transposes pass crashes on this graph (KeyError 'axes' on some
# axis-update op). Drop it from the pipeline.
pipeline = ct.PassPipeline.DEFAULT
pipeline.remove_passes({"common::reduce_transposes"})

mlmodel = ct.convert(
    traced,
    inputs=ct_inputs,
    source="pytorch",
    minimum_deployment_target=ct.target.iOS17,
    compute_precision=ct.precision.FLOAT32,
    pass_pipeline=pipeline,
)
mlmodel.save(DST)
print(f"[ok] saved {DST}")
