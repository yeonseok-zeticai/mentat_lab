"""Manual opset-18→13 downgrade for nodes onnx2torch's converter chokes on.
The official version_converter has no Split 18-adapter, so we patch the few
ops that actually changed shape between versions and lower the declared opset.
"""
import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

SRC = "build/max_ops_final.rewritten.onnx"
DST = "build/max_ops_final.opset13.onnx"

m = onnx.load(SRC)
g = m.graph

inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}

# ReduceSum moved axes→input at opset 13 (so must stay input at v17).
# Everything else moved at opset 18 (so attribute form is the v17 form).
REDUCE_OPS = (
    "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd",
    "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare",
)

new_nodes = []
for n in g.node:
    if n.op_type in REDUCE_OPS and len(n.input) > 1:
        x = n.input[0]
        axes_inp = n.input[1]
        axes_arr = inits.get(axes_inp)
        if axes_arr is None:
            print(f"[warn] {n.op_type} {n.name}: axes is non-constant — leaving as-is")
            new_nodes.append(n)
            continue
        axes_list = [int(v) for v in axes_arr.flatten().tolist()]
        keepdims = next((a.i for a in n.attribute if a.name == "keepdims"), 1)
        noop = next((a.i for a in n.attribute if a.name == "noop_with_empty_axes"), 0)
        if noop:
            print(f"[warn] {n.op_type} {n.name}: noop_with_empty_axes=1 not preserved")
        new_attrs = [helper.make_attribute("keepdims", keepdims)]
        if axes_list:
            new_attrs.append(helper.make_attribute("axes", axes_list))
        new_node = helper.make_node(
            n.op_type, [x], list(n.output), name=n.name,
        )
        new_node.attribute.extend(new_attrs)
        new_nodes.append(new_node)
        continue
    new_nodes.append(n)

del g.node[:]
g.node.extend(new_nodes)

# Strip training_mode attr from BatchNormalization (defaulted to 0 anyway pre-15).
for n in g.node:
    if n.op_type == "BatchNormalization":
        keep = [a for a in n.attribute if a.name != "training_mode"]
        del n.attribute[:]
        n.attribute.extend(keep)

# Lower opset import to 17 (avoids 18-only Reduce form; ≥15 keeps BatchNorm v15 schema).
for o in m.opset_import:
    if o.domain == "" or o.domain == "ai.onnx":
        o.version = 17

onnx.checker.check_model(m, full_check=False)
onnx.save(m, DST)
print(f"[ok] saved {DST}")

# Validate numerically.
inputs = {
    "x2d": np.load("inputs/00_01_x2d.npy"),
    "x1d": np.load("inputs/01_02_x1d.npy"),
    "x3d": np.load("inputs/02_03_x3d.npy"),
    "v16": np.load("inputs/03_04_v16.npy"),
    "seq": np.load("inputs/04_05_seq.npy"),
    "idx": np.load("inputs/05_06_idx.npy"),
}
sess_o = ort.InferenceSession(SRC, providers=["CPUExecutionProvider"])
sess_n = ort.InferenceSession(DST, providers=["CPUExecutionProvider"])
o = sess_o.run(None, inputs)[0]
n_ = sess_n.run(None, inputs)[0]
print(f"[info] before: {o}")
print(f"[info] after : {n_}")
print(f"[info] |diff| = {abs(float(o) - float(n_))}")
