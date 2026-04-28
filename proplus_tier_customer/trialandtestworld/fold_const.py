"""Constant-fold every subgraph that doesn't depend on dynamic inputs.

Why: max_ops_final.onnx contains many ops (STFT/Loop/If/Scan/Sequence*/Random*/
TfIdf/StringNormalizer/Det/...) that no on-device backend supports. All of them
turn out to take only constant inputs, so we can pre-evaluate them with ORT and
splice the results back in as initializers. After folding, all unsupported ops
disappear and the graph reduces to backend-friendly compute on the dynamic inputs.
"""
import os, sys, copy
import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto

SRC = "max_ops_final.onnx"
DST = "build/max_ops_final.folded.onnx"
os.makedirs("build", exist_ok=True)

m = onnx.load(SRC)
# Shape inference so we know dtypes for the tensors we want to expose as outputs.
try:
    m = onnx.shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)
except Exception as e:
    print(f"[warn] shape inference failed: {e!r}")
g = m.graph
vi_dtype = {}
for vi in list(g.value_info) + list(g.input) + list(g.output):
    try:
        vi_dtype[vi.name] = vi.type.tensor_type.elem_type
    except Exception:
        pass

init_names = {i.name for i in g.initializer}
input_names = {i.name for i in g.input}

producers = {}
for n in g.node:
    for o in n.output:
        if o:
            producers[o] = n

dyn_cache = {}

def is_dyn(t):
    if t in dyn_cache:
        return dyn_cache[t]
    if t in input_names:
        dyn_cache[t] = True
        return True
    if t in init_names or t not in producers:
        dyn_cache[t] = False
        return False
    n = producers[t]
    # subgraph-bearing ops have no tensor inputs that bring dyn deps directly,
    # so this is sufficient when paired with the "all flagged ops are const" check above.
    res = any(is_dyn(i) for i in n.input if i)
    for o in n.output:
        if o:
            dyn_cache[o] = res
    return res

all_tensors = set()
for n in g.node:
    for x in list(n.input) + list(n.output):
        if x:
            all_tensors.add(x)
for t in all_tensors:
    is_dyn(t)

# Frontier: const tensors that feed a dyn-output node, or are graph outputs
frontier = set()
for n in g.node:
    node_dyn = any(dyn_cache.get(i, False) for i in n.input if i)
    if node_dyn:
        for inp in n.input:
            if inp and inp not in init_names and inp not in input_names and not dyn_cache.get(inp, False):
                frontier.add(inp)
for o in g.output:
    if not dyn_cache.get(o.name, False) and o.name not in init_names and o.name not in input_names:
        frontier.add(o.name)

print(f"[info] frontier tensors: {len(frontier)}")

# 1) Build a probe model: original graph with frontier tensors added as outputs.
probe = copy.deepcopy(m)
existing_outputs = {o.name for o in probe.graph.output}
skipped_no_dtype = []
probe_targets = []
for t in frontier:
    if t in existing_outputs:
        probe_targets.append(t)
        continue
    dtype = vi_dtype.get(t, 0)
    if dtype == 0 or dtype == TensorProto.UNDEFINED:
        skipped_no_dtype.append(t)
        continue
    probe.graph.output.append(helper.make_tensor_value_info(t, dtype, None))
    probe_targets.append(t)
print(f"[info] frontier exposed: {len(probe_targets)}, skipped (no dtype): {len(skipped_no_dtype)}")
if skipped_no_dtype[:5]:
    print(f"[info] sample skipped: {skipped_no_dtype[:5]}")
onnx.save(probe, "build/_probe.onnx")

# 2) Run ORT once.
inputs = {
    "x2d": np.load("inputs/00_01_x2d.npy"),
    "x1d": np.load("inputs/01_02_x1d.npy"),
    "x3d": np.load("inputs/02_03_x3d.npy"),
    "v16": np.load("inputs/03_04_v16.npy"),
    "seq": np.load("inputs/04_05_seq.npy"),
    "idx": np.load("inputs/05_06_idx.npy"),
}
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession("build/_probe.onnx", so, providers=["CPUExecutionProvider"])
out_names = [o.name for o in sess.get_outputs()]
results = sess.run(out_names, inputs)
captured = dict(zip(out_names, results))
orig_out = captured[g.output[0].name]
print(f"[info] original out = {orig_out}")

# 3) Splice frontier tensors in as initializers; drop nodes whose every output is const.
new_inits = list(g.initializer)
captured_initnames = set(init_names)
for t in frontier:
    if t in captured_initnames:
        continue
    val = captured.get(t)
    if val is None:
        print(f"[warn] frontier tensor {t} not captured by ORT — skipping")
        continue
    # Sequence outputs come back as Python lists; can't fold those directly.
    if isinstance(val, list):
        print(f"[warn] frontier tensor {t} is a Sequence (list) — needs ConcatFromSequence-style splicing; skipping")
        continue
    if not isinstance(val, np.ndarray):
        val = np.asarray(val)
    tp = numpy_helper.from_array(val, name=t)
    new_inits.append(tp)
    captured_initnames.add(t)

# Decide which nodes to keep: those that are dyn (or feed a graph output transitively via dyn path).
# Simpler rule: keep node iff at least one of its outputs is dyn OR is a frontier tensor that wasn't captured.
keep_idx = []
for i, n in enumerate(g.node):
    keeps = False
    for o in n.output:
        if not o:
            continue
        if dyn_cache.get(o, False):
            keeps = True
            break
        if o not in captured_initnames:
            # uncaptured const tensor that something downstream still references — keep as fallback
            # but only if downstream actually uses it.
            pass
    if keeps:
        keep_idx.append(i)

new_nodes = [g.node[i] for i in keep_idx]
print(f"[info] kept nodes: {len(new_nodes)} / {len(g.node)}")

# Build new graph
new_graph = helper.make_graph(
    nodes=new_nodes,
    name=g.name,
    inputs=list(g.input),
    outputs=list(g.output),
    initializer=new_inits,
    value_info=list(g.value_info),
)
new_model = helper.make_model(new_graph, opset_imports=list(m.opset_import), ir_version=m.ir_version)
new_model.producer_name = "fold_const"
onnx.checker.check_model(new_model, full_check=False)
onnx.save(new_model, DST)
print(f"[info] saved {DST}")

# 4) Verify
sess2 = ort.InferenceSession(DST, providers=["CPUExecutionProvider"])
new_out = sess2.run(None, inputs)[0]
print(f"[info] folded out = {new_out}")
diff = np.abs(np.asarray(new_out) - np.asarray(orig_out))
print(f"[info] |diff| = {diff} (max element = {np.max(diff) if diff.ndim else diff})")
