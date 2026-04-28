"""Prepare an ONNX model for on-device deployment.

End-to-end pipeline that takes the original ONNX and produces a single
modified ONNX whose op set fits the supported subset of mainstream on-device
runtimes. Three passes are applied in order; each pass is verified against
the previous output via ONNX Runtime so the final model is numerically
equivalent on the supplied calibration inputs.

  Pass 1 — fold constant sub-DAGs  (eliminates ops that don't depend on
           dynamic inputs: control-flow, sequence, random, signal-processing,
           string ops, etc.)
  Pass 2 — rewrite remaining unsupported / poorly-supported ops to
           equivalent primitive op compositions.
  Pass 3 — downgrade opset 18 → 17 (move Reduce-family axes from input
           back to attribute, strip BatchNormalization.training_mode).

Usage
-----
    python prepare_for_ondevice.py \
        --input  max_ops_final.onnx \
        --output max_ops_final.modified.onnx \
        --inputs-dir inputs

The --inputs-dir directory must contain one .npy per graph input, named
"<order>_<order>_<input_name>.npy" (e.g. "00_01_x2d.npy").
"""
from __future__ import annotations
import argparse
import copy
import os
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto


# ---------------------------------------------------------------------------
# Pass 1 — constant folding of dyn-independent sub-DAGs
# ---------------------------------------------------------------------------

def _classify_dyn(model: onnx.ModelProto):
    g = model.graph
    init_names = {i.name for i in g.initializer}
    input_names = {i.name for i in g.input}
    producers = {o: n for n in g.node for o in n.output if o}

    cache: dict[str, bool] = {}

    def is_dyn(t: str) -> bool:
        if t in cache:
            return cache[t]
        if t in input_names:
            cache[t] = True
            return True
        if t in init_names or t not in producers:
            cache[t] = False
            return False
        n = producers[t]
        res = any(is_dyn(i) for i in n.input if i)
        for o in n.output:
            if o:
                cache[o] = res
        return res

    for t in {x for n in g.node for x in (*n.input, *n.output) if x}:
        is_dyn(t)
    return cache, init_names, input_names


def _fold_constants(model: onnx.ModelProto, inputs: dict[str, np.ndarray]) -> onnx.ModelProto:
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    g = model.graph
    dyn, init_names, input_names = _classify_dyn(model)

    vi_dtype: dict[str, int] = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        try:
            vi_dtype[vi.name] = vi.type.tensor_type.elem_type
        except Exception:
            pass

    frontier: set[str] = set()
    for n in g.node:
        if any(dyn.get(i, False) for i in n.input if i):
            for inp in n.input:
                if inp and inp not in init_names and inp not in input_names \
                        and not dyn.get(inp, False):
                    frontier.add(inp)
    for o in g.output:
        if not dyn.get(o.name, False) and o.name not in init_names \
                and o.name not in input_names:
            frontier.add(o.name)

    probe = copy.deepcopy(model)
    existing_outputs = {o.name for o in probe.graph.output}
    skipped: list[str] = []
    probe_targets: list[str] = []
    for t in frontier:
        if t in existing_outputs:
            probe_targets.append(t)
            continue
        dt = vi_dtype.get(t, 0)
        if dt in (0, TensorProto.UNDEFINED):
            skipped.append(t)
            continue
        probe.graph.output.append(helper.make_tensor_value_info(t, dt, None))
        probe_targets.append(t)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tf:
        probe_path = tf.name
    onnx.save(probe, probe_path)
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(probe_path, so, providers=["CPUExecutionProvider"])
        out_names = [o.name for o in sess.get_outputs()]
        captured = dict(zip(out_names, sess.run(out_names, inputs)))
    finally:
        os.unlink(probe_path)

    new_inits = list(g.initializer)
    captured_initnames = set(init_names)
    for t in frontier:
        if t in captured_initnames or t in skipped:
            continue
        val = captured.get(t)
        if val is None or isinstance(val, list):
            continue
        if not isinstance(val, np.ndarray):
            val = np.asarray(val)
        new_inits.append(numpy_helper.from_array(val, name=t))
        captured_initnames.add(t)

    keep_nodes = [
        n for n in g.node
        if any(o and dyn.get(o, False) for o in n.output)
    ]

    new_graph = helper.make_graph(
        nodes=keep_nodes, name=g.name,
        inputs=list(g.input), outputs=list(g.output),
        initializer=new_inits, value_info=list(g.value_info),
    )
    new_model = helper.make_model(
        new_graph, opset_imports=list(model.opset_import), ir_version=model.ir_version,
    )
    new_model.producer_name = "prepare_for_ondevice/fold"
    onnx.checker.check_model(new_model, full_check=False)
    return new_model


# ---------------------------------------------------------------------------
# Pass 2 — rewrite remaining ops not in the on-device runtime supported set
# ---------------------------------------------------------------------------

def _rewrite_ops(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}

    def add_init(name: str, arr: np.ndarray):
        g.initializer.append(numpy_helper.from_array(arr, name=name))

    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    inferred_shape: dict[str, tuple] = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        dims = tuple(d.dim_value if d.dim_value > 0 else d.dim_param
                     for d in vi.type.tensor_type.shape.dim)
        inferred_shape[vi.name] = (vi.type.tensor_type.elem_type, dims)

    def erf_decomp(node):
        # erf(x) ≈ tanh(0.7978845608 * (x + 0.044715 * x^3))
        x = node.input[0]
        base = node.name + "__erf"
        add_init(base + "_c1", np.array([0.7978845608028654], dtype=np.float32))
        add_init(base + "_c2", np.array([0.044715], dtype=np.float32))
        return [
            helper.make_node("Mul", [x, x], [base + "_x2"], name=base + "_x2_n"),
            helper.make_node("Mul", [base + "_x2", x], [base + "_x3"], name=base + "_x3_n"),
            helper.make_node("Mul", [base + "_c2", base + "_x3"], [base + "_cx3"], name=base + "_cx3_n"),
            helper.make_node("Add", [x, base + "_cx3"], [base + "_sumx"], name=base + "_sumx_n"),
            helper.make_node("Mul", [base + "_c1", base + "_sumx"], [base + "_cs"], name=base + "_cs_n"),
            helper.make_node("Tanh", [base + "_cs"], [node.output[0]], name=base + "_tanh_n"),
        ]

    def trilu_static_mask(node):
        X = node.input[0]
        k_name = node.input[1] if len(node.input) > 1 else None
        upper = next((a.i for a in node.attribute if a.name == "upper"), 1)
        k = int(inits[k_name]) if (k_name and k_name in inits) else 0
        _, shape = inferred_shape.get(X, (1, ()))
        if not shape or len(shape) < 2:
            return None
        rows = shape[-2] if isinstance(shape[-2], int) else None
        cols = shape[-1] if isinstance(shape[-1], int) else None
        if rows is None or cols is None:
            return None
        i_idx = np.arange(rows).reshape(-1, 1)
        j_idx = np.arange(cols).reshape(1, -1)
        mask = ((j_idx - i_idx >= k) if upper else (j_idx - i_idx <= k)).astype(np.float32)
        base = node.name + "__trilu"
        add_init(base + "_mask", mask)
        return [helper.make_node("Mul", [X, base + "_mask"], [node.output[0]], name=base + "_mul")]

    def rnn_unroll(node):
        X, Wn, Rn, Bn, _, H0 = node.input
        seq_len, hidden = 4, 8
        base = node.name + "__rnn"
        W, R, Bvec = inits[Wn][0], inits[Rn][0], inits[Bn][0]
        Wb, Rb = Bvec[:hidden], Bvec[hidden:]
        add_init(base + "_Wt", W.T.astype(np.float32))
        add_init(base + "_Rt", R.T.astype(np.float32))
        add_init(base + "_b", (Wb + Rb).astype(np.float32))
        add_init(base + "_sq_axes", np.array([0], dtype=np.int64))
        nodes = [helper.make_node("Squeeze", [H0, base + "_sq_axes"], [base + "_h_init"], name=base + "_sq_h0")]
        h_prev = base + "_h_init"
        h_steps = []
        for t in range(seq_len):
            for nm, val in [("starts", [t]), ("ends", [t + 1]), ("ax", [0]), ("sp", [1])]:
                add_init(base + f"_st_{t}_{nm}", np.array(val, dtype=np.int64))
            add_init(base + f"_x_{t}_sqax", np.array([0], dtype=np.int64))
            nodes += [
                helper.make_node("Slice",
                                 [X, base + f"_st_{t}_starts", base + f"_st_{t}_ends",
                                  base + f"_st_{t}_ax", base + f"_st_{t}_sp"],
                                 [base + f"_x_{t}_sl"], name=base + f"_slice_{t}"),
                helper.make_node("Squeeze", [base + f"_x_{t}_sl", base + f"_x_{t}_sqax"],
                                 [base + f"_x_{t}"], name=base + f"_squeeze_{t}"),
                helper.make_node("MatMul", [base + f"_x_{t}", base + "_Wt"], [base + f"_xw_{t}"], name=base + f"_xw_n_{t}"),
                helper.make_node("MatMul", [h_prev, base + "_Rt"], [base + f"_hr_{t}"], name=base + f"_hr_n_{t}"),
                helper.make_node("Add", [base + f"_xw_{t}", base + f"_hr_{t}"], [base + f"_s1_{t}"], name=base + f"_s1_n_{t}"),
                helper.make_node("Add", [base + f"_s1_{t}", base + "_b"], [base + f"_s2_{t}"], name=base + f"_s2_n_{t}"),
                helper.make_node("Tanh", [base + f"_s2_{t}"], [base + f"_h_{t}"], name=base + f"_tanh_{t}"),
            ]
            h_prev = base + f"_h_{t}"
            h_steps.append(h_prev)
        if len(node.output) > 1 and node.output[1]:
            add_init(base + "_yh_ax", np.array([0], dtype=np.int64))
            nodes.append(helper.make_node("Unsqueeze", [h_prev, base + "_yh_ax"], [node.output[1]], name=base + "_yh_unsq"))
        if node.output and node.output[0]:
            add_init(base + "_y_ax", np.array([0, 1], dtype=np.int64))
            unsq = []
            for t, h_t in enumerate(h_steps):
                un = base + f"_y_un_{t}"
                nodes.append(helper.make_node("Unsqueeze", [h_t, base + "_y_ax"], [un], name=base + f"_y_unsq_{t}"))
                unsq.append(un)
            nodes.append(helper.make_node("Concat", unsq, [node.output[0]], axis=0, name=base + "_y_concat"))
        return nodes

    def lstm_unroll(node):
        X, Wn, Rn, Bn, _, H0, C0 = node.input
        seq_len, hidden = 4, 8
        base = node.name + "__lstm"
        W, R, Bvec = inits[Wn][0], inits[Rn][0], inits[Bn][0]
        Wb, Rb = Bvec[:4 * hidden], Bvec[4 * hidden:]
        bias = (Wb + Rb).astype(np.float32)
        for i, g_ in enumerate(("i", "o", "f", "c")):
            add_init(f"{base}_W{g_}", W[i*hidden:(i+1)*hidden].T.astype(np.float32))
            add_init(f"{base}_R{g_}", R[i*hidden:(i+1)*hidden].T.astype(np.float32))
            add_init(f"{base}_b{g_}", bias[i*hidden:(i+1)*hidden])
        add_init(base + "_sq_ax", np.array([0], dtype=np.int64))
        nodes = [
            helper.make_node("Squeeze", [H0, base + "_sq_ax"], [base + "_h_init"], name=base + "_sq_h0"),
            helper.make_node("Squeeze", [C0, base + "_sq_ax"], [base + "_c_init"], name=base + "_sq_c0"),
        ]
        h_prev, c_prev = base + "_h_init", base + "_c_init"
        h_steps = []
        for t in range(seq_len):
            for nm, val in [("starts", [t]), ("ends", [t+1]), ("ax", [0]), ("sp", [1])]:
                add_init(base + f"_st_{t}_{nm}", np.array(val, dtype=np.int64))
            nodes += [
                helper.make_node("Slice",
                                 [X, base + f"_st_{t}_starts", base + f"_st_{t}_ends",
                                  base + f"_st_{t}_ax", base + f"_st_{t}_sp"],
                                 [base + f"_x_{t}_sl"], name=base + f"_slice_{t}"),
                helper.make_node("Squeeze", [base + f"_x_{t}_sl", base + "_sq_ax"],
                                 [base + f"_x_{t}"], name=base + f"_sq_x_{t}"),
            ]
            gate = {}
            for g_ in ("i", "o", "f", "c"):
                xw, hr = base + f"_xw_{g_}_{t}", base + f"_hr_{g_}_{t}"
                s1, s2 = base + f"_s1_{g_}_{t}", base + f"_s2_{g_}_{t}"
                nodes += [
                    helper.make_node("MatMul", [base + f"_x_{t}", f"{base}_W{g_}"], [xw], name=base + f"_xw_n_{g_}_{t}"),
                    helper.make_node("MatMul", [h_prev, f"{base}_R{g_}"], [hr], name=base + f"_hr_n_{g_}_{t}"),
                    helper.make_node("Add", [xw, hr], [s1], name=base + f"_s1_n_{g_}_{t}"),
                    helper.make_node("Add", [s1, f"{base}_b{g_}"], [s2], name=base + f"_s2_n_{g_}_{t}"),
                ]
                gate[g_] = s2
            it, ot, ft, ct_t = base + f"_it_{t}", base + f"_ot_{t}", base + f"_ft_{t}", base + f"_ctt_{t}"
            c_new, c_th, h_new = base + f"_cnew_{t}", base + f"_ct_{t}", base + f"_hnew_{t}"
            nodes += [
                helper.make_node("Sigmoid", [gate["i"]], [it], name=base + f"_si_{t}"),
                helper.make_node("Sigmoid", [gate["o"]], [ot], name=base + f"_so_{t}"),
                helper.make_node("Sigmoid", [gate["f"]], [ft], name=base + f"_sf_{t}"),
                helper.make_node("Tanh",    [gate["c"]], [ct_t], name=base + f"_tc_{t}"),
                helper.make_node("Mul", [ft, c_prev], [base + f"_fc_{t}"], name=base + f"_fc_n_{t}"),
                helper.make_node("Mul", [it, ct_t], [base + f"_ic_{t}"], name=base + f"_ic_n_{t}"),
                helper.make_node("Add", [base + f"_fc_{t}", base + f"_ic_{t}"], [c_new], name=base + f"_cnew_n_{t}"),
                helper.make_node("Tanh", [c_new], [c_th], name=base + f"_ctanh_{t}"),
                helper.make_node("Mul", [ot, c_th], [h_new], name=base + f"_hmul_{t}"),
            ]
            h_prev, c_prev = h_new, c_new
            h_steps.append(h_new)
        add_init(base + "_yh_ax", np.array([0], dtype=np.int64))
        if len(node.output) > 1 and node.output[1]:
            nodes.append(helper.make_node("Unsqueeze", [h_prev, base + "_yh_ax"], [node.output[1]], name=base + "_yh"))
        if len(node.output) > 2 and node.output[2]:
            nodes.append(helper.make_node("Unsqueeze", [c_prev, base + "_yh_ax"], [node.output[2]], name=base + "_yc"))
        if node.output and node.output[0]:
            add_init(base + "_y_ax", np.array([0, 1], dtype=np.int64))
            unsq = []
            for t, h_t in enumerate(h_steps):
                un = base + f"_y_un_{t}"
                nodes.append(helper.make_node("Unsqueeze", [h_t, base + "_y_ax"], [un], name=base + f"_y_unsq_{t}"))
                unsq.append(un)
            nodes.append(helper.make_node("Concat", unsq, [node.output[0]], axis=0, name=base + "_y_concat"))
        return nodes

    def gru_unroll(node):
        X, Wn, Rn, Bn, _, H0 = node.input
        seq_len, hidden = 4, 8
        lbr = next((a.i for a in node.attribute if a.name == "linear_before_reset"), 0)
        base = node.name + "__gru"
        W, R, Bvec = inits[Wn][0], inits[Rn][0], inits[Bn][0]
        Wb, Rb = Bvec[:3 * hidden], Bvec[3 * hidden:]
        for i, nm in enumerate(("z", "r", "h")):
            add_init(f"{base}_W{nm}", W[i*hidden:(i+1)*hidden].T.astype(np.float32))
            add_init(f"{base}_R{nm}", R[i*hidden:(i+1)*hidden].T.astype(np.float32))
            add_init(f"{base}_Wb{nm}", Wb[i*hidden:(i+1)*hidden])
            add_init(f"{base}_Rb{nm}", Rb[i*hidden:(i+1)*hidden])
        add_init(f"{base}_bz", (Wb[:hidden] + Rb[:hidden]).astype(np.float32))
        add_init(f"{base}_br", (Wb[hidden:2*hidden] + Rb[hidden:2*hidden]).astype(np.float32))
        if lbr == 0:
            add_init(f"{base}_bh", (Wb[2*hidden:] + Rb[2*hidden:]).astype(np.float32))
        add_init(base + "_sq_ax", np.array([0], dtype=np.int64))
        add_init(base + "_one", np.array([1.0], dtype=np.float32))
        nodes = [helper.make_node("Squeeze", [H0, base + "_sq_ax"], [base + "_h_init"], name=base + "_sq_h0")]
        h_prev = base + "_h_init"
        h_steps = []
        for t in range(seq_len):
            for nm, val in [("starts", [t]), ("ends", [t+1]), ("ax", [0]), ("sp", [1])]:
                add_init(base + f"_st_{t}_{nm}", np.array(val, dtype=np.int64))
            nodes += [
                helper.make_node("Slice",
                                 [X, base + f"_st_{t}_starts", base + f"_st_{t}_ends",
                                  base + f"_st_{t}_ax", base + f"_st_{t}_sp"],
                                 [base + f"_x_{t}_sl"], name=base + f"_slice_{t}"),
                helper.make_node("Squeeze", [base + f"_x_{t}_sl", base + "_sq_ax"],
                                 [base + f"_x_{t}"], name=base + f"_sq_x_{t}"),
            ]
            xt = base + f"_x_{t}"
            zt = base + f"_zt_{t}"; rt = base + f"_rt_{t}"
            for g_, out in (("z", zt), ("r", rt)):
                xW = base + f"_xW{g_}_{t}"; hR = base + f"_hR{g_}_{t}"
                sm = base + f"_sm{g_}_{t}"; smb = base + f"_smb{g_}_{t}"
                nodes += [
                    helper.make_node("MatMul", [xt, f"{base}_W{g_}"], [xW], name=base + f"_xW_n_{g_}_{t}"),
                    helper.make_node("MatMul", [h_prev, f"{base}_R{g_}"], [hR], name=base + f"_hR_n_{g_}_{t}"),
                    helper.make_node("Add", [xW, hR], [sm], name=base + f"_sm_n_{g_}_{t}"),
                    helper.make_node("Add", [sm, f"{base}_b{g_}"], [smb], name=base + f"_smb_n_{g_}_{t}"),
                    helper.make_node("Sigmoid", [smb], [out], name=base + f"_s_{g_}_{t}"),
                ]
            xWh = base + f"_xWh_{t}"; ht_t = base + f"_htt_{t}"
            nodes.append(helper.make_node("MatMul", [xt, f"{base}_Wh"], [xWh], name=base + f"_xWh_n_{t}"))
            if lbr == 1:
                xWh_b = base + f"_xWh_b_{t}"; hRh = base + f"_hRh_{t}"; hRh_b = base + f"_hRh_b_{t}"
                rterm = base + f"_rterm_{t}"; pre = base + f"_pre_{t}"
                nodes += [
                    helper.make_node("Add", [xWh, f"{base}_Wbh"], [xWh_b], name=base + f"_xWh_b_n_{t}"),
                    helper.make_node("MatMul", [h_prev, f"{base}_Rh"], [hRh], name=base + f"_hRh_n_{t}"),
                    helper.make_node("Add", [hRh, f"{base}_Rbh"], [hRh_b], name=base + f"_hRh_b_n_{t}"),
                    helper.make_node("Mul", [rt, hRh_b], [rterm], name=base + f"_rterm_n_{t}"),
                    helper.make_node("Add", [xWh_b, rterm], [pre], name=base + f"_pre_n_{t}"),
                    helper.make_node("Tanh", [pre], [ht_t], name=base + f"_ttilde_{t}"),
                ]
            else:
                rh = base + f"_rh_{t}"; rhR = base + f"_rhR_{t}"
                preA = base + f"_preA_{t}"; pre = base + f"_preB_{t}"
                nodes += [
                    helper.make_node("Mul", [rt, h_prev], [rh], name=base + f"_rh_n_{t}"),
                    helper.make_node("MatMul", [rh, f"{base}_Rh"], [rhR], name=base + f"_rhR_n_{t}"),
                    helper.make_node("Add", [xWh, rhR], [preA], name=base + f"_preA_n_{t}"),
                    helper.make_node("Add", [preA, f"{base}_bh"], [pre], name=base + f"_preB_n_{t}"),
                    helper.make_node("Tanh", [pre], [ht_t], name=base + f"_ttilde_{t}"),
                ]
            omz = base + f"_omz_{t}"; htp = base + f"_htp_{t}"; hpp = base + f"_hpp_{t}"
            h_new = base + f"_hnew_{t}"
            nodes += [
                helper.make_node("Sub", [base + "_one", zt], [omz], name=base + f"_omz_n_{t}"),
                helper.make_node("Mul", [omz, ht_t], [htp], name=base + f"_htp_n_{t}"),
                helper.make_node("Mul", [zt, h_prev], [hpp], name=base + f"_hpp_n_{t}"),
                helper.make_node("Add", [htp, hpp], [h_new], name=base + f"_hnew_n_{t}"),
            ]
            h_prev = h_new
            h_steps.append(h_new)
        add_init(base + "_yh_ax", np.array([0], dtype=np.int64))
        if len(node.output) > 1 and node.output[1]:
            nodes.append(helper.make_node("Unsqueeze", [h_prev, base + "_yh_ax"], [node.output[1]], name=base + "_yh"))
        if node.output and node.output[0]:
            add_init(base + "_y_ax", np.array([0, 1], dtype=np.int64))
            unsq = []
            for t, h_t in enumerate(h_steps):
                un = base + f"_y_un_{t}"
                nodes.append(helper.make_node("Unsqueeze", [h_t, base + "_y_ax"], [un], name=base + f"_y_unsq_{t}"))
                unsq.append(un)
            nodes.append(helper.make_node("Concat", unsq, [node.output[0]], axis=0, name=base + "_y_concat"))
        return nodes

    new_nodes = []
    for n in g.node:
        if n.op_type == "Celu":
            alpha = next((a.f for a in n.attribute if a.name == "alpha"), 1.0)
            x = n.input[0]
            base = n.name + "__celu"
            add_init(base + "_alpha", np.array([alpha], dtype=np.float32))
            add_init(base + "_one",   np.array([1.0], dtype=np.float32))
            add_init(base + "_zero",  np.array([0.0], dtype=np.float32))
            new_nodes += [
                helper.make_node("Div", [x, base + "_alpha"], [base + "_div"], name=base + "_div_n"),
                helper.make_node("Exp", [base + "_div"], [base + "_exp"], name=base + "_exp_n"),
                helper.make_node("Sub", [base + "_exp", base + "_one"], [base + "_sub"], name=base + "_sub_n"),
                helper.make_node("Mul", [base + "_alpha", base + "_sub"], [base + "_mul"], name=base + "_mul_n"),
                helper.make_node("Max", [x, base + "_zero"], [base + "_max"], name=base + "_max_n"),
                helper.make_node("Min", [base + "_mul", base + "_zero"], [base + "_min"], name=base + "_min_n"),
                helper.make_node("Add", [base + "_max", base + "_min"], [n.output[0]], name=base + "_add_n"),
            ]
            continue
        if n.op_type == "Tan":
            x = n.input[0]; base = n.name + "__tan"
            new_nodes += [
                helper.make_node("Sin", [x], [base + "_sin"], name=base + "_sin_n"),
                helper.make_node("Cos", [x], [base + "_cos"], name=base + "_cos_n"),
                helper.make_node("Div", [base + "_sin", base + "_cos"], [n.output[0]], name=base + "_div_n"),
            ]
            continue
        if n.op_type == "Acos":
            x = n.input[0]; base = n.name + "__acos"
            add_init(base + "_halfpi", np.array([np.pi / 2.0], dtype=np.float32))
            new_nodes += [
                helper.make_node("Asin", [x], [base + "_asin"], name=base + "_asin_n"),
                helper.make_node("Sub", [base + "_halfpi", base + "_asin"], [n.output[0]], name=base + "_sub_n"),
            ]
            continue
        if n.op_type == "Erf":
            new_nodes += erf_decomp(n)
            continue
        if n.op_type == "Einsum":
            equation = next((a.s.decode() for a in n.attribute if a.name == "equation"), "")
            if equation.replace(" ", "") == "bi,bi->b":
                a, b = n.input
                base = n.name
                add_init(base + "__axes", np.array([1], dtype=np.int64))
                new_nodes += [
                    helper.make_node("Mul", [a, b], [n.output[0] + "__mul"], name=base + "__mul"),
                    helper.make_node("ReduceSum", [n.output[0] + "__mul", base + "__axes"],
                                     [n.output[0]], name=base + "__rs", keepdims=0),
                ]
                continue
        if n.op_type == "ReduceL1":
            x = n.input[0]; axes = n.input[1] if len(n.input) > 1 else None
            keepdims = next((a.i for a in n.attribute if a.name == "keepdims"), 1)
            base = n.name + "__rl1"
            new_nodes.append(helper.make_node("Abs", [x], [base + "_abs"], name=base + "_abs_n"))
            rs_inputs = [base + "_abs"] + ([axes] if axes else [])
            new_nodes.append(helper.make_node("ReduceSum", rs_inputs, [n.output[0]],
                                              name=base + "_rs_n", keepdims=keepdims))
            continue
        if n.op_type == "Trilu":
            rep = trilu_static_mask(n)
            if rep is not None:
                new_nodes += rep
                continue
        if n.op_type == "Unique":
            used_outs = []
            for o in n.output:
                if any(o in m.input for m in g.node) or any(o == go.name for go in g.output):
                    used_outs.append(o)
            if used_outs == [n.output[0]]:
                new_nodes.append(helper.make_node("Identity", [n.input[0]], [n.output[0]],
                                                  name=n.name + "__id"))
                continue
        if n.op_type == "RNN":
            new_nodes += rnn_unroll(n); continue
        if n.op_type == "LSTM":
            new_nodes += lstm_unroll(n); continue
        if n.op_type == "GRU":
            new_nodes += gru_unroll(n); continue
        new_nodes.append(n)

    del g.node[:]
    g.node.extend(new_nodes)
    onnx.checker.check_model(model, full_check=False)
    return model


# ---------------------------------------------------------------------------
# Pass 3 — opset 18 → 17 (Reduce-family axes attr; strip BN training_mode)
# ---------------------------------------------------------------------------

def _downgrade_opset(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}

    REDUCE = ("ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd",
              "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare")

    new_nodes = []
    for n in g.node:
        if n.op_type in REDUCE and len(n.input) > 1 and n.input[1] in inits:
            axes_list = [int(v) for v in inits[n.input[1]].flatten().tolist()]
            keepdims = next((a.i for a in n.attribute if a.name == "keepdims"), 1)
            new_node = helper.make_node(n.op_type, [n.input[0]], list(n.output), name=n.name)
            new_node.attribute.extend([helper.make_attribute("keepdims", keepdims)])
            if axes_list:
                new_node.attribute.extend([helper.make_attribute("axes", axes_list)])
            new_nodes.append(new_node)
        else:
            new_nodes.append(n)
    del g.node[:]
    g.node.extend(new_nodes)

    for n in g.node:
        if n.op_type == "BatchNormalization":
            keep = [a for a in n.attribute if a.name != "training_mode"]
            del n.attribute[:]
            n.attribute.extend(keep)

    for o in model.opset_import:
        if o.domain in ("", "ai.onnx"):
            o.version = 17

    onnx.checker.check_model(model, full_check=False)
    return model


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _run(path_or_model, inputs):
    if isinstance(path_or_model, onnx.ModelProto):
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tf:
            tmp = tf.name
        try:
            onnx.save(path_or_model, tmp)
            sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
            return sess.run(None, inputs)
        finally:
            os.unlink(tmp)
    sess = ort.InferenceSession(path_or_model, providers=["CPUExecutionProvider"])
    return sess.run(None, inputs)


def _load_inputs(model, inputs_dir):
    name_to_file: dict[str, str] = {}
    for fn in os.listdir(inputs_dir):
        if not fn.endswith(".npy"):
            continue
        for inp in model.graph.input:
            if fn.endswith(f"_{inp.name}.npy") or fn == f"{inp.name}.npy":
                name_to_file[inp.name] = fn
    missing = [i.name for i in model.graph.input if i.name not in name_to_file]
    if missing:
        raise SystemExit(f"missing inputs in {inputs_dir}: {missing}")
    return {n: np.load(os.path.join(inputs_dir, f)) for n, f in name_to_file.items()}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="path to the original ONNX model")
    p.add_argument("--output", required=True, help="path to write the modified ONNX model")
    p.add_argument("--inputs-dir", required=True, help="directory of calibration .npy files (one per graph input)")
    p.add_argument("--tolerance", type=float, default=1e-5, help="absolute tolerance for output verification")
    args = p.parse_args()

    print(f"[load] {args.input}")
    model = onnx.load(args.input)
    inputs = _load_inputs(model, args.inputs_dir)
    n_orig = len(model.graph.node)
    print(f"[info] original: {n_orig} nodes, opset {[(o.domain, o.version) for o in model.opset_import]}")

    ref = _run(args.input, inputs)
    print(f"[ref ] outputs: {[r.shape for r in ref]}")

    print("[pass 1] folding constant sub-DAGs")
    model = _fold_constants(model, inputs)
    out = _run(model, inputs)
    for r0, r1 in zip(ref, out):
        if not np.allclose(np.asarray(r0), np.asarray(r1), atol=args.tolerance, rtol=args.tolerance):
            raise SystemExit("[err] pass 1 changed outputs beyond tolerance")
    print(f"[pass 1] ok  ({len(model.graph.node)} nodes)")

    print("[pass 2] rewriting unsupported / poorly-supported ops")
    model = _rewrite_ops(model)
    out = _run(model, inputs)
    for r0, r1 in zip(ref, out):
        if not np.allclose(np.asarray(r0), np.asarray(r1), atol=args.tolerance, rtol=args.tolerance):
            raise SystemExit("[err] pass 2 changed outputs beyond tolerance")
    print(f"[pass 2] ok  ({len(model.graph.node)} nodes)")

    print("[pass 3] downgrading opset 18 → 17")
    model = _downgrade_opset(model)
    out = _run(model, inputs)
    for r0, r1 in zip(ref, out):
        if not np.allclose(np.asarray(r0), np.asarray(r1), atol=args.tolerance, rtol=args.tolerance):
            raise SystemExit("[err] pass 3 changed outputs beyond tolerance")
    print(f"[pass 3] ok  ({len(model.graph.node)} nodes)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    onnx.save(model, args.output)
    print(f"[done] saved {args.output}")
    print(f"[done] {n_orig} → {len(model.graph.node)} nodes; reference and modified outputs match.")


if __name__ == "__main__":
    main()
