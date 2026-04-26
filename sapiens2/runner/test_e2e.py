"""End-to-end numerical parity test for one sapiens2 variant.

For a given variant directory, loads:

  * sample_input.npy   — the canonical input the build pipeline saved
  * sample_output.npy  — PyTorch reference output (ground truth)

and runs each backend forward, comparing every output against the
saved reference:

  1. PyTorch ExportedProgram (`model.pt2`) — re-run from disk.
  2. ONNX Runtime CPU (`model.onnx`).
  3. (optional) QNN CPU backend (`model.dlc` via `qnn-net-run libQnnCpu.so`).

Tolerances are *relative-to-range* (`max|Δ| / max|ref|`) — sapiens2 task
heads return unnormalised logits with dynamic ranges from ±1 (pose
heatmaps) up to ±1500 (seg logits), so absolute thresholds would
falsely flag low-relative drift on the high-range outputs:

  * pt2 round-trip is expected bit-exact (rel = 0).
  * ORT cross-runtime drift from kernel-level math differences
    (Conv accumulation order, softmax fast-paths) — rel = 1e-3.
  * QNN CPU is lower precision still — rel = 1e-2 by default.

Records numbers under ``stages.verify_*`` in ``metadata.json`` so the
parent runner can roll the result up to ``RESULTS.md``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Relative-to-range thresholds: max|Δ| divided by the largest |ref| value.
# Sapiens2 task heads return unnormalised logits (seg goes to ±1500, normal
# to ±80), so absolute tolerances would falsely flag low-relative drift.
PT2_RTOL = 0.0      # ExportedProgram round-trip is bit-exact
ORT_RTOL = 1e-3     # ONNX-Runtime cross-runtime float drift
QNN_CPU_RTOL = 1e-2 # QNN CPU backend, looser still


def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def compare(label: str, ref: np.ndarray, got: np.ndarray, rtol: float) -> dict:
    diff = np.abs(ref - got)
    rng = max(float(np.abs(ref).max()), 1e-9)
    rel = float(diff.max() / rng)
    res = {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "rel_to_range": rel,
        "rtol": rtol,
        "ok": bool(rel <= rtol + 1e-9),
    }
    print(
        f"  [{label:>14s}] max|Δ|={res['max_abs_diff']:.3e}  "
        f"mean|Δ|={res['mean_abs_diff']:.3e}  "
        f"rel-to-range={res['rel_to_range']:.3e}  ok={res['ok']}"
    )
    return res


def update_metadata(meta_path: Path, key: str, payload: dict) -> None:
    data = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    data.setdefault("stages", {})[key] = payload
    meta_path.write_text(json.dumps(data, indent=2))


def _as_tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x,)


def _summarise_multi(per_output: list[dict], rtol: float) -> dict:
    rels = [r["rel_to_range"] for r in per_output]
    abss = [r["max_abs_diff"] for r in per_output]
    return {
        "status": "ok" if all(r["ok"] for r in per_output) else "fail",
        "per_output": per_output,
        "max_abs_diff": float(max(abss)),
        "mean_abs_diff": float(np.mean([r["mean_abs_diff"] for r in per_output])),
        "rel_to_range": float(max(rels)),
        "rtol": rtol,
        "ok": all(r["ok"] for r in per_output),
    }


def verify_pt2(out_dir: Path, sample: np.ndarray, refs: tuple) -> dict:
    pt2 = out_dir / "model.pt2"
    if not pt2.exists():
        return {"status": "skipped", "reason": "model.pt2 missing"}
    ep = torch.export.load(str(pt2))
    with torch.inference_mode():
        outs = _as_tuple(ep.module()(torch.from_numpy(sample)))
    per = [
        compare(f"ExportedProgram[{i}]", r, _to_numpy(o), PT2_RTOL)
        for i, (r, o) in enumerate(zip(refs, outs))
    ]
    return _summarise_multi(per, PT2_RTOL)


def verify_onnx(out_dir: Path, sample: np.ndarray, refs: tuple) -> dict:
    onnx_path = out_dir / "model.onnx"
    if not onnx_path.exists():
        return {"status": "skipped", "reason": "model.onnx missing"}
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    outs = sess.run(None, {name: sample})
    per = [
        compare(f"ONNX Runtime[{i}]", r, o, ORT_RTOL)
        for i, (r, o) in enumerate(zip(refs, outs))
    ]
    return _summarise_multi(per, ORT_RTOL)


def verify_qnn_cpu(out_dir: Path, sample: np.ndarray, refs: tuple) -> dict:
    """Run model.dlc through QNN's CPU backend and compare.

    Routes through the py310 conda env where qairt-quantizer/qnn-net-run live;
    that env activation costs ~0.5s vs. minutes of qnn-net-run, so we always
    spawn a subshell rather than try to fork the QAIRT runtime in-process.
    """
    dlc = out_dir / "model.dlc"
    if not dlc.exists():
        return {"status": "skipped", "reason": "model.dlc missing"}

    rd = out_dir / "verify_qnn"
    rd.mkdir(exist_ok=True)
    sample.tofile(rd / "input.raw")
    (rd / "input_list.txt").write_text(f"input:={rd / 'input.raw'}\n")

    sdk = os.environ.get("QAIRT_SDK_ROOT", "/opt/qcom/aistack/qnn/2.44.0.260225")
    cmd = (
        "source /home/yeonseok/miniconda3/etc/profile.d/conda.sh && "
        "conda activate py310 && "
        f"export PATH={sdk}/bin/x86_64-linux-clang:$PATH && "
        f"export LD_LIBRARY_PATH={sdk}/lib/x86_64-linux-clang:${{LD_LIBRARY_PATH:-}} && "
        f"qnn-net-run --backend {sdk}/lib/x86_64-linux-clang/libQnnCpu.so "
        f"--dlc_path {dlc} --input_list {rd}/input_list.txt --output_dir {rd}/output"
    )
    res = subprocess.run(["bash", "-eo", "pipefail", "-c", cmd], capture_output=True, text=True)
    if res.returncode != 0:
        return {"status": "fail", "error": (res.stderr or res.stdout)[-400:]}

    # qnn-net-run names the output file after the ONNX output tensor; pick whatever raw landed.
    raws = sorted((rd / "output" / "Result_0").glob("*.raw"))
    if len(raws) < len(refs):
        return {"status": "fail", "error": f"only {len(raws)} output raws written, need {len(refs)}"}
    per = []
    for i, (raw, r) in enumerate(zip(raws, refs)):
        out = np.frombuffer(raw.read_bytes(), dtype=np.float32).reshape(r.shape)
        per.append(compare(f"QNN CPU[{i}]", r, out, QNN_CPU_RTOL))
    return _summarise_multi(per, QNN_CPU_RTOL)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-dir", required=True)
    parser.add_argument("--skip-qnn", action="store_true",
                        help="skip the qnn-net-run CPU backend check (slow on ViT)")
    args = parser.parse_args()

    out_dir = Path(args.variant_dir)
    sample = np.load(out_dir / "sample_input.npy")
    # Single-output variants store sample_output.npy; multi-output variants
    # (e.g. pointmap) store sample_output_<i>.npy in graph order.
    refs: tuple
    single = out_dir / "sample_output.npy"
    if single.exists():
        refs = (np.load(single),)
    else:
        multi_files = sorted(out_dir.glob("sample_output_*.npy"))
        if not multi_files:
            print(f"[e2e] no sample outputs in {out_dir}")
            return 1
        refs = tuple(np.load(f) for f in multi_files)
    print(f"[e2e] {out_dir}  input={sample.shape}  outputs={len(refs)}: "
          f"{[r.shape for r in refs]}")

    meta = out_dir / "metadata.json"

    print("\n[1/3] PyTorch ExportedProgram ...")
    pt2_res = verify_pt2(out_dir, sample, refs)
    update_metadata(meta, "verify_pt2", pt2_res)

    print("\n[2/3] ONNX Runtime ...")
    onnx_res = verify_onnx(out_dir, sample, refs)
    update_metadata(meta, "verify_onnx", onnx_res)

    qnn_res: Optional[dict] = None
    if args.skip_qnn:
        qnn_res = {"status": "skipped", "reason": "--skip-qnn"}
    else:
        print("\n[3/3] QNN CPU backend ...")
        qnn_res = verify_qnn_cpu(out_dir, sample, refs)
    update_metadata(meta, "verify_qnn_cpu", qnn_res)

    failures = [k for k, r in {"pt2": pt2_res, "onnx": onnx_res, "qnn_cpu": qnn_res}.items()
                if r and r.get("status") == "fail"]
    print(f"\n[done] failures: {failures or 'none'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
