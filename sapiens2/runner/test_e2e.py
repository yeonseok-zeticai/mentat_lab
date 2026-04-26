"""End-to-end numerical parity test for one sapiens2 variant.

For a given variant directory, loads:

  * sample_input.npy   — the canonical input the build pipeline saved
  * sample_output.npy  — PyTorch reference output (ground truth)

and runs each backend forward, comparing every output against the
saved reference:

  1. PyTorch ExportedProgram (`model.pt2`) — re-run from disk.
  2. ONNX Runtime CPU (`model.onnx`).
  3. (optional) QNN CPU backend (`model.dlc` via `qnn-net-run libQnnCpu.so`).

Tolerances:
  * pt2 round-trip is expected bit-exact (atol = 0).
  * ORT cross-runtime drift comes from kernel-level math differences
    (Conv accumulation order, softmax fast-paths) — atol = 1e-3.
  * QNN CPU is lower precision still — atol = 1e-2 by default.

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

PT2_ATOL = 0.0
ORT_ATOL = 1e-3
QNN_CPU_ATOL = 1e-2


def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def compare(label: str, ref: np.ndarray, got: np.ndarray, atol: float) -> dict:
    diff = np.abs(ref - got)
    rng = max(float(np.abs(ref).max()), 1e-9)
    res = {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "rel_to_range": float(diff.max() / rng),
        "atol": atol,
        "ok": bool(np.all(diff <= atol + 1e-7)),
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


def verify_pt2(out_dir: Path, sample: np.ndarray, ref: np.ndarray) -> dict:
    pt2 = out_dir / "model.pt2"
    if not pt2.exists():
        return {"status": "skipped", "reason": "model.pt2 missing"}
    ep = torch.export.load(str(pt2))
    with torch.inference_mode():
        out = ep.module()(torch.from_numpy(sample))
    if isinstance(out, (tuple, list)):
        out = out[0]
    res = compare("ExportedProgram", ref, _to_numpy(out), PT2_ATOL)
    return {"status": "ok" if res["ok"] else "fail", **res}


def verify_onnx(out_dir: Path, sample: np.ndarray, ref: np.ndarray) -> dict:
    onnx_path = out_dir / "model.onnx"
    if not onnx_path.exists():
        return {"status": "skipped", "reason": "model.onnx missing"}
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: sample})[0]
    res = compare("ONNX Runtime", ref, out, ORT_ATOL)
    return {"status": "ok" if res["ok"] else "fail", **res}


def verify_qnn_cpu(out_dir: Path, sample: np.ndarray, ref: np.ndarray) -> dict:
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
    if not raws:
        return {"status": "fail", "error": "no output raw written"}
    out = np.frombuffer(raws[0].read_bytes(), dtype=np.float32).reshape(ref.shape)
    cmp_res = compare("QNN CPU", ref, out, QNN_CPU_ATOL)
    return {"status": "ok" if cmp_res["ok"] else "fail", "output_file": raws[0].name, **cmp_res}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-dir", required=True)
    parser.add_argument("--skip-qnn", action="store_true",
                        help="skip the qnn-net-run CPU backend check (slow on ViT)")
    args = parser.parse_args()

    out_dir = Path(args.variant_dir)
    sample = np.load(out_dir / "sample_input.npy")
    ref = np.load(out_dir / "sample_output.npy")
    print(f"[e2e] {out_dir}  input={sample.shape}  ref={ref.shape} {ref.dtype}")

    meta = out_dir / "metadata.json"

    print("\n[1/3] PyTorch ExportedProgram ...")
    pt2_res = verify_pt2(out_dir, sample, ref)
    update_metadata(meta, "verify_pt2", pt2_res)

    print("\n[2/3] ONNX Runtime ...")
    onnx_res = verify_onnx(out_dir, sample, ref)
    update_metadata(meta, "verify_onnx", onnx_res)

    qnn_res: Optional[dict] = None
    if args.skip_qnn:
        qnn_res = {"status": "skipped", "reason": "--skip-qnn"}
    else:
        print("\n[3/3] QNN CPU backend ...")
        qnn_res = verify_qnn_cpu(out_dir, sample, ref)
    update_metadata(meta, "verify_qnn_cpu", qnn_res)

    failures = [k for k, r in {"pt2": pt2_res, "onnx": onnx_res, "qnn_cpu": qnn_res}.items()
                if r and r.get("status") == "fail"]
    print(f"\n[done] failures: {failures or 'none'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
