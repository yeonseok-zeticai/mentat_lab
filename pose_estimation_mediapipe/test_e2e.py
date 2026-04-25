"""End-to-end numerical parity test:

  ONNX Runtime  vs.  PyTorch ExportedProgram  vs.  saved sample outputs

Tolerances:
  * ExportedProgram round-trip is expected to be **bit-exact** (self-consistency).
  * Cross-runtime (ONNX Runtime ↔ PyTorch) drift comes from kernel-level math
    differences (Conv accumulation order, bilinear-resize edge handling) and is
    measured against a looser ``atol=2e-1`` / ``rtol=1e-2`` budget.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

OUTPUT_NAMES = ["landmarks", "conf", "mask", "heatmap", "landmarks_word"]

# Cross-runtime budgets (loose - kernel-level drift is expected).
CROSS_ATOL = 2e-1
CROSS_RTOL = 1e-2

# Self-consistency budget (saved NPY <-> ExportedProgram should be bit-exact).
SELF_ATOL = 0.0
SELF_RTOL = 0.0


def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def compare(label: str, a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> bool:
    diff = np.abs(a - b)
    rel = diff / (np.abs(a) + 1e-9)
    ok = np.allclose(a, b, atol=atol, rtol=rtol)
    print(
        f"  [{label:>20s}] max|Δ|={diff.max():.3e}  mean|Δ|={diff.mean():.3e}  "
        f"max-rel={rel.max():.3e}  ok={ok}"
    )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="/mnt/disks/zeticai_database/models/pose_estimation_mediapipe",
    )
    args = parser.parse_args()
    md = Path(args.model_dir)

    sample = np.load(md / "sample_input.npy")
    print(f"sample input: shape={sample.shape} dtype={sample.dtype}")

    print("\n[1/3] ONNX Runtime forward ...")
    sess = ort.InferenceSession(str(md / "source.onnx"), providers=["CPUExecutionProvider"])
    onnx_outs = sess.run(None, {sess.get_inputs()[0].name: sample})
    onnx_map = dict(zip(OUTPUT_NAMES, onnx_outs))

    print("\n[2/3] PyTorch ExportedProgram forward ...")
    ep = torch.export.load(str(md / "model.pt2"))
    with torch.inference_mode():
        ep_outs = ep.module()(torch.from_numpy(sample))
    ep_map = dict(zip(OUTPUT_NAMES, [_to_numpy(t) for t in ep_outs]))

    print("\n[3/3] Saved-NPY reference outputs ...")
    saved_map = {
        name: np.load(md / f"sample_output_{name}.npy") for name in OUTPUT_NAMES
    }

    failures: list[str] = []

    print(f"\nONNX vs PyTorch ExportedProgram (atol={CROSS_ATOL}, rtol={CROSS_RTOL}):")
    for name in OUTPUT_NAMES:
        if not compare(name, onnx_map[name], ep_map[name], CROSS_ATOL, CROSS_RTOL):
            failures.append(f"onnx-vs-ep:{name}")

    print(f"\nPyTorch ExportedProgram vs saved NPY (atol={SELF_ATOL}, rtol={SELF_RTOL}):")
    for name in OUTPUT_NAMES:
        if not compare(name, ep_map[name], saved_map[name], SELF_ATOL, SELF_RTOL):
            failures.append(f"ep-vs-saved:{name}")

    if failures:
        print(f"\nFAILED: {failures}")
        return 1
    print("\nAll outputs match within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
