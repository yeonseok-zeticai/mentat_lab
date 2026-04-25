"""Build a PyTorch ExportedProgram and sample NPY input from
the OpenCV/MediaPipe pose-estimation ONNX model.

Pipeline:
  HF download (opencv/pose_estimation_mediapipe)
    -> ONNX source (NHWC input, 5 outputs)
    -> onnx2torch -> torch.fx.GraphModule
    -> torch.export.export -> ExportedProgram (.pt2)
    -> Sample input saved as .npy

Outputs land under MODEL_DIR (defaults to /mnt/disks/zeticai_database/...).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnx2torch
import torch
from huggingface_hub import hf_hub_download

from static_patch import patch_dynamic_ops, sanitize_module_names

REPO_ID = "opencv/pose_estimation_mediapipe"
ONNX_FILENAME = "pose_estimation_mediapipe_2023mar.onnx"
INPUT_SHAPE = (1, 256, 256, 3)  # NHWC float32 in [0, 1]
OUTPUT_NAMES = ["landmarks", "conf", "mask", "heatmap", "landmarks_word"]


def fetch_onnx(target_path: Path) -> Path:
    if target_path.exists():
        print(f"[fetch] reusing {target_path}")
        return target_path
    src = Path(hf_hub_download(repo_id=REPO_ID, filename=ONNX_FILENAME))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(src.read_bytes())
    print(f"[fetch] copied to {target_path}")
    return target_path


def build_sample_input(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=INPUT_SHAPE).astype(np.float32)


class PoseModel(torch.nn.Module):
    """Wrap onnx2torch's GraphModule so torch.export sees a clean nn.Module."""

    def __init__(self, onnx_path: str):
        super().__init__()
        gm = onnx2torch.convert(onnx_path)
        # Rename modules whose targets contain ';', '/', etc. (they break the
        # ExportedProgram serializer) and bake constant initializers into
        # Pad/Reshape/Resize so torch.export doesn't trip on data-dependent
        # guards in fake-tensor tracing.
        gm = sanitize_module_names(gm)
        self.gm = patch_dynamic_ops(gm)

    def forward(self, x: torch.Tensor):
        out = self.gm(x)
        # onnx2torch returns a tuple in graph order: matches OUTPUT_NAMES order.
        return out


def export_program(model: torch.nn.Module, sample: torch.Tensor) -> torch.export.ExportedProgram:
    print(f"[export] torch={torch.__version__}")
    ep = torch.export.export(model, (sample,), strict=False)
    return ep


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="/mnt/disks/zeticai_database/models/pose_estimation_mediapipe",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = fetch_onnx(model_dir / "source.onnx")

    # Inspect & record metadata so downstream tooling can read shapes/names.
    onnx_model = onnx.load(str(onnx_path))
    meta = {
        "repo_id": REPO_ID,
        "onnx_filename": ONNX_FILENAME,
        "ir_version": onnx_model.ir_version,
        "opset": [(o.domain or "ai.onnx", o.version) for o in onnx_model.opset_import],
        "input": {
            "name": onnx_model.graph.input[0].name,
            "shape": list(INPUT_SHAPE),
            "dtype": "float32",
            "layout": "NHWC",
            "value_range": "[0, 1] (RGB / 255)",
        },
        "outputs": [
            {
                "name": OUTPUT_NAMES[i],
                "onnx_name": o.name,
                "shape": [d.dim_value for d in o.type.tensor_type.shape.dim],
            }
            for i, o in enumerate(onnx_model.graph.output)
        ],
        "torch_version": torch.__version__,
    }
    (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[meta] wrote {model_dir / 'metadata.json'}")

    # Sample input as numpy (NHWC for ONNX/QNN parity).
    sample_np = build_sample_input(args.seed)
    np.save(model_dir / "sample_input.npy", sample_np)
    print(f"[sample] wrote {model_dir / 'sample_input.npy'} shape={sample_np.shape}")

    # Build PyTorch model and exported program.
    model = PoseModel(str(onnx_path)).eval()
    sample_t = torch.from_numpy(sample_np)

    with torch.inference_mode():
        ref_out = model(sample_t)
    if not isinstance(ref_out, (tuple, list)):
        ref_out = (ref_out,)
    for name, t in zip(OUTPUT_NAMES, ref_out):
        np.save(model_dir / f"sample_output_{name}.npy", t.detach().cpu().numpy())
        print(f"[sample] wrote sample_output_{name}.npy shape={tuple(t.shape)}")

    ep = export_program(model, sample_t)
    pt2_path = model_dir / "model.pt2"
    torch.export.save(ep, str(pt2_path))
    print(f"[export] wrote {pt2_path}")

    # Replay the saved program to make sure it round-trips.
    loaded = torch.export.load(str(pt2_path))
    with torch.inference_mode():
        loaded_out = loaded.module()(sample_t)
    if not isinstance(loaded_out, (tuple, list)):
        loaded_out = (loaded_out,)
    for name, ref_t, new_t in zip(OUTPUT_NAMES, ref_out, loaded_out):
        diff = (ref_t - new_t).abs().max().item()
        print(f"[verify] ExportedProgram[{name}] max|Δ|={diff:.3e}")

    print("[done] all artifacts written to", model_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
