"""Convert the pose ExportedProgram to a CoreML mlpackage and verify outputs.

Pipeline:
  model.pt2 (ExportedProgram, NHWC float32 input)
    --coremltools.convert(source='pytorch')-> Core ML MLProgram
    -> save model.mlpackage
    -> sanity-check tensor I/O metadata (libcoremlpython is unavailable on
       Linux so we can't actually predict()).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

INPUT_NAME = "input_1"
OUTPUT_NAMES = ["landmarks", "conf", "mask", "heatmap", "landmarks_word"]
INPUT_SHAPE = (1, 256, 256, 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="/mnt/disks/zeticai_database/models/pose_estimation_mediapipe",
    )
    args = parser.parse_args()
    md = Path(args.model_dir)

    pt2 = md / "model.pt2"
    print(f"[coreml] loading {pt2}")
    ep = torch.export.load(str(pt2))
    # coremltools only consumes ATEN/EDGE dialects; saved ExportedPrograms
    # default to the higher-level TRAINING dialect, so decompose first.
    ep = ep.run_decompositions({})

    # coremltools accepts an ExportedProgram directly; declare the float32
    # NHWC input tensor.
    inputs = [ct.TensorType(name=INPUT_NAME, shape=INPUT_SHAPE, dtype=np.float32)]
    outputs = [ct.TensorType(name=n, dtype=np.float32) for n in OUTPUT_NAMES]

    print("[coreml] coremltools.convert(...)")
    mlmodel = ct.convert(
        ep,
        source="pytorch",
        inputs=inputs,
        outputs=outputs,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        # Target a recent iOS so MLProgram is supported and the latest ops can be emitted.
        minimum_deployment_target=ct.target.iOS17,
    )

    # Inspect the MLProgram's I/O metadata.
    spec = mlmodel.get_spec()
    desc = {
        "model_type": spec.WhichOneof("Type"),
        "inputs": [
            {
                "name": i.name,
                "shape": list(i.type.multiArrayType.shape),
                "dtype": int(i.type.multiArrayType.dataType),
            }
            for i in spec.description.input
        ],
        "outputs": [
            {
                "name": o.name,
                "shape": list(o.type.multiArrayType.shape),
                "dtype": int(o.type.multiArrayType.dataType),
            }
            for o in spec.description.output
        ],
    }
    print("[coreml] spec summary:")
    print(json.dumps(desc, indent=2))

    out_path = md / "model.mlpackage"
    if out_path.exists():
        # ct.models.MLModel.save expects a non-existent destination for mlpackage.
        import shutil
        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))
    print(f"[coreml] wrote {out_path}")

    # Sanity-check the saved package re-loads.
    reloaded = ct.models.MLModel(str(out_path))
    print("[coreml] reloaded mlpackage:", reloaded.get_spec().description.metadata.shortDescription or "(no description)")

    # libcoremlpython is unavailable on Linux, so we can't predict().
    # Instead persist a JSON manifest that downstream tooling / on-device
    # tests can use to feed sample_input.npy and assert the same output shapes.
    manifest = {
        "input": {"name": INPUT_NAME, "shape": list(INPUT_SHAPE), "dtype": "float32"},
        "outputs": [
            {
                "name": OUTPUT_NAMES[i],
                "shape": [int(d) for d in spec.description.output[i].type.multiArrayType.shape],
            }
            for i in range(len(OUTPUT_NAMES))
        ],
        "deployment_target": "iOS17",
        "compute_precision": "float32",
        "verification_note": (
            "libcoremlpython is missing on Linux — predict() cannot run here. "
            "Run scripts/verify_coreml_macos.py on macOS to compare predict() "
            "output against sample_output_*.npy."
        ),
    }
    (md / "coreml_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[coreml] wrote {md / 'coreml_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
