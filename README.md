# mentat_lab

Bundles for shipping HuggingFace vision models to on-device runtimes
(Qualcomm Hexagon HTP via QNN/QAIRT, Apple CoreML), plus the PyTorch
`ExportedProgram` and sample NPY I/O the rest of our toolchain consumes.

Each subdirectory is a self-contained pipeline: `python` to fetch and
re-export the model, shell scripts to drive the QNN tooling, and a
README that documents the I/O contract and reproduction steps.

## Layout

```
mentat_lab/
├── pose_estimation_mediapipe/  ← BlazePose pose-landmark, single ONNX
│                                  → onnx2torch → torch.export → QNN/HTP + CoreML
└── sapiens2/                   ← facebook/sapiens2 (21 variants: pretrain
    │                             + pose/seg/normal/pointmap × 4 sizes)
    │                             auto-cataloguer + per-variant builder
    └── runner/
        ├── catalog.py          ← walks HF index, writes per-variant manifest
        ├── build_variant.py    ← download → build → export → ONNX → QNN → CoreML
        ├── run_qnn.sh          ← qairt-converter + INT8 + HTP offline cache
        └── run_all.sh          ← loops every catalog entry smallest-first
```

Per-variant artifacts (model.pt2 / model.dlc / model.mlpackage / sample
NPYs / metadata) land under `/mnt/disks/zeticai_database/models/<name>/`,
not in this repo — they're large binary blobs that are reproducible by
running the scripts.

## Conventions every bundle follows

* **Sample input** — saved as `sample_input.npy` (NHWC for NPU-style
  models, NCHW for ViT-style). RNG-seeded so reference outputs are
  reproducible.
* **Reference outputs** — saved as `sample_output_<name>.npy`, one per
  output tensor, captured from the PyTorch `ExportedProgram`. Used as
  the cross-runtime parity baseline.
* **PyTorch ≥ 2.9 ExportedProgram** — `model.pt2`, bit-exact round-trip.
* **QNN / QAIRT** — FP32 `model.dlc` from `qairt-converter`, INT8
  `model_int8.dlc` (per-channel) from `qairt-quantizer`, and
  `model_int8_htp.dlc` with offline cache for Hexagon `v68/v73/v75/v79`
  built by `snpe-dlc-graph-prepare`. The successful prepare across all
  four archs is the deployment-readiness signal.
* **CoreML** — `model.mlpackage` (MLProgram, FP16 or FP32, iOS17 minimum).
  `predict()` is macOS-only — the conversion side validates and saves;
  on-device verification is done from a Mac.

## Repo hygiene

`qnn_244` is a host-specific symlink to the QNN SDK install
(`/opt/qcom/aistack/qnn/2.44.0.260225`); it is `.gitignore`d. Pull the
SDK from Qualcomm's developer portal on a new machine and recreate the
symlink before running the QNN pipelines.

`sapiens2/sapiens2/` is a sibling clone of
`github.com/facebookresearch/sapiens2` that the runner imports for the
ViT backbone + task-head modules. It is also `.gitignore`d — clone it
locally:

```bash
git clone --depth 1 https://github.com/facebookresearch/sapiens2.git \
    sapiens2/sapiens2
pip install -e sapiens2/sapiens2 --no-deps
```
