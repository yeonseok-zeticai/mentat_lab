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
* **QNN / QAIRT** — `qairt-converter` runs twice: once with
  `--float_bitwidth 32` to produce `model.dlc` (reference, every op
  mapped) and once with `--float_bitwidth 16` to produce
  `model_fp16.dlc` (HTP-deployable). Nothing else: no CPU smoke run,
  no `qairt-quantizer` (INT8 calibration belongs to the deployment
  pipeline with real data), and no `snpe-dlc-graph-prepare` (offline
  HTP cache is rebuilt on-target by the deploying app with its own
  VTCM / thread / DLBC settings). The DLC alone is the deployment
  artifact; the on-device QNN runtime JIT-compiles to Hexagon at first
  inference.
* **CoreML** — `model.mlpackage` (MLProgram, FP16 or FP32, iOS17 minimum).
  `predict()` is macOS-only — the conversion side validates and saves;
  on-device verification is done from a Mac.

## Disk layout (host-specific)

The pipelines pull tens of GB per variant of safetensors and spill multi-GB
scratch during torch.export / coremltools / qairt-converter runs. On this
host (`/` is 3.9 TB but constantly near full) we keep all heavy storage on
`/mnt/disks/zeticai_database/`:

| path on `/` | symlink target (on `/mnt`) | what's there |
|---|---|---|
| `~/.cache/huggingface` | `/mnt/disks/zeticai_database/heavy_cache/huggingface` | `hf_hub_download` cache (~395 GB across all sapiens2 + LLM downloads) |
| `$TMPDIR` (forced by runner) | `/mnt/disks/zeticai_database/tmp_scratch` | torch.export / coremltools / qairt-converter scratch |
| (no symlink) | `/mnt/disks/zeticai_database/models/` | the per-variant artifact tree (`.pt2`, `.dlc`, `.mlpackage`, …) |

Re-create the symlink on a fresh machine before running anything heavy:

```bash
mkdir -p /mnt/disks/zeticai_database/heavy_cache /mnt/disks/zeticai_database/tmp_scratch
[ ! -e ~/.cache/huggingface ] && \
    ln -s /mnt/disks/zeticai_database/heavy_cache/huggingface ~/.cache/huggingface
```

`runner/build_variant.py` and `runner/run_qnn.sh` set
`TMPDIR=/mnt/disks/zeticai_database/tmp_scratch` automatically when that
directory exists.

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
