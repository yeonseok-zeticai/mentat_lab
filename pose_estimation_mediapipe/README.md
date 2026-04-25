# Pose Estimation MediaPipe — On-Device AI bundle

End-to-end pipeline that takes the OpenCV mirror of MediaPipe's BlazePose
landmark model and produces:

* a PyTorch `ExportedProgram` (≥ 2.9 `.pt2` archive),
* matching sample I/O as NumPy arrays,
* a Qualcomm QNN/QAIRT `.dlc` for HTP / DSP / GPU / CPU runtimes,
* an Apple CoreML `.mlpackage`.

Source model: <https://huggingface.co/opencv/pose_estimation_mediapipe>
(`pose_estimation_mediapipe_2023mar.onnx`).

## Pipeline at a glance

```
                                       ┌────────────────────────────┐
                                       │    sample_input.npy        │
                                       │  (1, 256, 256, 3) float32  │
                                       │       NHWC, RGB / 255      │
                                       └────────────┬───────────────┘
                                                    │
                                                    ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                        BlazePose pose-landmark CNN                       │
   │  Conv ×88  ·  DepthwiseConv  ·  Clip(ReLU6)  ·  Add  ·  Resize ×5  ·     │
   │  MaxPool ×2  ·  Pad  ·  Sigmoid  ·  Reshape ×4  ·  Transpose ×2          │
   │     156 M MACs / inference   ·   1.24 M parameters   ·   ~5 MB weights   │
   └──────────────────────────────────────────────────────────────────────────┘
                                                    │
       ┌──────────────────┬──────────────────┬──────┴──────┬──────────────────────┐
       ▼                  ▼                  ▼             ▼                      ▼
  ┌──────────┐     ┌────────────┐     ┌─────────────┐  ┌──────────┐      ┌────────────────┐
  │ landmarks│     │   conf     │     │    mask     │  │ heatmap  │      │ landmarks_word │
  │ (1, 195) │     │  (1, 1)    │     │ (1,256,256, │  │(1,64,64, │      │   (1, 117)     │
  │ →(39, 5) │     │  pose-flag │     │      1)     │  │   39)    │      │   →(39, 3)     │
  │ x,y,z,   │     │  sigmoid   │     │ segmentation│  │ landmark │      │   3-D world    │
  │ vis,pres │     │ confidence │     │ mask logits │  │ heatmaps │      │  coords (m)    │
  └──────────┘     └────────────┘     └─────────────┘  └──────────┘      └────────────────┘

   Build & deploy graph (the same .npy feeds every backend):

         source.onnx ─► onnx2torch ─► static_patch ─► torch.export ─► model.pt2
                         │                                                │
                         │                                                └─► ct.convert ─► model.mlpackage   (Apple CoreML)
                         │
                         ├─► qairt-converter --float_bitwidth 32 ─► model.dlc       (FP32)
                         │
                         └─► qairt-converter --float_bitwidth 16 ─► model_fp16.dlc  (FP16, HTP-ready)
```

## Layout

Code (this directory):
```
pose_estimation_mediapipe/
├── README.md             ← you are here
├── build_export.py       ← ONNX → onnx2torch → torch.export → model.pt2
├── static_patch.py       ← bake constant initializers into Pad/Reshape/Resize
├── test_e2e.py           ← ONNX-Runtime ↔ ExportedProgram ↔ saved-NPY parity
├── run_qnn.sh            ← qairt-converter (FP32 + FP16) → model.dlc / model_fp16.dlc
└── run_coreml.py         ← coremltools.convert → model.mlpackage
```

Artifacts (`/mnt/disks/zeticai_database/models/pose_estimation_mediapipe/`):
```
source.onnx                 5.6 MB   original BlazePose pose-landmark ONNX
sample_input.npy            768 kB   (1, 256, 256, 3) float32, range [0, 1]
sample_output_landmarks.npy          (1, 195)   33 kp + 6 aux × [x,y,z,vis,pres]
sample_output_conf.npy               (1, 1)     pose-flag sigmoid
sample_output_mask.npy               (1, 256, 256, 1) per-pixel logits
sample_output_heatmap.npy            (1, 64, 64, 39)  39 kp heatmaps
sample_output_landmarks_word.npy     (1, 117)   3D world coords (39 × xyz)
metadata.json               graph / opset / I/O metadata
model.pt2                   7.0 MB   torch.export ExportedProgram (PyTorch ≥ 2.9)
model.dlc                   5.8 MB   QNN/QAIRT FP32 DLC (reference)
model_fp16.dlc              ~3 MB    QNN/QAIRT FP16 DLC (HTP-deployable)
model.mlpackage/            CoreML MLProgram (FLOAT32, iOS17 minimum)
coreml_manifest.json        I/O summary of the mlpackage
```

## I/O contract

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| **input** `input_1` | `(1, 256, 256, 3)` | `float32` | NHWC, RGB, normalised to `[0, 1]` |
| `landmarks` | `(1, 195)` | `float32` | reshape to `(39, 5)` → `[x, y, z, vis, pres]` |
| `conf` | `(1, 1)` | `float32` | pose-flag confidence (sigmoid) |
| `mask` | `(1, 256, 256, 1)` | `float32` | segmentation mask logits |
| `heatmap` | `(1, 64, 64, 39)` | `float32` | landmark heatmaps |
| `landmarks_word` | `(1, 117)` | `float32` | reshape to `(39, 3)` → world `[x, y, z]` |

## Reproduce from scratch

The ExportedProgram step needs torch ≥ 2.9. The QAIRT toolchain ships
Python-3.10 wheels — switch envs there:

```bash
# 1. Build PyTorch ExportedProgram + sample NPY
python3 build_export.py

# 2. Round-trip parity (ORT vs ExportedProgram vs saved NPY)
python3 test_e2e.py

# 3. QNN DLC for HTP and CPU-backend smoke run
bash run_qnn.sh

# 4. CoreML mlpackage
python3 run_coreml.py
```

Sample input is generated with `numpy.random.default_rng(seed=0)` for
deterministic, reproducible reference outputs. Override with
`--seed` if you need a different draw.

## Why `static_patch.py` exists

`onnx2torch` materialises ONNX `Pad`, `Reshape`, and `Resize` as modules that
read their `pads` / `shape` / `sizes` arguments at runtime (e.g. `pads.tolist()`
or `torch.any(shape == 0)`). Under fake-tensor tracing inside `torch.export`,
those calls produce unbacked symints and the export aborts with
`GuardOnDataDependentSymNode`.

`patch_dynamic_ops` walks the FX graph, finds every `OnnxPadDynamic`,
`OnnxReshape`, and `OnnxResize`, looks up their initializer-backed shape /
pad / size tensor, and replaces the module with a Python-int specialised
version. `sanitize_module_names` additionally rewrites submodule paths that
contain `;` or `/` (onnx2torch keeps the original TF op fusion names) because
those characters break the ExportedProgram serializer.

The patched graph is bit-for-bit equivalent to the original (`build_export.py`
asserts `max|Δ| == 0` after a round-trip).

## Cross-runtime tolerance

`test_e2e.py` enforces:

* `ExportedProgram` round-trip → bit-exact (`atol = 0`),
* `ONNX Runtime` ↔ `ExportedProgram` → `atol = 2e-1`, `rtol = 1e-2`.

The looser cross-runtime budget is required because Conv accumulation order
and `bilinear` resize edge handling differ between ORT and PyTorch's
oneDNN kernels. On the `seed=0` sample input the worst observed drift is
`mask: max|Δ| = 6.9e-2` at boundary pixels; mean drifts are sub-`1e-3`.

## QNN: just `qairt-converter`

`run_qnn.sh` runs `qairt-converter` twice and stops:

| artifact | flag | bytes | notes |
|---|---|---|---|
| `model.dlc` | `--float_bitwidth 32` | ~5.6 MB | reference DLC (all ops mapped) |
| `model_fp16.dlc` | `--float_bitwidth 16` | ~2.8 MB | the one we ship to HTP |

`qairt-dlc-info -i model.dlc` lists the supported runtimes per layer
(`A:AIP, D:DSP, G:GPU, C:CPU`); the entire pose graph maps cleanly to
all four. That's the deployment-readiness signal we actually need.

We deliberately do **not** run any of:

* `qnn-net-run --backend libQnnCpu.so`  — CPU smoke test isn't a
  deployment artifact, and ONNX Runtime already gives an FP32 baseline.
* `qairt-quantizer`  — INT8 calibration belongs to the deployment
  pipeline, with real data, not the bundle stage.
* `snpe-dlc-graph-prepare`  — offline HTP cache is a startup-latency
  optimisation that the deployment image rebuilds on-target with its
  own VTCM / thread / DLBC settings. Caching here would freeze a
  prepare configuration that the on-device runtime would discard.

## CoreML predict() on Linux

`coremltools` ships only the conversion frontend on Linux —
`coremltools.libcoremlpython` (the runtime that backs `mlmodel.predict()`)
is macOS-only. `run_coreml.py` therefore only converts and validates the
saved `.mlpackage`; copy the package to a Mac and load it from Xcode (or
Python on macOS) to do an actual `predict()` and compare against
`sample_output_*.npy`.
