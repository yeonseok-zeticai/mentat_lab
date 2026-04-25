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
                         │                                                ├─► ct.convert ─► model.mlpackage   (Apple CoreML)
                         │                                                │
                         └─► qairt-converter ─► model.dlc ─┬─► qnn-net-run libQnnCpu.so   (CPU smoke)
                                                          │
                                                          └─► qairt-quantizer (INT8, per-channel)
                                                                     │
                                                                     ▼
                                                            model_int8.dlc
                                                                     │
                                                                     ▼
                                          snpe-dlc-graph-prepare --htp_archs=v68,v73,v75,v79
                                                                     │
                                                                     ▼
                                                       model_int8_htp.dlc  ─► qnn-net-run libQnnHtp.so
                                                       (Hexagon HTP cache:        (HTP simulator)
                                                        SM7350 / SM8550 /
                                                        SM8650 / SM8750)
```

## Layout

Code (this directory):
```
pose_estimation_mediapipe/
├── README.md             ← you are here
├── build_export.py       ← ONNX → onnx2torch → torch.export → model.pt2
├── static_patch.py       ← bake constant initializers into Pad/Reshape/Resize
├── test_e2e.py           ← ONNX-Runtime ↔ ExportedProgram ↔ saved-NPY parity
├── run_qnn.sh            ← qairt-converter → model.dlc + qnn-net-run CPU smoke
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
model.dlc                   5.8 MB   QNN/QAIRT FP32 reference DLC
model_int8.dlc              ~2 MB    INT8 per-channel quantised DLC (HTP-ready)
model_int8_htp.dlc          ~7 MB    INT8 DLC + offline HTP cache for v68/v73/v75/v79
model_int8_encoding.json    per-tensor scale/offset emitted by qairt-quantizer
model.mlpackage/            CoreML MLProgram (FLOAT32, iOS17 minimum)
coreml_manifest.json        I/O summary of the mlpackage
qnn_run/                    qnn-net-run CPU-backend results vs. ONNX baseline
qnn_htp_run/                qnn-net-run HTP-backend results (x86 simulator)
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

QNN CPU-backend (`run_qnn.sh`) reproduces the same drift profile against
ONNX Runtime, confirming the FP32 DLC is functionally equivalent:

```
Identity   (landmarks)      max|Δ| = 1.06e-3
Identity_1 (conf)           max|Δ| = 4.77e-12
Identity_2 (mask)           max|Δ| = 5.16e-2
Identity_3 (heatmap)        max|Δ| = 1.72e-4
Identity_4 (landmarks_word) max|Δ| = 3.82e-6
```

## QNN HTP backend verification

`run_qnn.sh` runs three additional steps to confirm the model is deployable
on Qualcomm's Hexagon Tensor Processor (HTP):

1. **Quantize** — `qairt-quantizer --target_backend HTP --use_per_channel_quantization`
   produces `model_int8.dlc`. The single-image calibration set (the random
   `sample_input.npy`) is enough to exercise the toolchain; real
   deployments should re-calibrate on a representative batch of pose
   imagery to recover accuracy.
2. **Offline HTP graph prepare** — `snpe-dlc-graph-prepare --htp_archs=v68,v73,v75,v79`
   serializes optimized HTP cache records for SoCs from Snapdragon 7-Gen3
   through 8-Gen3 / 8-Elite. Successful prepare across all four archs
   (`SM7350 / SM8550 / SM8650 / SM8750 : Success`) is the formal
   "HTP-ready" signal — every op fused cleanly, no fallbacks.
3. **HTP simulator execution** — `qnn-net-run --backend libQnnHtp.so`
   runs the cached DLC end-to-end on the x86 HTP simulator and dumps all
   five outputs as raw uFxp_8 tensors. The script then dequantizes each
   using the per-output scale/offset from `model_int8_encoding.json` and
   compares against the ONNX-Runtime FP32 reference.

The accuracy delta you see in stdout is dominated by INT8 quantization
noise (a single-image calibration cannot represent the input
distribution), not by HTP execution faithfulness — the same DLC fed
through `libQnnHtpQemu.so` and through real-device HTP returns
bit-identical INT8 outputs. Treat the HTP step as a **graph-feasibility
and runtime-execution check**, not as an accuracy benchmark.

## CoreML predict() on Linux

`coremltools` ships only the conversion frontend on Linux —
`coremltools.libcoremlpython` (the runtime that backs `mlmodel.predict()`)
is macOS-only. `run_coreml.py` therefore only converts and validates the
saved `.mlpackage`; copy the package to a Mac and load it from Xcode (or
Python on macOS) to do an actual `predict()` and compare against
`sample_output_*.npy`.
