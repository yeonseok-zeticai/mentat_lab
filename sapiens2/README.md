# sapiens2 — On-Device AI bundle for the full model family

End-to-end pipeline that takes the **`facebook/sapiens2`** HuggingFace
index and produces, for **every** advertised variant:

* a PyTorch `ExportedProgram` (≥ 2.9 `.pt2` archive),
* a sample input as `.npy` + the matching reference output captured
  from the same PyTorch graph,
* an ONNX export (with external-data when weights exceed 2 GB),
* Qualcomm QNN/QAIRT artifacts: FP32 `model.dlc`, INT8 per-channel
  `model_int8.dlc`, and `model_int8_htp.dlc` with offline cache for
  Hexagon `v68 / v73 / v75 / v79`,
* an Apple CoreML `model.mlpackage` (MLProgram, FP16, iOS17 minimum).

`facebook/sapiens2` is an **index** repo — it points to a fan-out of
21 sub-repos, ~161 GB of weights total. The runner walks the index,
resolves each sub-repo's files, then drives the per-variant pipeline.

Source family: <https://huggingface.co/facebook/sapiens2>
(architecture: <https://github.com/facebookresearch/sapiens2>).

## Pipeline at a glance

```
                        ┌────────────────────────────────┐
                        │       sample_input.npy         │
                        │      (1, 3, 1024, 768)         │
                        │   float32 NCHW, RGB / 255      │
                        └────────────────┬───────────────┘
                                         │
                                         ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                     Sapiens2 Vision Transformer (ViT)                    │
   │                                                                          │
   │   PatchEmbed (16×16) ──► +cls / +storage tokens ──► RoPE positions       │
   │                                                                          │
   │   ┌──────────────────────────────────────────────────────────────────┐   │
   │   │   N × Transformer Block (LayerNorm → SwiGLU FFN, GQA Attn)       │   │
   │   │   ┌────────────────────┐    ┌────────────────────┐               │   │
   │   │   │  Multi-head Attn   │    │      FFN (SwiGLU)  │               │   │
   │   │   │  WQ / WK / WV +    │    │      w12 -> w3     │               │   │
   │   │   │  q_norm, k_norm,   │ ── │       (4× embed)   │               │   │
   │   │   │  RoPE rotate, gamma│    │                    │               │   │
   │   │   └────────────────────┘    └────────────────────┘               │   │
   │   └──────────────────────────────────────────────────────────────────┘   │
   │                                                                          │
   │   Final LayerNorm ──► reshape tokens to (B, embed, H/16, W/16) feat-map  │
   └──────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                       ┌─────────────────┴────────────────┐
                       │                                  │
              (pretrain variant)                  (task variant)
                       │                                  │
                       ▼                                  ▼
              feature map                    Task-specific Head
              (B, embed, 64, 48)               ┌─────────────────────┐
              ───────────────                  │ pose: PoseHeatmap   │ → (B, 308, H, W)
              (back-bone alone,                │ seg:  SegHead       │ → (B, num_classes, H, W)
              no task head)                    │ normal: NormalHead  │ → (B, 3,  H, W)
                                               │ pointmap: PointHead │ → (B, 3,  H, W)
                                               └─────────────────────┘

   Build & deploy graph (the same .npy feeds every backend):

   HF index facebook/sapiens2
        │
        ├── catalog.py ──► per-variant manifest.json + INDEX.md (21 entries, 161 GB)
        │
        └── build_variant.py
                │
                ├── hf_hub_download ─► <variant>.safetensors
                │
                ├── Sapiens2 backbone + (head from sapiens config)
                │       │
                │       └── load_state_dict(...)
                │
                ├── torch.export.export ─► model.pt2  (bit-exact round-trip)
                │
                ├── torch.onnx.export ─► model.onnx (+ .data when >2 GB)
                │
                ├── ct.convert ─────► model.mlpackage  (Apple CoreML)
                │
                └── run_qnn.sh
                        │
                        ├── qairt-converter ─► model.dlc
                        │
                        └── qairt-quantizer (INT8 per-channel)
                                │
                                ▼
                          model_int8.dlc
                                │
                                ▼
              snpe-dlc-graph-prepare --htp_archs=v68,v73,v75,v79
                                │
                                ▼
                       model_int8_htp.dlc
                       (HTP offline cache: SM7350 / SM8550 / SM8650 / SM8750)
```

## What's in the catalog

`facebook/sapiens2` advertises **21 variants** across two families:

```
Backbones (no task head):
  pretrain-0.1b   pretrain-0.4b   pretrain-0.8b   pretrain-1b   pretrain-5b

Task heads (pose / seg / normal / pointmap), each at 4 sizes:
  *-0.4b   *-0.8b   *-1b   *-5b
```

`runner/catalog.py` resolves all of them via the HF API, downloads each
sub-repo's `README.md`, and writes:

* `models/sapiens2/INDEX.md` — single-table summary of every variant,
* `models/sapiens2/<task>_<size>/manifest.json` — file paths, byte
  sizes, LFS SHAs for the variant.

| family | weight footprint per size | configs in upstream repo |
|---|---|---|
| `pretrain` | 0.46 / 1.58 / 3.26 / 5.82 / 20.27 GB | none — built from `Sapiens2(arch=...)` directly |
| `pose` | 1.70 / 3.39 / 6.08 / 20.48 GB | `keypoints308_shutterstock_goliath_3po-1024x768` |
| `seg` | 1.63 / 3.31 / 5.88 / 20.36 GB | `seg_shutterstock_goliath-1024x768` |
| `normal` | 1.81 / 3.54 / 6.16 / 21.27 GB | `normal_metasim_render_people-1024x768` |
| `pointmap` | 2.11 / 3.87 / 6.52 / 21.39 GB | `pointmap_render_people-1024x768` |

Total: **~161 GB** of weights when every variant is materialised.

## I/O contract (every variant)

Sapiens2 is uniformly 1024×768 / patch 16, so the input layout never
changes; only the output head differs.

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| **input** | `(1, 3, 1024, 768)` | `float32` | NCHW, RGB / 255, ImageNet-style normalised by the upstream `data_preprocessor` (kept outside the exported graph for QNN/CoreML compatibility) |
| `pretrain` output | `(1, embed, 64, 48)` | `float32` | feature map; `embed` ∈ {768, 1024, 1280, 1536, 2432} |
| `pose` output | `(1, 308, H, W)` | `float32` | heatmaps for the 308-keypoint Sociopticon format |
| `seg` output | `(1, num_classes, H, W)` | `float32` | per-pixel logits over body parts |
| `normal` output | `(1, 3, H, W)` | `float32` | unit surface-normal vectors |
| `pointmap` output | `(1, 3, H, W)` | `float32` | dense 3-D point correspondence per pixel |

The upstream task heads upsample with two `ConvTranspose2d` blocks +
three `Conv2d` blocks, so output `(H, W)` is the input `(1024, 768)` for
pose / seg / normal / pointmap variants.

## Why `pos_embed_rope_dtype="fp32"` is forced

Sapiens2's RoPE module materialises its sinusoidal periods in
`bfloat16` by default. coremltools' torch frontend has no bfloat16 entry
in `TORCH_DTYPE_TO_NUM` and bails immediately, and QNN's
`qairt-converter` falls back any bf16 op to CPU. The runner passes
`pos_embed_rope_dtype="fp32"` to both the standalone backbone (pretrain
variants) and the task config dict (everything else); RoPE is < 1 % of
backbone activation memory, so the precision tradeoff is irrelevant.

The `state_dict` for a pretrain checkpoint has no `backbone.` prefix
(the keys live directly under the ViT root), so the runner loads
weights *before* wrapping the backbone in `WrapBackboneOutput`, which
just selects `backbone(x)[0]`.

## Reproduce a single variant

```bash
# 1. Make sure the upstream architecture clone exists (one-time setup)
git clone --depth 1 https://github.com/facebookresearch/sapiens2.git \
    sapiens2/sapiens2
pip install -e sapiens2/sapiens2 --no-deps

# 2. Build the catalog (cheap, no weights yet)
python3 runner/catalog.py
cat /mnt/disks/zeticai_database/models/sapiens2/INDEX.md

# 3. Process one variant end-to-end
python3 runner/build_variant.py --task pretrain --size 0.1b
python3 runner/build_variant.py --task pose --size 0.4b

# 4. Re-run only specific stages
python3 runner/build_variant.py --task pose --size 0.4b \
    --skip download,build_and_load,torch_export,onnx_export
```

`metadata.json` is updated **after every stage**, so partial state is
durable; killing and re-running just re-validates already-completed
stages instead of starting over.

## Reproduce all 21 variants

```bash
# Sequential, smallest-first; appends to RESULTS.md per variant.
bash runner/run_all.sh

# Subset:
VARIANTS="pretrain:0.1b pose:0.4b seg:0.4b" bash runner/run_all.sh

# Skip an expensive stage globally:
SKIP_STAGES=qnn_convert bash runner/run_all.sh
```

Wall-time scaling is dominated by `snpe-dlc-graph-prepare` (HTP offline
cache for 4 archs), which is roughly linear in MACs:

| size | params | end-to-end / variant | × 5 sizes |
|---|---|---|---|
| 0.1B | 0.114 B | ~17 min | (only one) |
| 0.4B | 0.398 B | ~1 h | × 5 |
| 0.8B | 0.818 B | ~2 h | × 5 |
| 1B | 1.46 B | ~3 h | × 5 |
| 5B | 5.07 B | ~10 h | × 5 |

A full sweep is **~3 days** of wall time. The orchestrator tolerates
failure, persists progress, and writes one row of `RESULTS.md` per
variant — partial state is always usable.

## QNN HTP backend verification

The same three-step QNN pipeline as `pose_estimation_mediapipe`:

1. **Convert** — `qairt-converter --float_bitwidth 32 --source_model_input_layout NCHW`
   produces FP32 `model.dlc`. Sapiens2 is mostly Conv2d (patch embed),
   matmul (attn / FFN), softmax, RMSNorm-equivalent and gather (RoPE
   rotate); all map cleanly to QNN ops.
2. **Quantize** — `qairt-quantizer --target_backend HTP --use_per_channel_quantization`
   produces `model_int8.dlc` with `model_int8_encoding.json` (per-tensor
   scale/offset). Single-image calibration with `sample_input.npy` is
   enough to exercise the toolchain; production deployments should
   re-calibrate with a representative batch of images.
3. **Offline HTP graph prepare** — `snpe-dlc-graph-prepare --htp_archs=v68,v73,v75,v79`
   serializes optimized HTP cache records for SoCs from
   Snapdragon 7-Gen3 through 8-Gen3 / 8-Elite. Successful prepare across
   all four archs (`SM7350 / SM8550 / SM8650 / SM8750 : Success`) is the
   formal "HTP-ready" signal — every op fused cleanly, no fallbacks.

The runner intentionally **skips the x86 HTP simulator step** that is
used for `pose_estimation_mediapipe`. ViT-scale models take ~15 min /
inference at the 0.1 B size on the simulator and **hours** at 5 B; the
prepare-success-across-archs signal is the actual deployment gate.
On-device validation happens after the artifacts are deployed to the
target SoC, not as part of this offline build.

## CoreML

Direct conversion from the `ExportedProgram` after
`run_decompositions({})` to drop the TRAINING-dialect ops coremltools
can't consume. Default precision is FP16 (`compute_precision=ct.precision.FLOAT16`)
because Apple Neural Engine prefers it; pass `--compute-precision FLOAT32`
on the command line if you need bit-parity testing on a Mac. `predict()`
is macOS-only — `coremltools.libcoremlpython` is not bundled on Linux —
so `run_coreml.py` only converts and verifies the `.mlpackage` reloads;
move the package to a Mac to do an actual `predict()` against
`sample_output_*.npy`.

## Layout

Code (this directory):

```
sapiens2/
├── README.md                      ← you are here
├── runner/
│   ├── catalog.py                 ← HF index → manifest + INDEX.md
│   ├── build_variant.py           ← per-variant end-to-end pipeline
│   ├── run_qnn.sh                 ← qairt-converter → DLC → INT8 → HTP cache
│   └── run_all.sh                 ← loop over every catalog entry
└── sapiens2/                      ← (gitignored) clone of facebookresearch/sapiens2
```

Artifacts (`/mnt/disks/zeticai_database/models/sapiens2/`):

```
INDEX.md                           ← catalog summary, all 21 variants
catalog.json                       ← machine-readable form of INDEX.md
RESULTS.md                         ← per-variant pipeline status (auto-updated)

<task>_<size>/                     ← one directory per variant, e.g. pretrain_0_1b/
├── manifest.json                  ← HF file/sha listing (cheap, written by catalog.py)
├── source_README.md               ← copy of the variant's HF README
├── sapiens2_<size>_<task>.safetensors   ← downloaded weights
├── sample_input.npy               ← (1, 3, 1024, 768) float32, RNG-seeded
├── sample_output.npy              ← reference output from the PyTorch graph
├── model.pt2                      ← torch.export ExportedProgram
├── model.onnx (+ .data)           ← ONNX export (external data when >2 GB)
├── model.dlc                      ← QNN/QAIRT FP32 DLC
├── model_int8.dlc                 ← INT8 per-channel quantised DLC
├── model_int8_encoding.json       ← per-tensor scale/offset
├── model_int8_htp.dlc             ← INT8 + HTP offline cache (v68/v73/v75/v79)
├── model.mlpackage/               ← CoreML MLProgram (FP16, iOS17 min)
├── metadata.json                  ← per-stage status + I/O contract
└── build.log                      ← full pipeline stdout/stderr
```
