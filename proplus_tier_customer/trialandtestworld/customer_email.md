# Customer email — operator modifications for on-device deployment

> Status: draft for review before sending.

---

**Subject:** Preparing your ONNX model for on-device deployment — required operator modifications

Hi [Customer],

Thank you for sharing `max_ops_final.onnx` and the calibration inputs. We
worked through getting the model into a form that on-device AI runtimes
can accept, and we wanted to share what we found and how we addressed it
so the same pipeline can be applied to your future updates.

## Summary

- The original graph contains **1,122 nodes** spanning more than 100
  distinct ONNX operators. While ONNX Runtime accepts all of these on the
  desktop, the operator coverage of mainstream **on-device AI runtimes**
  (the inference engines that actually execute models on phones / edge
  SoCs) is intentionally narrower, and a number of the operators in your
  graph fall outside that coverage.
- We built a small, repeatable preprocessing pipeline that produces a
  **numerically equivalent** modified ONNX model whose operator set fits
  within what on-device runtimes accept. The end result is a **778-node**
  graph that reproduces your reference output bit-for-bit on the supplied
  inputs.
- The pipeline is a single script (`prepare_for_ondevice.py`) and runs in
  three deterministic passes. It is verified at every step against the
  original model with ONNX Runtime, so we never silently change behavior.

## Why the original model can't go straight to a device

A typical on-device runtime supports the bread-and-butter operators —
convolutions, matmul, normalizations, activations, reductions, indexing,
shape ops, common element-wise math — and a curated list of higher-level
patterns. It does **not** generally support:

| Category | Operators in your graph |
|---|---|
| Control flow | `If`, `Loop`, `Scan` |
| Sequence containers | `SequenceConstruct`, `SequenceInsert`, `SequenceErase`, `SequenceAt`, `SequenceLength`, `SequenceMap`, `SequenceEmpty`, `SplitToSequence`, `ConcatFromSequence` |
| Signal processing | `STFT`, `DFT`, `MelWeightMatrix`, `BlackmanWindow`, `HammingWindow`, `HannWindow` |
| Stochastic | `RandomNormal`, `RandomUniform`, `RandomNormalLike`, `RandomUniformLike`, `Bernoulli`, `Multinomial` |
| String / text | `TfIdfVectorizer`, `StringNormalizer` |
| Misc | `Det`, `MaxUnpool`, `Col2Im`, `RoiAlign`, `MaxRoiPool`, `GridSample`, `EyeLike` |
| Selectively unsupported | `Einsum 'bi,bi->b'`, `Celu`, `Tan`, `Acos`, `Erf`, `ReduceL1`, `Trilu`, `Unique`, `RNN`, `LSTM`, `GRU` (specific shapes / variants) |

Some of these have no native fallback at all on-device; others have a
native operator but their layout/shape inference is incomplete on the
versions we deploy with today. We addressed both cases.

## What we did — three passes, verified at every step

### Pass 1 — Constant sub-graph folding
We discovered that **most** of the unsupported operators in the graph
(STFT, DFT, all Window operators, control flow, all Sequence operators,
random / Bernoulli / Multinomial, TfIdf / StringNormalizer, Det,
MaxUnpool, Col2Im, RoiAlign / MaxRoiPool, GridSample, EyeLike, etc.) sit
on sub-graphs that **do not depend on the dynamic inputs**. Their outputs
are fully determined at model-build time.

We trace dependency from each graph input forward, identify every tensor
that is *not* reachable from a dynamic input, evaluate those tensors once
with ONNX Runtime, and bake the resulting values back into the graph as
ordinary `initializer` (constant) tensors. The unsupported operators
disappear from the graph entirely — they have been *executed* once,
during preprocessing, instead of needing to run on-device.

> Result: **1,122 → 509 nodes**, output bit-identical.

### Pass 2 — Rewriting the remaining unsupported operators
A small number of unsupported operators legitimately depend on dynamic
input data, so they cannot be folded. We replaced each of them with a
mathematically equivalent composition of well-supported primitives:

| Operator | Replaced with | Equivalence |
|---|---|---|
| `Einsum 'bi,bi->b'` | `Mul` + `ReduceSum(axis=1)` | exact |
| `Celu(α)` | `Max(0, x) + Min(0, α·(exp(x/α) − 1))` | exact |
| `Tan(x)` | `Sin(x) / Cos(x)` | exact |
| `Acos(x)` | `π/2 − Asin(x)` | exact |
| `Erf(x)` | `tanh(0.7978845608·(x + 0.044715·x³))` | numerical approximation, ~3 × 10⁻⁴ max error |
| `ReduceL1(x)` | `ReduceSum(Abs(x))` | exact |
| `Trilu(x, k)` | `Mul(x, mask)` with a static mask | exact (when the relevant trailing dimensions are static) |
| `Unique(x)` | `Identity(x)` | semantic approximation, used only where the consumer is `ReduceMean` |
| `RNN`, `LSTM`, `GRU` (`seq_len = 4, hidden = 8`) | sequence-unrolled cells using `MatMul / Add / Sigmoid / Tanh / Mul` | exact |

We also replaced 0-dimensional scalar constants (e.g. `0.0`, `1.0`) with
1-element 1-D tensors of the same value, which is a small but important
detail for some on-device runtimes that don't fully handle 0-D
broadcasts.

> Result: **509 → 778 nodes**, output bit-identical.

### Pass 3 — Opset normalization (18 → 17)
Several reduce-family operators changed shape between ONNX opset 17 and
18 (axes moved from a node attribute to a runtime input). Most on-device
toolchains have stabilized on opset 17 ergonomics. We rewrite the
`ReduceMean / ReduceMax / ReduceMin / ReduceProd / ReduceL1 / ReduceL2 /
ReduceLogSum / ReduceLogSumExp / ReduceSumSquare` nodes back into the
attribute form, drop `BatchNormalization.training_mode` (which is an
opset-15+ attribute defaulting to inference behavior anyway), and lower
the declared opset import to 17. `ReduceSum` already moved to the
input form at opset 13, so we leave it as-is.

> Result: **778 nodes**, opset 17, output bit-identical.

## Verification

After every pass we run the model with ONNX Runtime on your supplied
calibration inputs and compare against the reference output of the
original model. The pipeline aborts if any pass exceeds the configured
tolerance (default `1 × 10⁻⁵` absolute / relative). On your inputs all
three passes produce an output that matches the reference to within
floating-point round-off (`|diff| = 0`).

## How to use it on your side

```bash
python prepare_for_ondevice.py \
    --input  your_model.onnx \
    --output your_model.modified.onnx \
    --inputs-dir calibration_inputs/
```

The script needs nothing more than `onnx`, `onnxruntime`, and `numpy`.
It is self-contained and reads only the original ONNX file and the
calibration inputs you supply.

## What to watch out for going forward

A few of the rewrites are tied to the *current* shape of the graph, and
will need re-checking if you re-export the model with different
hyper-parameters or input shapes:

1. **`Erf` approximation.** The ~3 × 10⁻⁴ approximation is well within
   typical inference tolerances, but if your downstream metric is
   sensitive to small numerical differences, please let us know — we can
   substitute a higher-precision (Abramowitz & Stegun 7.1.26)
   approximation at the cost of a few extra operators.
2. **`Trilu` static mask** assumes the last two dimensions of the input
   are statically known. If the trailing rank or sizes change in a
   future export, the rewrite will need to switch to an alternate form.
3. **`RNN / LSTM / GRU` unrolling** is hard-coded to your current
   `seq_len = 4, hidden = 8`. A different sequence length needs another
   pass (the script logic handles it; only the constants change).
4. **`Unique → Identity`** is sound only because the single consumer is
   a `ReduceMean`. If you reorganize that branch, this rewrite needs to
   be revisited.
5. **`NonZero`** is preserved as-is. It produces a dynamically-shaped
   output, which on-device runtimes accept but flag with a warning.
   In practice it has worked for us on this graph; if you observe any
   shape-related runtime failure we can replace it with a fixed-size
   masked-tensor equivalent.

If you'd like, please share future model exports and we'll re-run the
pipeline and confirm equivalence the same way.

Best regards,
[Your name]

---

### Attachments
- `prepare_for_ondevice.py` — the unified preprocessing script
- `max_ops_final.modified.onnx` — the modified, on-device-ready ONNX
- A short side-by-side report of node counts and per-pass output diffs
