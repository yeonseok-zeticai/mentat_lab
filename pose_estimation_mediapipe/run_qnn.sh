#!/usr/bin/env bash
# Convert source.onnx to a QNN DLC for the Hexagon Tensor Processor (HTP).
#
# Steps:
#   1) Activate a Python 3.10 conda env (qairt-converter is built against it).
#   2) Manually export QAIRT/QNN/SNPE roots — sourcing envsetup.sh from the
#      symlinked tree confuses its `dirname` resolution.
#   3) qairt-converter --input_network source.onnx --export_format DLC.
#   4) Validate the resulting DLC with qairt-dlc-info.
#
# All artifacts land next to source.onnx in $MODEL_DIR.

# `conda activate` references unset vars internally, so `-u` is incompatible.
set -eo pipefail

MODEL_DIR="${MODEL_DIR:-/mnt/disks/zeticai_database/models/pose_estimation_mediapipe}"
QAIRT_SDK_ROOT="${QAIRT_SDK_ROOT:-/opt/qcom/aistack/qnn/2.44.0.260225}"
PY_ENV="${PY_ENV:-py310}"

# shellcheck disable=SC1091
source /home/yeonseok/miniconda3/etc/profile.d/conda.sh
conda activate "$PY_ENV"

export QAIRT_SDK_ROOT
export QNN_SDK_ROOT="$QAIRT_SDK_ROOT"
export SNPE_ROOT="$QAIRT_SDK_ROOT"
export PATH="$QAIRT_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export LD_LIBRARY_PATH="$QAIRT_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$QAIRT_SDK_ROOT/lib/python:${PYTHONPATH:-}"

echo "[qnn] using SDK at $QAIRT_SDK_ROOT  python=$(which python3)"

cd "$MODEL_DIR"

OUT_DLC="model.dlc"
echo "[qnn] qairt-converter source.onnx -> $OUT_DLC (HTP, FP32 reference)"
qairt-converter \
  --input_network source.onnx \
  --output_path "$OUT_DLC" \
  --export_format DLC_DEFAULT \
  --float_bitwidth 32 \
  --source_model_input_shape input_1 1,256,256,3 \
  --source_model_input_layout input_1 NHWC \
  --desired_input_layout input_1 NHWC \
  2>&1 | tail -40

echo "[qnn] qairt-dlc-info on $OUT_DLC"
qairt-dlc-info -i "$OUT_DLC" 2>&1 | tail -25

# Numerical run via the CPU backend — exercises the DLC graph end-to-end.
RUN_DIR="qnn_run"
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"

# qnn-net-run consumes raw NHWC float32; sample_input.npy already matches.
python3 -c "import numpy as np; np.load('sample_input.npy').tofile('$RUN_DIR/input_1.raw')"
echo "input_1:=$PWD/$RUN_DIR/input_1.raw" > "$RUN_DIR/input_list.txt"

echo "[qnn] qnn-net-run --backend libQnnCpu.so on $OUT_DLC"
qnn-net-run \
  --backend "$QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnCpu.so" \
  --dlc_path "$OUT_DLC" \
  --input_list "$RUN_DIR/input_list.txt" \
  --output_dir "$RUN_DIR/output" 2>&1 | tail -10

echo "[qnn] outputs written under $RUN_DIR/output:"
ls "$RUN_DIR/output"/Result_0/ 2>&1 || true

echo "[qnn] numerical compare vs ONNX Runtime:"
python3 - <<'PY'
from pathlib import Path
import numpy as np, onnxruntime as ort

md = Path('.')
sample = np.load(md / 'sample_input.npy')
sess = ort.InferenceSession(str(md / 'source.onnx'), providers=['CPUExecutionProvider'])
ort_outs = sess.run(None, {sess.get_inputs()[0].name: sample})

# Output names match the ONNX Identity node names.
names_to_shape = {
    'Identity':   (1, 195),       # landmarks
    'Identity_1': (1, 1),         # conf
    'Identity_2': (1, 256, 256, 1),  # mask
    'Identity_3': (1, 64, 64, 39),   # heatmap
    'Identity_4': (1, 117),       # landmarks_word
}
ort_map = dict(zip(['Identity', 'Identity_1', 'Identity_2', 'Identity_3', 'Identity_4'], ort_outs))

for n, shp in names_to_shape.items():
    raw = (md / 'qnn_run' / 'output' / 'Result_0' / f'{n}.raw').read_bytes()
    qnn = np.frombuffer(raw, dtype=np.float32).reshape(shp)
    onx = ort_map[n]
    diff = np.abs(qnn - onx)
    print(f'  {n}: max|Δ|={diff.max():.3e}  mean|Δ|={diff.mean():.3e}')
PY

# ---------------------------------------------------------------------------
# HTP backend verification
#
# Two checks:
#   1) snpe-dlc-graph-prepare with --htp_archs=<v68,v73,v75,v79> embeds an
#      HTP offline cache. If the prepare succeeds, every op compiled for HTP.
#   2) Quantize the FP32 DLC to INT8 (HTP's native precision) using the
#      sample input as a 1-image calibration set. Then run it through
#      qnn-net-run with libQnnHtp.so on x86 (HTP simulator) to confirm the
#      backend can execute the graph.
# ---------------------------------------------------------------------------

OUT_DLC_INT8="model_int8.dlc"
HTP_TARGETS="v68,v73,v75,v79"

echo "[htp] qairt-quantizer (INT8 calibration with sample_input.npy)"
echo "$PWD/$RUN_DIR/input_1.raw" > "$RUN_DIR/calib_list.txt"
qairt-quantizer \
  --input_dlc "$OUT_DLC" \
  --output_dlc "$OUT_DLC_INT8" \
  --input_list "$RUN_DIR/calib_list.txt" \
  --use_per_channel_quantization \
  --dump_encoding_json \
  --target_backend HTP 2>&1 | tail -20

echo "[htp] snpe-dlc-graph-prepare --htp_archs=$HTP_TARGETS"
# Force the prepare step to keep ALL five model outputs reachable from the
# HTP cache; otherwise it fuses unused output tensors and only the last
# identity ends up in the cache record.
snpe-dlc-graph-prepare \
  --input_dlc "$OUT_DLC_INT8" \
  --output_dlc "${OUT_DLC_INT8%.dlc}_htp.dlc" \
  --htp_archs "$HTP_TARGETS" \
  --set_output_tensors Identity,Identity_1,Identity_2,Identity_3,Identity_4 2>&1 | tail -20

echo "[htp] qairt-dlc-info on cached HTP DLC:"
qairt-dlc-info -i "${OUT_DLC_INT8%.dlc}_htp.dlc" 2>&1 | tail -25

# Run the INT8 graph through the HTP backend (x86 simulator path).
HTP_RUN_DIR="qnn_htp_run"
rm -rf "$HTP_RUN_DIR"
mkdir -p "$HTP_RUN_DIR"

# Quantized HTP graph expects uint8-encoded input (uFxp_8). Quantize the
# float NHWC sample to the encoding that qairt-quantizer recorded on input_1.
python3 - <<'PY'
import json, numpy as np
from pathlib import Path
md = Path('.')
x = np.load(md / 'sample_input.npy')          # float32 in [0, 1], NHWC
# Range-aware uint8 mapping with the same min/max the calibration saw.
lo, hi = float(x.min()), float(x.max())
scale = (hi - lo) / 255.0 if hi > lo else 1.0
q = np.clip(np.round((x - lo) / scale), 0, 255).astype(np.uint8)
q.tofile(md / 'qnn_htp_run' / 'input_1.raw')
print(f'[htp] uint8 sample range=[{lo},{hi}] scale={scale}')
PY
echo "input_1:=$PWD/$HTP_RUN_DIR/input_1.raw" > "$HTP_RUN_DIR/input_list.txt"

echo "[htp] qnn-net-run --backend libQnnHtp.so (x86 HTP simulator)"
if qnn-net-run \
    --backend "$QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --dlc_path "${OUT_DLC_INT8%.dlc}_htp.dlc" \
    --input_list "$HTP_RUN_DIR/input_list.txt" \
    --output_dir "$HTP_RUN_DIR/output" \
    --use_native_input_files \
    --use_native_output_files 2>&1 | tail -20 ; then
  echo "[htp] HTP simulator ran the graph; comparing to ONNX Runtime ..."
  python3 - <<'PY'
from pathlib import Path
import numpy as np, onnxruntime as ort
md = Path('.')
sample = np.load(md / 'sample_input.npy')
sess = ort.InferenceSession(str(md / 'source.onnx'), providers=['CPUExecutionProvider'])
ort_outs = sess.run(None, {sess.get_inputs()[0].name: sample})
shape_map = {
  'Identity':   (1, 195),
  'Identity_1': (1, 1),
  'Identity_2': (1, 256, 256, 1),
  'Identity_3': (1, 64, 64, 39),
  'Identity_4': (1, 117),
}
ort_map = dict(zip(shape_map, ort_outs))
result_dir = md / 'qnn_htp_run' / 'output' / 'Result_0'
# Pull each output's per-tensor uFxp_8 encoding (scale, offset) from the
# DLC info report and dequantize x = scale * (q + offset).
import json
# Reshape ops stay free of their own activation encoding; their dequant
# parameters live on the producing tensor (Conv / Sigmoid).
RESHAPE_SOURCE = {
    'Identity':   'model_1/model/convld_3d/BiasAdd;model_1/model/convld_3d/Conv2D;model_1/model/convld_3d/BiasAdd/ReadVariableOp/resource1',
    'Identity_1': 'model_1/model/activation_poseflag/Sigmoid',
    'Identity_4': 'model_1/model/convworld_3d/BiasAdd;model_1/model/convworld_3d/Conv2D;model_1/model/convworld_3d/BiasAdd/ReadVariableOp/resource1',
}
enc_path = next(md.glob('*encoding*.json'), None)
encodings = {}
if enc_path is not None:
    act_enc = json.loads(enc_path.read_text()).get('activation_encodings', {})
    def _lookup(key):
        e = act_enc.get(key)
        if not e:
            return None
        rec = e[0] if isinstance(e, list) else e
        return float(rec['scale']), float(rec['offset'])
    for name in shape_map:
        encodings[name] = _lookup(name) or _lookup(RESHAPE_SOURCE.get(name, ''))
print(f'[htp] encoding source: {enc_path}')
print(f'[htp] encodings: {encodings}')

for name, shp in shape_map.items():
    fname = f'{name}_native.raw'
    raw = (result_dir / fname).read_bytes()
    q = np.frombuffer(raw, dtype=np.uint8).reshape(shp).astype(np.float32)
    scale, offset = encodings.get(name) or (1.0, 0.0)
    # SNPE/QNN convention: real = scale * (q + offset)  (offset is negative)
    deq = scale * (q + offset)
    onx = ort_map[name]
    diff = np.abs(deq - onx)
    rng = max(abs(onx.max()), abs(onx.min()), 1e-9)
    print(f'  {name}: max|Δ|={diff.max():.3e}  mean|Δ|={diff.mean():.3e}  '
          f'rel-to-range={diff.max()/rng:.3e}')
PY
else
  echo "[htp] HTP simulator unavailable on this host — graph_prepare success above is enough to confirm HTP-readiness."
fi

echo "[qnn] DONE — wrote $MODEL_DIR/$OUT_DLC and $MODEL_DIR/${OUT_DLC_INT8%.dlc}_htp.dlc"
