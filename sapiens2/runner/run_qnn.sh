#!/usr/bin/env bash
# QNN/QAIRT pipeline for one sapiens2 variant.
#
# Args:
#   $1 — variant directory  (e.g. /mnt/.../models/sapiens2/pose_0_4b)
#   $2 — ONNX path inside the variant dir  (e.g. .../model.onnx)
#   $3 — input tensor name  (default: "input")
#
# Produces inside $1:
#   model.dlc                 FP32 DLC
#   model_int8.dlc            INT8 per-channel quantised DLC
#   model_int8_htp.dlc        INT8 + offline HTP cache for v68/v73/v75/v79
#   model_int8_encoding.json  per-tensor scale/offset
#   qnn_run/                  qnn-net-run CPU-backend results
#   qnn_htp_run/              qnn-net-run HTP-backend results

# `conda activate` references unset vars; -u is incompatible.
set -eo pipefail

MODEL_DIR="${1:?model dir required}"
ONNX_PATH="${2:?onnx path required}"
INPUT_NAME="${3:-input}"

QAIRT_SDK_ROOT="${QAIRT_SDK_ROOT:-/opt/qcom/aistack/qnn/2.44.0.260225}"
PY_ENV="${PY_ENV:-py310}"
HTP_TARGETS="v68,v73,v75,v79"

# shellcheck disable=SC1091
source /home/yeonseok/miniconda3/etc/profile.d/conda.sh
conda activate "$PY_ENV"

export QAIRT_SDK_ROOT
export QNN_SDK_ROOT="$QAIRT_SDK_ROOT"
export SNPE_ROOT="$QAIRT_SDK_ROOT"
export PATH="$QAIRT_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export LD_LIBRARY_PATH="$QAIRT_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$QAIRT_SDK_ROOT/lib/python:${PYTHONPATH:-}"

cd "$MODEL_DIR"
echo "[qnn] $(pwd)  py=$(which python3)"

# Sapiens2 input is fixed (1, 3, H, W) — read it back from sample_input.npy
# so we don't need to plumb the shape through the shell.
read H W <<<"$(python3 -c "import numpy as np; a=np.load('sample_input.npy'); print(a.shape[2], a.shape[3])")"
echo "[qnn] input shape (1, 3, $H, $W)"

OUT_DLC="model.dlc"
echo "[qnn] qairt-converter $ONNX_PATH -> $OUT_DLC"
qairt-converter \
  --input_network "$ONNX_PATH" \
  --output_path "$OUT_DLC" \
  --export_format DLC_DEFAULT \
  --float_bitwidth 32 \
  --source_model_input_shape "$INPUT_NAME" "1,3,$H,$W" \
  --source_model_input_layout "$INPUT_NAME" NCHW \
  --desired_input_layout "$INPUT_NAME" NCHW \
  2>&1 | tail -10

echo "[qnn] qairt-dlc-info on $OUT_DLC"
qairt-dlc-info -i "$OUT_DLC" 2>&1 | tail -15

# Calibration with the single sample input.
RUN_DIR="qnn_run"
rm -rf "$RUN_DIR"; mkdir -p "$RUN_DIR"
python3 -c "import numpy as np; np.load('sample_input.npy').tofile('$RUN_DIR/input.raw')"
echo "$INPUT_NAME:=$PWD/$RUN_DIR/input.raw" > "$RUN_DIR/input_list.txt"

echo "[qnn] qnn-net-run --backend libQnnCpu.so on $OUT_DLC"
qnn-net-run \
  --backend "$QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnCpu.so" \
  --dlc_path "$OUT_DLC" \
  --input_list "$RUN_DIR/input_list.txt" \
  --output_dir "$RUN_DIR/output" 2>&1 | tail -8

# ----------------------------------------------------------------------
# HTP path: quantize → graph_prepare → execute on x86 HTP simulator.
OUT_DLC_INT8="model_int8.dlc"
echo "[htp] qairt-quantizer (INT8 calibration with $RUN_DIR/input.raw)"
echo "$PWD/$RUN_DIR/input.raw" > "$RUN_DIR/calib_list.txt"
qairt-quantizer \
  --input_dlc "$OUT_DLC" \
  --output_dlc "$OUT_DLC_INT8" \
  --input_list "$RUN_DIR/calib_list.txt" \
  --use_per_channel_quantization \
  --dump_encoding_json \
  --target_backend HTP 2>&1 | tail -10

echo "[htp] snpe-dlc-graph-prepare --htp_archs=$HTP_TARGETS"
snpe-dlc-graph-prepare \
  --input_dlc "$OUT_DLC_INT8" \
  --output_dlc "${OUT_DLC_INT8%.dlc}_htp.dlc" \
  --htp_archs "$HTP_TARGETS" 2>&1 | tail -10

# NOTE: the x86 HTP simulator (libQnnHtp.so + qnn-net-run) is too slow for
# transformer-scale models — a 0.1B ViT takes ~15 min per inference, a 5B
# would take hours. The HTP-readiness signal we actually want is the
# *successful offline-prepare across all four Hexagon archs above* (cache
# records embed the compiled HTP kernels). On-device validation is the
# next step in the deploy pipeline, not part of this offline build.
echo "[qnn] DONE — $MODEL_DIR/model.dlc and ${OUT_DLC_INT8%.dlc}_htp.dlc"
