#!/usr/bin/env bash
# Convert source.onnx to QNN DLC (FP32 + FP16) for HTP deployment.
#
# Just `qairt-converter` — no CPU smoke test, no INT8 quantization, no
# offline HTP cache. The on-device QNN runtime handles HTP graph
# preparation at first inference; INT8 calibration belongs to the
# deployment stage with real data.
#
# Steps:
#   1) Activate Python 3.10 conda env (qairt-converter is built against it).
#   2) Manually export QAIRT/QNN/SNPE roots — sourcing envsetup.sh from
#      the symlinked tree confuses its `dirname` resolution.
#   3) qairt-converter --float_bitwidth 32  -> model.dlc
#   4) qairt-converter --float_bitwidth 16  -> model_fp16.dlc
#   5) qairt-dlc-info to confirm op coverage.

# `conda activate` references unset vars internally; `-u` is incompatible.
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

for bw in 32 16; do
  out="model_fp${bw}.dlc"
  [ "$bw" = "32" ] && out="model.dlc"
  echo "[qnn] qairt-converter source.onnx --float_bitwidth $bw -> $out"
  qairt-converter \
    --input_network source.onnx \
    --output_path "$out" \
    --export_format DLC_DEFAULT \
    --float_bitwidth "$bw" \
    --source_model_input_shape input_1 1,256,256,3 \
    --source_model_input_layout input_1 NHWC \
    --desired_input_layout input_1 NHWC \
    2>&1 | tail -5
done

echo "[qnn] qairt-dlc-info on model.dlc"
qairt-dlc-info -i model.dlc 2>&1 | tail -25

echo "[qnn] DONE — wrote $MODEL_DIR/model.dlc and model_fp16.dlc"
