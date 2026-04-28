#!/usr/bin/env bash
# QNN/QAIRT pipeline for one sapiens2 variant — minimal.
#
# Just two `qairt-converter` invocations: FP32 and FP16. Both are
# HTP-deployable (the on-device QNN runtime JIT-compiles to Hexagon
# at first inference). We deliberately do NOT run:
#
#   * `qnn-net-run --backend libQnnCpu.so`  — CPU smoke test, not a
#                                             deployment artifact and
#                                             ONNX Runtime already
#                                             gives us an FP32 reference.
#   * `qairt-quantizer`                      — INT8 calibration belongs
#                                             to deployment with real
#                                             data, not the bundle stage.
#   * `snpe-dlc-graph-prepare`               — offline HTP cache is a
#                                             startup-latency optimization
#                                             that the deployment image
#                                             can rebuild on-target with
#                                             its own VTCM/thread settings.
#
# Args:
#   $1 — variant directory  (e.g. /mnt/.../models/sapiens2/pose_0_4b)
#   $2 — ONNX path inside the variant dir  (e.g. .../model.onnx)
#   $3 — input tensor name  (default: "input")

# `conda activate` references unset vars; -u is incompatible.
set -eo pipefail

MODEL_DIR="${1:?model dir required}"
ONNX_PATH="${2:?onnx path required}"
INPUT_NAME="${3:-input}"

QAIRT_SDK_ROOT="${QAIRT_SDK_ROOT:-/opt/qcom/aistack/qnn/2.44.0.260225}"
PY_ENV="${PY_ENV:-py310}"

# qairt-converter spills multi-GB scratch files for 5B-class models — route
# them to the big disk so / doesn't fill up.
if [ -d /mnt/disks/zeticai_database/tmp_scratch ]; then
  export TMPDIR=/mnt/disks/zeticai_database/tmp_scratch
fi

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

# Pull input shape from the saved sample so we don't plumb it through.
read H W <<<"$(python3 -c "import numpy as np; a=np.load('sample_input.npy'); print(a.shape[2], a.shape[3])")"
echo "[qnn] input shape (1, 3, $H, $W)"

for bw in 32 16; do
  out="model_fp${bw}.dlc"
  [ "$bw" = "32" ] && out="model.dlc"  # FP32 is the canonical name.
  echo "[qnn] qairt-converter --float_bitwidth $bw  -> $out"
  qairt-converter \
    --input_network "$ONNX_PATH" \
    --output_path "$out" \
    --export_format DLC_DEFAULT \
    --float_bitwidth "$bw" \
    --source_model_input_shape "$INPUT_NAME" "1,3,$H,$W" \
    --source_model_input_layout "$INPUT_NAME" NCHW \
    --desired_input_layout "$INPUT_NAME" NCHW \
    2>&1 | tail -5
done

echo "[qnn] qairt-dlc-info on model.dlc"
qairt-dlc-info -i model.dlc 2>&1 | tail -15

echo "[qnn] DONE — $MODEL_DIR/model.dlc + model_fp16.dlc"
