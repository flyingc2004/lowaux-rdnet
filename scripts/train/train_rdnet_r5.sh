#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

XREFLECTION_ROOT="${XREFLECTION_ROOT:-${ROOT}/XReflection}"
RRW_ROOT="${RRW_ROOT:-${ROOT}/RRW}"
RRW_MANIFEST_DIR="${RRW_MANIFEST_DIR:-${ROOT}/results/rdnet_data/r5_rrw}"

python "${ROOT}/scripts/data/prepare_rrw_r5_manifests.py" \
  --rrw-root "${RRW_ROOT}" \
  --output-dir "${RRW_MANIFEST_DIR}"

export XREFLECTION_ROOT RRW_ROOT
export RRW_TRAIN_MANIFEST="${RRW_MANIFEST_DIR}/train.csv"
export TRAIN_PIPELINE="r5_rrw_only"
export RUN_NAME="${RUN_NAME:-rdnet_r5_rrw_only_from_r4_e1}"
export PRETRAIN_NETWORK_G="${PRETRAIN_NETWORK_G:-${ROOT}/external/weights/rdnet_r4_epoch1.ckpt}"
export MAX_EPOCHS="${MAX_EPOCHS:-40}"
export PRECISION="bf16-mixed"
export REFLECTION_TARGET_MODE="residual_lowpass_aux"
export REFLECTION_LOWPASS_KERNEL="31"
export REFLECTION_LOWPASS_SIGMA="5.0"
export REFLECTION_LOWPASS_AUX_WEIGHT="0.2"
export BASEBALL_LR="5e-6"
export OTHER_LR="1e-5"
export SAVE_TOP_K="${SAVE_TOP_K:-3}"

exec bash "${ROOT}/scripts/train/rdnet_train_internal.sh"
