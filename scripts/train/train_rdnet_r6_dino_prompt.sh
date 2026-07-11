#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

first_existing_path() {
  local candidate
  for candidate in "$@"; do
    if [[ -e "${candidate}" ]]; then
      printf '%s' "${candidate}"
      return
    fi
  done
  printf '%s' "$1"
}

if [[ -z "${DINO_MODEL_PATH:-}" ]]; then
  printf 'DINO_MODEL_PATH must point to a local DINO/DINOv3 model directory.\n' >&2
  exit 1
fi

export RUN_NAME="${RUN_NAME:-rdnet_r6_dino_prompt_from_r5}"
export PRETRAIN_NETWORK_G="${PRETRAIN_NETWORK_G:-$(first_existing_path \
  "${ROOT}/external/weights/rdnet_r5_final.ckpt" \
  "${ROOT}/XReflection/experiments/rdnet_r5a_rrw_only_from_r4a_e1/checkpoints/epoch=34-step=34685-metrics/average/psnr=27.4339.ckpt")}"
export TRAIN_PIPELINE="r5_rrw_only"
export REFLECTION_TARGET_MODE="residual_lowpass_aux"
export REFLECTION_LOWPASS_KERNEL="${REFLECTION_LOWPASS_KERNEL:-31}"
export REFLECTION_LOWPASS_SIGMA="${REFLECTION_LOWPASS_SIGMA:-5.0}"
export REFLECTION_LOWPASS_AUX_WEIGHT="${REFLECTION_LOWPASS_AUX_WEIGHT:-0.2}"
export BASEBALL_LR="${BASEBALL_LR:-5e-6}"
export OTHER_LR="${OTHER_LR:-1e-5}"
export PRECISION="${PRECISION:-bf16-mixed}"
export STRATEGY="${STRATEGY:-ddp_static_graph}"
export ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export SAVE_TOP_K="${SAVE_TOP_K:-3}"

export DINO_PROMPT_ENABLE="1"
export DINO_PROMPT_INPUT_SIZE="${DINO_PROMPT_INPUT_SIZE:-224}"
export DINO_PROMPT_STRENGTH="${DINO_PROMPT_STRENGTH:-0.1}"
export DINO_PROMPT_ADAPTER_HIDDEN_DIM="${DINO_PROMPT_ADAPTER_HIDDEN_DIM:-256}"
export DINO_PROMPT_NORMALIZE_FEATURES="${DINO_PROMPT_NORMALIZE_FEATURES:-1}"

exec bash "${ROOT}/scripts/train/train_rdnet_r5.sh"
