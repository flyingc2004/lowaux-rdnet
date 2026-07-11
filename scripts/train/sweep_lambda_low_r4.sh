#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
PARENT_ROOT="$(cd "${ROOT}/.." && pwd)"

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

LAMBDA_LOW_GRID="${LAMBDA_LOW_GRID:-0.0,0.05,0.1,0.2,0.4,0.8}"
RUN_PREFIX="${RUN_PREFIX:-rdnet_r4_lowlambda}"
RUN_SUFFIX="${RUN_SUFFIX:-small}"

export XREFLECTION_ROOT="${XREFLECTION_ROOT:-${ROOT}/XReflection}"
export SIRS_ROOT="${SIRS_ROOT:-$(first_existing_path \
  "${ROOT}/XReflection/data/sirs" \
  "${PARENT_ROOT}/XReflection/data/sirs")}"
export ERRNET_DATA_ROOT="${ERRNET_DATA_ROOT:-$(first_existing_path \
  "${ROOT}/ERRNet/datasets/processed_data" \
  "${PARENT_ROOT}/ERRNet/datasets/processed_data")}"
export CLS_MODEL="${CLS_MODEL:-$(first_existing_path \
  "${ROOT}/XReflection/pretrained/cls_model.pth" \
  "${ROOT}/external/weights/cls_model.pth" \
  "${PARENT_ROOT}/XReflection/pretrained/cls_model.pth")}"
export FOCAL_MODEL="${FOCAL_MODEL:-$(first_existing_path \
  "${ROOT}/XReflection/pretrained/focal.pth" \
  "${ROOT}/external/weights/focal.pth" \
  "${PARENT_ROOT}/XReflection/pretrained/focal.pth")}"
export PRETRAIN_NETWORK_G="${PRETRAIN_NETWORK_G:-$(first_existing_path \
  "${ROOT}/external/weights/rdnet-26.4849.ckpt" \
  "${ROOT}/XReflection/pretrained/rdnet-26.4849.ckpt" \
  "${PARENT_ROOT}/XReflection/pretrained/rdnet-26.4849.ckpt")}"
export TRAIN_PIPELINE="default"
export MAX_EPOCHS="${MAX_EPOCHS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export PRECISION="${PRECISION:-bf16-mixed}"
export STRATEGY="${STRATEGY:-ddp_static_graph}"
export NUM_SANITY_VAL_STEPS="${NUM_SANITY_VAL_STEPS:-0}"
export SAVE_TOP_K="${SAVE_TOP_K:-1}"
export LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-0.125}"
export LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.25}"
export REFLECTION_TARGET_MODE="residual_lowpass_aux"
export REFLECTION_LOWPASS_KERNEL="${REFLECTION_LOWPASS_KERNEL:-31}"
export REFLECTION_LOWPASS_SIGMA="${REFLECTION_LOWPASS_SIGMA:-5.0}"
export BASEBALL_LR="${BASEBALL_LR:-5e-6}"
export OTHER_LR="${OTHER_LR:-1e-5}"

CONFIG_DIR="${CONFIG_DIR:-${ROOT}/results/rdnet_configs}"

safe_lambda_name() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}

IFS=',' read -r -a lambda_values <<< "${LAMBDA_LOW_GRID}"

printf '[SWEEP] R4 lambda_low grid: %s\n' "${LAMBDA_LOW_GRID}"
printf '[SWEEP] run suffix: %s\n' "${RUN_SUFFIX}"
printf '[SWEEP] init checkpoint: %s\n' "${PRETRAIN_NETWORK_G}"
printf '[SWEEP] SIRS root: %s\n' "${SIRS_ROOT}"
printf '[SWEEP] ERRNet data root: %s\n' "${ERRNET_DATA_ROOT}"
printf '[SWEEP] train batches: %s, val batches: %s, epochs: %s\n' \
  "${LIMIT_TRAIN_BATCHES}" "${LIMIT_VAL_BATCHES}" "${MAX_EPOCHS}"

for raw_lambda in "${lambda_values[@]}"; do
  lambda="$(printf '%s' "${raw_lambda}" | tr -d '[:space:]')"
  if [[ -z "${lambda}" ]]; then
    continue
  fi

  safe_lambda="$(safe_lambda_name "${lambda}")"
  if [[ -n "${RUN_SUFFIX}" ]]; then
    run_name="${RUN_PREFIX}_${safe_lambda}_${RUN_SUFFIX}"
  else
    run_name="${RUN_PREFIX}_${safe_lambda}"
  fi

  printf '\n[SWEEP] lambda_low=%s run=%s\n' "${lambda}" "${run_name}"
  RUN_NAME="${run_name}" \
  CONFIG_PATH="${CONFIG_DIR}/${run_name}.yml" \
  REFLECTION_LOWPASS_AUX_WEIGHT="${lambda}" \
    bash "${ROOT}/scripts/train/rdnet_train_internal.sh"
done
