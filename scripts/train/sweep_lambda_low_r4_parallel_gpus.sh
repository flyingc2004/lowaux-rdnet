#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

LAMBDA_LOW_GRID="${LAMBDA_LOW_GRID:-0.1,0.2,0.4,0.8}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
RUN_PREFIX="${RUN_PREFIX:-rdnet_r4_lowlambda_1gpu}"
RUN_SUFFIX="${RUN_SUFFIX:-mid12}"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a lambda_values <<< "${LAMBDA_LOW_GRID}"
IFS=',' read -r -a gpu_values <<< "${GPU_IDS}"

if (( ${#lambda_values[@]} > ${#gpu_values[@]} )); then
  printf 'Need at least one GPU per lambda: lambdas=%s gpus=%s\n' \
    "${LAMBDA_LOW_GRID}" "${GPU_IDS}" >&2
  exit 1
fi

printf '[PARALLEL SWEEP] lambdas: %s\n' "${LAMBDA_LOW_GRID}"
printf '[PARALLEL SWEEP] gpus: %s\n' "${GPU_IDS}"
printf '[PARALLEL SWEEP] run prefix/suffix: %s / %s\n' "${RUN_PREFIX}" "${RUN_SUFFIX}"

pids=()
for index in "${!lambda_values[@]}"; do
  lambda="$(printf '%s' "${lambda_values[$index]}" | tr -d '[:space:]')"
  gpu="$(printf '%s' "${gpu_values[$index]}" | tr -d '[:space:]')"
  if [[ -z "${lambda}" || -z "${gpu}" ]]; then
    continue
  fi

  log_dir="${ROOT}/results/lambda_low_sweep/logs"
  mkdir -p "${log_dir}"
  safe_lambda="${lambda//./p}"
  safe_lambda="${safe_lambda//-/m}"
  log_path="${log_dir}/${RUN_PREFIX}_${safe_lambda}_${RUN_SUFFIX}_gpu${gpu}.log"

  printf '[PARALLEL SWEEP] lambda=%s gpu=%s log=%s\n' "${lambda}" "${gpu}" "${log_path}"
  (
    LAMBDA_LOW_GRID="${lambda}" \
    GPU_IDS="${gpu}" \
    RUN_PREFIX="${RUN_PREFIX}" \
    RUN_SUFFIX="${RUN_SUFFIX}" \
    STRATEGY="${STRATEGY:-auto}" \
      bash "${ROOT}/scripts/train/sweep_lambda_low_r4.sh"
  ) >"${log_path}" 2>&1 &
  pids+=("$!")

  if [[ "${DRY_RUN}" == "1" ]]; then
    wait "${pids[-1]}"
  fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[PARALLEL SWEEP] dry-run complete.\n'
  exit 0
fi

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
