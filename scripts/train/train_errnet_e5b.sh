#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}/ERRNet"

STAGE="${STAGE:-all}"
CONDA_ENV="${CONDA_ENV:-errnet}"
GPU_ID="${GPU_ID:-0}"
TMPDIR="${TMPDIR:-/tmp/ljz-errnet}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ljz-errnet-mpl}"
DRY_RUN="${DRY_RUN:-0}"

STAGE1_NAME="${STAGE1_NAME:-errnet_e5b_refiner_stage1}"
STAGE2_NAME="${STAGE2_NAME:-errnet_e5b_refiner_stage2}"
STAGE1_INIT_CKPT="${STAGE1_INIT_CKPT:-checkpoints/errnet_e3_long/errnet_latest.pt}"
STAGE2_INIT_CKPT="${STAGE2_INIT_CKPT:-checkpoints/${STAGE1_NAME}/errnet_latest.pt}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-5}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-5}"
STAGE1_LR="${STAGE1_LR:-5e-5}"
STAGE2_LR="${STAGE2_LR:-2e-6}"

BATCH_SIZE="${BATCH_SIZE:-4}"
NTHREADS="${NTHREADS:-0}"
NO_PIN_MEMORY="${NO_PIN_MEMORY:-1}"
MAX_DATASET_SIZE="${MAX_DATASET_SIZE:-2000}"
SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-1}"

UNALIGNED_FUSION_RATIOS="${UNALIGNED_FUSION_RATIOS:-0.4,0.2,0.4}"
SYNTHETIC_KERNEL_SIZES="${SYNTHETIC_KERNEL_SIZES:-7,11,15}"
LOW_ALPHA="${LOW_ALPHA:-0.6}"
HIGH_ALPHA="${HIGH_ALPHA:-1.0}"
GHOST_PROBABILITY="${GHOST_PROBABILITY:-0.25}"
GHOST_MAX_SHIFT="${GHOST_MAX_SHIFT:-8}"
REFINER_CHANNELS="${REFINER_CHANNELS:-32}"
REFINER_DILATIONS="${REFINER_DILATIONS:-1,2,4,2,1}"
REFINER_RES_SCALE="${REFINER_RES_SCALE:-0.1}"

validate_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer, got: %s\n' "${name}" "${value}" >&2
    exit 1
  fi
}

print_command() {
  printf '[CMD]'
  printf ' %q' "$@"
  printf '\n'
}

require_checkpoint() {
  local checkpoint="$1"
  if [[ ! -f "${checkpoint}" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '[DRY-RUN] checkpoint does not exist yet: %s\n' "${checkpoint}"
      return
    fi
    printf 'Checkpoint not found: %s\n' "${checkpoint}" >&2
    exit 1
  fi
}

run_command() {
  print_command "$@"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  "$@"
}

runtime_prefix=(
  conda run --no-capture-output -n "${CONDA_ENV}" env
  "TMPDIR=${TMPDIR}"
  "MPLCONFIGDIR=${MPLCONFIGDIR}"
  "CUDA_VISIBLE_DEVICES=${GPU_ID}"
  python -u
)

pin_memory_args=()
if [[ "${NO_PIN_MEMORY}" == "1" ]]; then
  pin_memory_args+=(--no_pin_memory)
fi

augmentation_args=(
  --synthetic_kernel_sizes "${SYNTHETIC_KERNEL_SIZES}"
  --low_alpha "${LOW_ALPHA}"
  --high_alpha "${HIGH_ALPHA}"
  --ghost_probability "${GHOST_PROBABILITY}"
  --ghost_max_shift "${GHOST_MAX_SHIFT}"
  --random_reflection_pair
)

common_args=(
  --hyper
  -r
  --gpu_ids 0
  --batchSize "${BATCH_SIZE}"
  --max_dataset_size "${MAX_DATASET_SIZE}"
  --save_epoch_freq "${SAVE_EPOCH_FREQ}"
  --lambda_gan 0
  --gan_start_epoch -1
  --unaligned_loss vgg
  --unaligned_fusion_ratios "${UNALIGNED_FUSION_RATIOS}"
  --refiner_mode dilated
  --refiner_channels "${REFINER_CHANNELS}"
  --refiner_dilations "${REFINER_DILATIONS}"
  --refiner_res_scale "${REFINER_RES_SCALE}"
  --nThreads "${NTHREADS}"
  "${pin_memory_args[@]}"
  --no-verbose
)

run_stage1() {
  require_checkpoint "${STAGE1_INIT_CKPT}"
  printf '[RUN] E5b stage1 name=%s checkpoint=%s extra_epochs=%s lr=%s freeze_backbone=1\n' \
    "${STAGE1_NAME}" "${STAGE1_INIT_CKPT}" "${STAGE1_EPOCHS}" "${STAGE1_LR}"
  run_command "${runtime_prefix[@]}" train_errnet_unaligned.py \
    --name "${STAGE1_NAME}" \
    --icnn_path "${STAGE1_INIT_CKPT}" \
    --extra_epochs "${STAGE1_EPOCHS}" \
    --lr "${STAGE1_LR}" \
    --freeze_backbone \
    "${augmentation_args[@]}" \
    "${common_args[@]}"
}

run_stage2() {
  require_checkpoint "${STAGE2_INIT_CKPT}"
  printf '[RUN] E5b stage2 name=%s checkpoint=%s extra_epochs=%s lr=%s freeze_backbone=0\n' \
    "${STAGE2_NAME}" "${STAGE2_INIT_CKPT}" "${STAGE2_EPOCHS}" "${STAGE2_LR}"
  run_command "${runtime_prefix[@]}" train_errnet_unaligned.py \
    --name "${STAGE2_NAME}" \
    --icnn_path "${STAGE2_INIT_CKPT}" \
    --extra_epochs "${STAGE2_EPOCHS}" \
    --lr "${STAGE2_LR}" \
    "${augmentation_args[@]}" \
    "${common_args[@]}"
}

case "${STAGE}" in
  all|stage1|stage2) ;;
  *)
    printf 'STAGE must be one of: all, stage1, stage2; got: %s\n' "${STAGE}" >&2
    exit 1
    ;;
esac
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  printf 'DRY_RUN must be 0 or 1, got: %s\n' "${DRY_RUN}" >&2
  exit 1
fi
validate_positive_integer STAGE1_EPOCHS "${STAGE1_EPOCHS}"
validate_positive_integer STAGE2_EPOCHS "${STAGE2_EPOCHS}"
validate_positive_integer BATCH_SIZE "${BATCH_SIZE}"
validate_positive_integer MAX_DATASET_SIZE "${MAX_DATASET_SIZE}"
validate_positive_integer SAVE_EPOCH_FREQ "${SAVE_EPOCH_FREQ}"
validate_positive_integer REFINER_CHANNELS "${REFINER_CHANNELS}"

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"
fi

case "${STAGE}" in
  all)
    run_stage1
    run_stage2
    ;;
  stage1) run_stage1 ;;
  stage2) run_stage2 ;;
esac
