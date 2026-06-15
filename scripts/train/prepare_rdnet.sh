#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

XREFLECTION_ROOT="${XREFLECTION_ROOT:-${ROOT}/XReflection}"
CONDA_ENV="${CONDA_ENV:-xreflection}"
BASE_ENV="${BASE_ENV:-errnet}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PRETRAINED_DIR="${PRETRAINED_DIR:-${XREFLECTION_ROOT}/pretrained}"
DATA_DIR="${DATA_DIR:-${XREFLECTION_ROOT}/data}"
INSTALL_ENV="${INSTALL_ENV:-1}"
DOWNLOAD_WEIGHTS="${DOWNLOAD_WEIGHTS:-1}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-0}"
DRY_RUN="${DRY_RUN:-0}"
RESUME_EXISTING_DOWNLOADS="${RESUME_EXISTING_DOWNLOADS:-0}"
ALLOW_RESTART_DOWNLOAD="${ALLOW_RESTART_DOWNLOAD:-1}"

RDNET_CKPT_URL="${RDNET_CKPT_URL:-https://checkpoints.mingjia.li/rdnet-26.4849.ckpt}"
FOCAL_URL="${FOCAL_URL:-https://checkpoints.mingjia.li/focal.pth}"
CLS_MODEL_URL="${CLS_MODEL_URL:-https://checkpoints.mingjia.li/cls_model.pth}"
SIRS_URL="${SIRS_URL:-https://checkpoints.mingjia.li/sirs.zip}"

run_cmd() {
  shift || true
  printf '[RUN]'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

download_if_missing() {
  local url="$1"
  local target="$2"
  if [[ -f "${target}" ]]; then
    if [[ "${RESUME_EXISTING_DOWNLOADS}" == "1" ]]; then
      if [[ ! -f "${target}.part" ]]; then
        run_cmd mv mv "${target}" "${target}.part"
      fi
    else
      printf '[SKIP] %s exists\n' "${target}"
      return
    fi
  fi
  if [[ -f "${target}.part" && "${RESUME_EXISTING_DOWNLOADS}" == "1" ]]; then
    printf '[RESUME] %s\n' "${target}.part"
  fi
  mkdir -p "$(dirname "${target}")"
  if [[ -f "${target}.part" && "${RESUME_EXISTING_DOWNLOADS}" == "1" ]]; then
    if ! run_cmd curl curl -L --fail --retry 3 --retry-delay 5 -C - -o "${target}.part" "${url}"; then
      if [[ "${ALLOW_RESTART_DOWNLOAD}" != "1" ]]; then
        return 1
      fi
      printf '[RESTART] server does not support resume, restarting: %s\n' "${target}"
      run_cmd curl curl -L --fail --retry 3 --retry-delay 5 -o "${target}.part" "${url}"
    fi
  else
    run_cmd curl curl -L --fail --retry 3 --retry-delay 5 -o "${target}.part" "${url}"
  fi
  if [[ "${DRY_RUN}" != "1" ]]; then
    mv "${target}.part" "${target}"
  fi
  if [[ -f "${target}" ]]; then
    printf '[SKIP] %s exists\n' "${target}"
  fi
}

if [[ ! -d "${XREFLECTION_ROOT}" ]]; then
  printf 'XReflection not found: %s\n' "${XREFLECTION_ROOT}" >&2
  printf 'Clone it first: git clone https://github.com/hainuo-wang/XReflection.git %s\n' "${XREFLECTION_ROOT}" >&2
  exit 1
fi

if [[ "${INSTALL_ENV}" == "1" ]]; then
  if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    printf '[SKIP] conda env exists: %s\n' "${CONDA_ENV}"
  elif conda env list | awk '{print $1}' | grep -qx "${BASE_ENV}"; then
    run_cmd conda conda create -n "${CONDA_ENV}" --clone "${BASE_ENV}" -y
  else
    run_cmd conda conda create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
  fi

  run_cmd conda conda run --no-capture-output -n "${CONDA_ENV}" python -m pip install \
    -r "${XREFLECTION_ROOT}/requirements.txt" \
    timm einops ema-pytorch fsspec fvcore scikit-learn tensorboardx torchmetrics wandb
  run_cmd conda conda run --no-capture-output -n "${CONDA_ENV}" python -m pip install -e "${XREFLECTION_ROOT}"
fi

if [[ "${DOWNLOAD_WEIGHTS}" == "1" ]]; then
  download_if_missing "${RDNET_CKPT_URL}" "${PRETRAINED_DIR}/rdnet-26.4849.ckpt"
  download_if_missing "${FOCAL_URL}" "${PRETRAINED_DIR}/focal.pth"
  download_if_missing "${CLS_MODEL_URL}" "${PRETRAINED_DIR}/cls_model.pth"
fi

if [[ "${DOWNLOAD_DATA}" == "1" ]]; then
  download_if_missing "${SIRS_URL}" "${DATA_DIR}/sirs.zip"
  if [[ ! -d "${DATA_DIR}/sirs" ]]; then
    mkdir -p "${DATA_DIR}"
    run_cmd unzip unzip -q -n "${DATA_DIR}/sirs.zip" -d "${DATA_DIR}"
  else
    printf '[SKIP] data directory exists: %s\n' "${DATA_DIR}/sirs"
  fi
fi

printf '\n[OK] RDNet preparation script finished.\n'
printf 'XReflection root: %s\n' "${XREFLECTION_ROOT}"
printf 'Conda env: %s\n' "${CONDA_ENV}"
printf 'Pretrained dir: %s\n' "${PRETRAINED_DIR}"
printf 'Data dir: %s\n' "${DATA_DIR}"
