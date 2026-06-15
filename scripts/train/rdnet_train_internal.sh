#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

XREFLECTION_ROOT="${XREFLECTION_ROOT:-${ROOT}/XReflection}"
CONDA_ENV="${CONDA_ENV:-xreflection}"
RUN_NAME="${RUN_NAME:-rdnet_repro_40ep}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
MAX_EPOCHS="${MAX_EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PRECISION="${PRECISION:-16-mixed}"
STRATEGY="${STRATEGY:-ddp_static_graph}"
DDP_TIMEOUT_SECONDS="${DDP_TIMEOUT_SECONDS:-7200}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-1.0}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-50}"
SAVE_TOP_K="${SAVE_TOP_K:-3}"
NUM_SANITY_VAL_STEPS="${NUM_SANITY_VAL_STEPS:-0}"
SAVE_VAL_IMAGES="${SAVE_VAL_IMAGES:-0}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
RDNET_TMPDIR="${RDNET_TMPDIR:-/tmp/ljz-rdnet}"
RDNET_MPLCONFIGDIR="${RDNET_MPLCONFIGDIR:-/tmp/ljz-rdnet-mpl}"
RDNET_TORCH_HOME="${RDNET_TORCH_HOME:-/tmp/ljz-torch-cache}"
DRY_RUN="${DRY_RUN:-0}"
TRAIN_PIPELINE="${TRAIN_PIPELINE:-default}"

SIRS_ROOT="${SIRS_ROOT:-${XREFLECTION_ROOT}/data/sirs}"
ERRNET_DATA_ROOT="${ERRNET_DATA_ROOT:-${ROOT}/ERRNet/datasets/processed_data}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-${XREFLECTION_ROOT}/experiments}"
CONFIG_DIR="${CONFIG_DIR:-${ROOT}/results/rdnet_configs}"
CONFIG_PATH="${CONFIG_PATH:-${CONFIG_DIR}/${RUN_NAME}.yml}"

REAL_TRAIN_DIR="${REAL_TRAIN_DIR:-${SIRS_ROOT}/train/real}"
NATURE_TRAIN_DIR="${NATURE_TRAIN_DIR:-${SIRS_ROOT}/train/nature}"
SYN_TRAIN_DIR="${SYN_TRAIN_DIR:-${SIRS_ROOT}/train/VOCdevkit/VOC2012/PNGImages}"
SYN_FNS="${SYN_FNS:-${SIRS_ROOT}/train/VOC2012_224_train_png.txt}"
RRW_ROOT="${RRW_ROOT:-${ROOT}/RRW}"
RRW_TRAIN_MANIFEST="${RRW_TRAIN_MANIFEST:-${ROOT}/results/rdnet_data/r5_rrw/train.csv}"
CLS_MODEL="${CLS_MODEL:-${XREFLECTION_ROOT}/pretrained/cls_model.pth}"
FOCAL_MODEL="${FOCAL_MODEL:-${XREFLECTION_ROOT}/pretrained/focal.pth}"
PRETRAIN_NETWORK_G="${PRETRAIN_NETWORK_G:-}"
REFLECTION_TARGET_MODE="${REFLECTION_TARGET_MODE:-residual}"
REFLECTION_LOWPASS_KERNEL="${REFLECTION_LOWPASS_KERNEL:-31}"
REFLECTION_LOWPASS_SIGMA="${REFLECTION_LOWPASS_SIGMA:-5.0}"
REFLECTION_LOWPASS_AUX_WEIGHT="${REFLECTION_LOWPASS_AUX_WEIGHT:-0.0}"
BASEBALL_LR="${BASEBALL_LR:-1e-4}"
OTHER_LR="${OTHER_LR:-2e-4}"
RESUME="${RESUME:-}"

IFS=',' read -r -a gpu_array <<< "${GPU_IDS}"
NUM_DEVICES="${#gpu_array[@]}"

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '[WARN] Missing %s: %s\n' "${label}" "${path}" >&2
      return
    fi
    printf 'Missing %s: %s\n' "${label}" "${path}" >&2
    exit 1
  fi
}

require_path "${XREFLECTION_ROOT}/options/train_rdnet.yml" "XReflection RDNet config"
require_path "${REAL_TRAIN_DIR}" "real training dataset"
require_path "${NATURE_TRAIN_DIR}" "nature training dataset"
require_path "${SYN_TRAIN_DIR}" "synthetic VOC training dataset"
if [[ -n "${SYN_FNS}" ]]; then
  require_path "${SYN_FNS}" "synthetic filename list"
fi
if [[ "${TRAIN_PIPELINE}" == "r5a_rrw_only" ]]; then
  require_path "${RRW_ROOT}" "R5a RRW root"
  require_path "${RRW_TRAIN_MANIFEST}" "R5a RRW train manifest"
elif [[ "${TRAIN_PIPELINE}" != "default" ]]; then
  printf 'Unsupported TRAIN_PIPELINE: %s\n' "${TRAIN_PIPELINE}" >&2
  exit 1
fi
require_path "${CLS_MODEL}" "cls_model checkpoint"
require_path "${FOCAL_MODEL}" "focal checkpoint"
if [[ -n "${PRETRAIN_NETWORK_G}" ]]; then
  require_path "${PRETRAIN_NETWORK_G}" "RDNet initial checkpoint"
fi

mkdir -p "${CONFIG_DIR}" "${RDNET_TMPDIR}" "${RDNET_MPLCONFIGDIR}" "${RDNET_TORCH_HOME}"

export XREFLECTION_ROOT RUN_NAME MAX_EPOCHS BATCH_SIZE ACCUMULATE_GRAD_BATCHES NUM_WORKERS PRECISION STRATEGY
export DDP_TIMEOUT_SECONDS SAVE_VAL_IMAGES LIMIT_TRAIN_BATCHES LIMIT_VAL_BATCHES
export VAL_CHECK_INTERVAL LOG_EVERY_N_STEPS SAVE_TOP_K SIRS_ROOT ERRNET_DATA_ROOT
export EXPERIMENTS_ROOT CONFIG_PATH REAL_TRAIN_DIR NATURE_TRAIN_DIR SYN_TRAIN_DIR SYN_FNS
export TRAIN_PIPELINE RRW_ROOT RRW_TRAIN_MANIFEST
export CLS_MODEL FOCAL_MODEL PRETRAIN_NETWORK_G NUM_DEVICES NUM_SANITY_VAL_STEPS
export REFLECTION_TARGET_MODE REFLECTION_LOWPASS_KERNEL REFLECTION_LOWPASS_SIGMA REFLECTION_LOWPASS_AUX_WEIGHT
export BASEBALL_LR OTHER_LR

python - <<'PY'
import os
from pathlib import Path
import yaml

xroot = Path(os.environ["XREFLECTION_ROOT"])
template = xroot / "options" / "train_rdnet.yml"
with template.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["name"] = os.environ["RUN_NAME"]
cfg["devices"] = int(os.environ["NUM_DEVICES"])
cfg["accelerator"] = "gpu"
cfg["precision"] = os.environ["PRECISION"]
cfg["val_check_interval"] = float(os.environ["VAL_CHECK_INTERVAL"])
cfg["log_every_n_steps"] = int(os.environ["LOG_EVERY_N_STEPS"])
cfg["lightning"]["max_epochs"] = int(os.environ["MAX_EPOCHS"])
cfg["lightning"]["strategy"] = os.environ["STRATEGY"]
cfg["lightning"]["ddp_timeout_seconds"] = int(os.environ["DDP_TIMEOUT_SECONDS"])
cfg["lightning"]["accumulate_grad_batches"] = int(os.environ["ACCUMULATE_GRAD_BATCHES"])
cfg["lightning"]["num_sanity_val_steps"] = int(os.environ["NUM_SANITY_VAL_STEPS"])
cfg["lightning"]["limit_train_batches"] = float(os.environ["LIMIT_TRAIN_BATCHES"])
cfg["lightning"]["limit_val_batches"] = float(os.environ["LIMIT_VAL_BATCHES"])
cfg["checkpoint"]["save_top_k"] = int(os.environ["SAVE_TOP_K"])
cfg["path"]["experiments_root"] = os.environ["EXPERIMENTS_ROOT"]
if os.environ["PRETRAIN_NETWORK_G"]:
    cfg["path"]["pretrain_network_g"] = os.environ["PRETRAIN_NETWORK_G"]
cfg.setdefault("val", {})["save_img"] = os.environ["SAVE_VAL_IMAGES"] == "1"

train = cfg["datasets"]["train"]
train["num_worker_per_gpu"] = int(os.environ["NUM_WORKERS"])
train["batch_size_per_gpu"] = int(os.environ["BATCH_SIZE"])
if os.environ["TRAIN_PIPELINE"] == "r5a_rrw_only":
    train["size"] = 7932
    train["fused_datasets"] = [
        {
            "name": "real-dataset",
            "ratio": 0.15,
            "type": "DSRTestDataset",
            "datadir": os.environ["REAL_TRAIN_DIR"],
            "enable_transforms": True,
        },
        {
            "name": "nature-dataset",
            "ratio": 0.15,
            "type": "DSRTestDataset",
            "datadir": os.environ["NATURE_TRAIN_DIR"],
            "enable_transforms": True,
        },
        {
            "name": "generic-synthetic-dataset",
            "ratio": 0.55,
            "type": "DSRDataset",
            "datadir": os.environ["SYN_TRAIN_DIR"],
            "fns": os.environ["SYN_FNS"] or None,
            "size": None,
            "enable_transforms": True,
        },
        {
            "name": "rrw-scene-balanced-dataset",
            "ratio": 0.15,
            "type": "RRWScenePairDataset",
            "datadir": os.environ["RRW_ROOT"],
            "manifest": os.environ["RRW_TRAIN_MANIFEST"],
            "frames_per_scene": 8,
            "enable_transforms": True,
        },
    ]
else:
    fused = train["fused_datasets"]
    fused[0]["datadir"] = os.environ["REAL_TRAIN_DIR"]
    fused[1]["datadir"] = os.environ["NATURE_TRAIN_DIR"]
    fused[2]["datadir"] = os.environ["SYN_TRAIN_DIR"]
    fused[2]["fns"] = os.environ["SYN_FNS"] or None

data_root = Path(os.environ["ERRNET_DATA_ROOT"])
cfg["datasets"]["val_datasets"] = [
    {"name": "ceilnet_table2", "type": "DSRTestDataset", "mode": "eval", "datadir": str(data_root / "testdata_CEILNET_table2")},
    {"name": "real20", "type": "DSRTestDataset", "mode": "eval", "datadir": str(data_root / "real20")},
    {"name": "postcard", "type": "DSRTestDataset", "mode": "eval", "datadir": str(data_root / "postcard")},
    {"name": "objects", "type": "DSRTestDataset", "mode": "eval", "datadir": str(data_root / "objects")},
    {"name": "wild", "type": "DSRTestDataset", "mode": "eval", "datadir": str(data_root / "wild")},
]
for val in cfg["datasets"]["val_datasets"]:
    val["io_backend"] = {"type": "disk"}
    val["use_shuffle"] = False
    val["num_worker_per_gpu"] = int(os.environ["NUM_WORKERS"])
    val["batch_size_per_gpu"] = 1

cfg["network_g"]["pretrained_models"]["cls_model"] = os.environ["CLS_MODEL"]
cfg["network_g"]["pretrained_models"]["base_network"] = os.environ["FOCAL_MODEL"]
cfg["logger"]["wandb"]["enable"] = False

reflection_target_mode = os.environ["REFLECTION_TARGET_MODE"]
reflection_lowpass_kernel = int(os.environ["REFLECTION_LOWPASS_KERNEL"])
reflection_lowpass_sigma = float(os.environ["REFLECTION_LOWPASS_SIGMA"])
reflection_lowpass_aux_weight = float(os.environ["REFLECTION_LOWPASS_AUX_WEIGHT"])
if reflection_target_mode not in {
    "residual",
    "lowpass_residual",
    "residual_lowpass_aux",
}:
    raise ValueError(
        "REFLECTION_TARGET_MODE must be residual, lowpass_residual, "
        "or residual_lowpass_aux, "
        f"got {reflection_target_mode}"
    )
if reflection_lowpass_kernel <= 0 or reflection_lowpass_kernel % 2 == 0:
    raise ValueError("REFLECTION_LOWPASS_KERNEL must be a positive odd integer")
if reflection_lowpass_sigma <= 0:
    raise ValueError("REFLECTION_LOWPASS_SIGMA must be positive")
if reflection_lowpass_aux_weight < 0:
    raise ValueError("REFLECTION_LOWPASS_AUX_WEIGHT must be non-negative")
cfg["train"]["reflection_target"] = {
    "mode": reflection_target_mode,
    "lowpass_kernel_size": reflection_lowpass_kernel,
    "lowpass_sigma": reflection_lowpass_sigma,
    "lowpass_aux_weight": reflection_lowpass_aux_weight,
}
cfg["train"]["optim_g"]["baseball_lr"] = float(os.environ["BASEBALL_LR"])
cfg["train"]["optim_g"]["other_lr"] = float(os.environ["OTHER_LR"])

config_path = Path(os.environ["CONFIG_PATH"])
config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print(config_path)
PY

cmd=(
  conda run --no-capture-output -n "${CONDA_ENV}" env
  CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
  NCCL_DEBUG="${NCCL_DEBUG}"
  NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING}"
  TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING}"
  NCCL_IB_DISABLE="${NCCL_IB_DISABLE}"
  NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE}"
  TMPDIR="${RDNET_TMPDIR}"
  MPLCONFIGDIR="${RDNET_MPLCONFIGDIR}"
  TORCH_HOME="${RDNET_TORCH_HOME}"
  PYTHONPATH="${XREFLECTION_ROOT}:${PYTHONPATH:-}"
  python -u "${XREFLECTION_ROOT}/xreflection/tools/train.py"
  --config "${CONFIG_PATH}"
)

if [[ -n "${RESUME}" ]]; then
  cmd+=(--resume "${RESUME}")
fi

printf '[RUN] %s\n' "${cmd[*]}"
if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

exec "${cmd[@]}"
