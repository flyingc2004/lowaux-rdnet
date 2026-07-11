# R6 DINO Prompt Training Guide

R6 adds a frozen DINOv3 semantic prompt to RDNet while keeping the R5 training
recipe: low-pass reflection auxiliary supervision and scene-balanced RRW real
pairs. The DINO backbone is used only as a frozen feature extractor. Only the
small prompt adapter is trained and saved in checkpoints.

## What R6 Changes

R6 keeps the RDNet transmission/reflection decoder unchanged. During training
and inference, the mixed input image is resized to `224x224` and passed through
a frozen DINOv3 model. The global DINO patch-token feature is projected to a
64-dimensional prompt delta and fused into the original RDNet prompt gate:

```text
prompt_final = prompt_base * (1 + strength * tanh(dino_delta))
```

The DINO prompt adapter is zero-initialized, so enabling R6 starts from the same
prediction as the loaded RDNet checkpoint before training.

## Required Local Resources

The repository does not track model weights or datasets. Prepare these local
paths before running R6:

```text
DINO_MODEL_PATH=/path/to/local/dinov3
PRETRAIN_NETWORK_G=/path/to/rdnet-26.4849.ckpt
RRW_ROOT=/path/to/RRW
XReflection/data/sirs/train/
ERRNet/datasets/processed_data/
XReflection/pretrained/cls_model.pth
XReflection/pretrained/focal.pth
```

For the ModelScope DINOv3 ViT-L/16 model used in local experiments:

```bash
export DINO_MODEL_PATH=/mnt/a/ljz/.tmp/lowaux-rdnet/modelscope-cache/models/facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/master
```

Verify the model path:

```bash
TMPDIR=/mnt/a/ljz/.tmp/lowaux-rdnet/tmp \
conda run --no-capture-output -n xreflection \
python - <<'PY'
import os
from transformers import AutoModel

path = os.environ["DINO_MODEL_PATH"]
model = AutoModel.from_pretrained(path, trust_remote_code=True, local_files_only=True)
print("loaded", model.config.hidden_size, getattr(model.config, "patch_size", None))
PY
```

Expected ViT-L/16 output:

```text
loaded 1024 16
```

## Four-GPU Smoke Test

Run a short four-GPU test before a full 40-epoch job. Use `STRATEGY=ddp`, not
`ddp_static_graph`, because the DINO prompt path can make static-graph DDP hang
during initialization.

```bash
cd /mnt/a/ljz/DIP/lowaux-rdnet

export DINO_MODEL_PATH=/mnt/a/ljz/.tmp/lowaux-rdnet/modelscope-cache/models/facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/master

RUN_NAME=rdnet_r6_dino_prompt_vitl_smoke_ddp4 \
PRETRAIN_NETWORK_G=/mnt/a/ljz/DIP/XReflection/pretrained/rdnet-26.4849.ckpt \
GPU_IDS=4,5,6,7 \
MAX_EPOCHS=1 \
BATCH_SIZE=1 \
ACCUMULATE_GRAD_BATCHES=1 \
NUM_WORKERS=0 \
PRECISION=bf16-mixed \
STRATEGY=ddp \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
DDP_TIMEOUT_SECONDS=28800 \
LIMIT_TRAIN_BATCHES=8 \
LIMIT_VAL_BATCHES=1 \
LOG_EVERY_N_STEPS=1 \
SAVE_TOP_K=1 \
bash scripts/train/train_rdnet_r6_dino_prompt.sh
```

The smoke test is healthy when all ranks enter training steps and TensorBoard
event files grow beyond the initial startup record.

## Full R6 Training

The clean R6 definition starts from the official RDNet checkpoint and trains for
40 epochs with R5 data and losses:

```bash
cd /mnt/a/ljz/DIP/lowaux-rdnet

export DINO_MODEL_PATH=/mnt/a/ljz/.tmp/lowaux-rdnet/modelscope-cache/models/facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/master

RUN_NAME=rdnet_r6_dino_prompt_vitl_from_official_40ep_ddp_b1 \
PRETRAIN_NETWORK_G=/mnt/a/ljz/DIP/XReflection/pretrained/rdnet-26.4849.ckpt \
GPU_IDS=4,5,6,7 \
MAX_EPOCHS=40 \
BATCH_SIZE=1 \
ACCUMULATE_GRAD_BATCHES=1 \
NUM_WORKERS=0 \
PRECISION=bf16-mixed \
STRATEGY=ddp \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
DDP_TIMEOUT_SECONDS=28800 \
LOG_EVERY_N_STEPS=10 \
SAVE_TOP_K=3 \
bash scripts/train/train_rdnet_r6_dino_prompt.sh
```

The R6 script fixes:

```text
TRAIN_PIPELINE=r5_rrw_only
REFLECTION_TARGET_MODE=residual_lowpass_aux
REFLECTION_LOWPASS_KERNEL=31
REFLECTION_LOWPASS_SIGMA=5.0
REFLECTION_LOWPASS_AUX_WEIGHT=0.2
BASEBALL_LR=5e-6
OTHER_LR=1e-5
DINO_PROMPT_STRENGTH=0.1
```

Override any of these only for an explicit ablation.

## Public Benchmark

After training, update `configs/benchmark/course_local5_r6_dino_prompt.yml`:

```yaml
models:
  r6_dino_prompt:
    checkpoint: /path/to/best-r6.ckpt
    network_g:
      dino_prompt:
        model_path: /path/to/local/dinov3
```

Then run:

```bash
cd /mnt/a/ljz/DIP/lowaux-rdnet

conda run --no-capture-output -n xreflection \
python scripts/benchmark/run_public_benchmark.py \
  --config configs/benchmark/course_local5_r6_dino_prompt.yml \
  --gpu-ids 4,5,6,7 \
  --output-dir results/public_benchmark/course_local5_r6_dino_prompt
```

Add `--resume` to continue an interrupted benchmark.

## Troubleshooting

- If training hangs before the first step, stop the run and retry with
  `STRATEGY=ddp`, `BATCH_SIZE=1`, and `NUM_WORKERS=0`.
- Ignore the warning that `NCCL_ASYNC_ERROR_HANDLING` is deprecated; the active
  variable is `TORCH_NCCL_ASYNC_ERROR_HANDLING`.
- Do not use `/tmp` for caches on machines where `/tmp` is full. Use:
  `TMPDIR=/mnt/a/ljz/.tmp/lowaux-rdnet/tmp`.
- If `transformers` is missing, install it in the `xreflection` environment:
  `conda run -n xreflection python -m pip install -U transformers accelerate`.
- R6 checkpoints should stay much smaller than the full DINO model because the
  frozen DINO backbone is not stored in RDNet checkpoints.
