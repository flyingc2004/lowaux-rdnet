# Low-Pass Auxiliary Supervision for RDNet

Course-project repository for single-image reflection removal. It contains the
ERRNet baseline and E5b refinement implementation, the final RDNet R5 training
pipeline, and a unified public benchmark.

The final R5 method keeps the RDNet architecture unchanged and trains it with:

- the original raw reflection supervision;
- a Gaussian low-pass reflection auxiliary loss with weight `0.2`;
- scene-balanced paired frames from RRW.

Model weights, datasets, experiment logs, and benchmark outputs are intentionally
excluded from Git.

## Repository Layout

```text
ERRNet/                 ERRNet baseline and E5b model/training source
XReflection/            XReflection package containing the RDNet/R5 implementation
scripts/train/          Final training and environment preparation entrypoints
scripts/data/           RRW scene-manifest preparation
scripts/benchmark/      Unified public benchmark and model adapters
scripts/verify/         R5 loss and data-pipeline verification
configs/benchmark/      Fixed course-local-five benchmark configuration
manifests/              Small filename lists safe to track in Git
external/               Local-only weights, datasets, and third-party repositories
```

## Environment

Create the ERRNet and XReflection environments according to their original
requirements. The RDNet preparation script can install the XReflection
environment and download official RDNet dependencies:

```bash
bash scripts/train/prepare_rdnet.sh
```

The following local-only resources are required for the complete project:

```text
ERRNet/checkpoints/errnet/errnet_060_00463920.pt
ERRNet/checkpoints/errnet_e3_long/errnet_latest.pt
XReflection/data/sirs/
XReflection/pretrained/cls_model.pth
XReflection/pretrained/focal.pth
external/weights/rdnet_r4_epoch1.ckpt
external/weights/rdnet_r5_final.ckpt
external/datasets/course_local5/
external/models/{IBCLN,DSRNet,DSIT}/
```

These paths are ignored by Git. Override them through the environment variables
documented in the scripts when using another directory layout.

## ERRNet Baseline and E5b

Run the original ERRNet baseline training from `ERRNet/`:

```bash
cd ERRNet
python train_errnet.py --name errnet --hyper
```

Run the final E5b two-stage refinement pipeline from the repository root:

```bash
bash scripts/train/train_errnet_e5b.sh
```

E5b first freezes the ERRNet backbone and trains the dilated refiner, then
unfreezes the complete model for joint fine-tuning. Set `STAGE1_INIT_CKPT` when
the E3-long initialization checkpoint is stored elsewhere.

## Final RDNet R5 Training

R5 uses the fixed mixture:

```text
Real89 15% | Nature200 15% | VOC synthetic 55% | scene-balanced RRW 15%
```

RRW is split by scene with seed `20260611` into 147 training, 15 validation,
and 5 holdout scenes. Eight frames are sampled from each training scene per
epoch.

Run the final 40-epoch R5 training:

```bash
RRW_ROOT=/path/to/RRW \
PRETRAIN_NETWORK_G=/path/to/rdnet_r4_epoch1.ckpt \
GPU_IDS=0,1,2,3 \
bash scripts/train/train_rdnet_r5.sh
```

The script fixes R5 to `residual_lowpass_aux`, Gaussian kernel `31`,
sigma `5.0`, auxiliary weight `0.2`, and saves the best three checkpoints.

## Unified Public Benchmark

The benchmark evaluates ERRNet E0, ERRNet E5b, IBCLN, DSRNet, DSIT, RDNet
official, and R5 through model-specific adapters and one shared evaluator.
It computes float-domain PSNR, SSIM, NCC, and LMSE on CEILNet Table2, Real20,
Postcard, Objects, and Wild.

Place external resources according to
`configs/benchmark/course_local5_open_models.yml`, then run:

```bash
PYTHONPATH="$PWD/ERRNet:$PWD/scripts/benchmark" \
conda run --no-capture-output -n xreflection \
python scripts/benchmark/run_public_benchmark.py \
  --config configs/benchmark/course_local5_open_models.yml \
  --gpu-ids 0,1,2,3 \
  --output-dir results/public_benchmark/course_local5
```

Resume an interrupted benchmark by adding `--resume`.

## Verification

Static and configuration checks:

```bash
bash -n scripts/train/*.sh

python -m py_compile \
  scripts/data/prepare_rrw_r5_manifests.py \
  scripts/verify/verify_rdnet_lowpass.py \
  scripts/verify/verify_rdnet_r5a_data.py \
  scripts/benchmark/run_public_benchmark.py \
  scripts/benchmark/public_benchmark/adapters.py

DRY_RUN=1 RRW_ROOT=/path/to/RRW bash scripts/train/train_rdnet_r5.sh
```

## Source Attribution

- ERRNet baseline: CVPR 2019, *Single Image Reflection Removal with
  Perceptual Losses*.
- XReflection/RDNet: CVPR 2025, *Location-aware Single Image Reflection
  Removal*.
- RRW paired real-world data: CVPR 2024.

Third-party model repositories and pretrained weights remain external and are
used only by the unified benchmark adapters.
