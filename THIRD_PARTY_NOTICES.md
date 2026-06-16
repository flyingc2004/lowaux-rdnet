# Third-Party Notices

This repository combines course-project code with adapted research code and
uses several external datasets, pretrained weights, and model repositories.
Large assets are intentionally not redistributed through Git.

## ERRNet

- Upstream project: ERRNet / DIP26 course baseline.
- Paper: *Single Image Reflection Removal Exploiting Misaligned Training Data
  and Network Enhancements*, CVPR 2019.
- Local code location: `ERRNet/`.
- License file: `ERRNet/license`.

The local ERRNet code is included for the course baseline, E3/E5b experiments,
and the unified benchmark adapter. Users should follow the upstream license and
terms.

## XReflection / RDNet

- Upstream project: XReflection toolbox.
- RDNet paper: *Reversible Decoupling Network for Single Image Reflection
  Removal*, CVPR 2025.
- Local code location: `XReflection/`.

This repository adapts the RDNet training code for the LowAux course project.
The final method keeps the RDNet inference architecture unchanged and adds
training-time low-pass residual auxiliary supervision. Follow the upstream
XReflection/RDNet license and terms for any reuse. This repository does not
redistribute RDNet pretrained weights.

## RRW

- Dataset paper: *Revisiting Single Image Reflection Removal in the Wild*,
  CVPR 2024.
- Usage in this project: scene-balanced real paired data for R5 training.

RRW data are external and are not committed to this repository. Users must
obtain RRW from the original source and comply with its dataset terms.

## Open-Model Benchmark Dependencies

The unified public benchmark can optionally evaluate official pretrained
models from:

- IBCLN
- DSRNet
- DSIT

Their source repositories and pretrained weights are expected under
`external/models/` and `external/weights/`. They are not redistributed through
Git. Users are responsible for obtaining these projects and weights from their
original sources and following the corresponding licenses and terms.

## Local-Only Assets

The following classes of files must remain outside Git:

- datasets and benchmark images;
- model checkpoints and pretrained weights;
- training logs, benchmark outputs, and generated predictions;
- third-party model repositories cloned under `external/`.

See `.gitignore` and the root `README.md` for the expected local directory
layout.
