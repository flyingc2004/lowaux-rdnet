#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
import yaml


def assert_close(value: float, expected: float, tol: float, message: str) -> None:
    if abs(value - expected) > tol:
        raise AssertionError(f"{message}: got {value}, expected {expected}")


def verify_lowpass_helpers(xreflection_root: Path) -> None:
    sys.path.insert(0, str(xreflection_root))
    from xreflection.models.rdnet_model import (
        build_gaussian_kernel,
        build_reflection_target,
        gaussian_lowpass,
    )

    kernel = build_gaussian_kernel(31, 5.0, device=torch.device("cpu"), dtype=torch.float32)
    assert kernel.shape == (31, 31)
    assert kernel.dtype == torch.float32
    assert_close(float(kernel.sum()), 1.0, 1e-6, "Gaussian kernel is not normalized")

    inp = torch.zeros(1, 3, 64, 64, dtype=torch.float32)
    inp[:, :, ::2, :] = 1.0
    inp.requires_grad_(True)
    target_t = torch.zeros_like(inp)
    target_r = torch.randn_like(inp)

    default_target, raw_residual = build_reflection_target(
        inp, target_t, target_r, mode="residual"
    )
    if default_target is not target_r:
        raise AssertionError("residual mode must return batch target_r unchanged")
    if raw_residual.requires_grad:
        raise AssertionError("raw residual target should be detached")

    lowpass_target, raw_residual = build_reflection_target(
        inp,
        target_t,
        target_r,
        mode="lowpass_residual",
        lowpass_kernel_size=31,
        lowpass_sigma=5.0,
    )
    if lowpass_target.shape != inp.shape:
        raise AssertionError("lowpass target shape changed")
    if lowpass_target.dtype != inp.dtype:
        raise AssertionError("lowpass target dtype changed")
    if lowpass_target.device != inp.device:
        raise AssertionError("lowpass target device changed")
    if lowpass_target.requires_grad:
        raise AssertionError("lowpass target should not require gradients")
    if torch.allclose(lowpass_target, raw_residual):
        raise AssertionError("lowpass target unexpectedly equals raw I-T residual")

    direct_lowpass = gaussian_lowpass(raw_residual, 31, 5.0)
    if not torch.allclose(lowpass_target, direct_lowpass):
        raise AssertionError("reflection target helper does not use gaussian_lowpass")

    aux_target, _ = build_reflection_target(
        inp,
        target_t,
        target_r,
        mode="residual_lowpass_aux",
        lowpass_kernel_size=31,
        lowpass_sigma=5.0,
    )
    if aux_target is not target_r:
        raise AssertionError("residual_lowpass_aux mode must keep raw target_r as main target")

    out_r = torch.randn_like(inp, requires_grad=True)
    aux_loss = torch.nn.functional.mse_loss(
        gaussian_lowpass(out_r, 31, 5.0),
        gaussian_lowpass(raw_residual, 31, 5.0),
    )
    aux_loss.backward()
    if out_r.grad is None or not bool(torch.isfinite(out_r.grad).all()):
        raise AssertionError("lowpass auxiliary loss did not produce finite output gradients")


def verify_config(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    reflection_target = cfg["train"].get("reflection_target", {})
    mode = reflection_target.get("mode")
    if mode not in {"lowpass_residual", "residual_lowpass_aux"}:
        raise AssertionError(
            f"{config_path}: reflection target mode is not a lowpass experiment"
        )
    if int(reflection_target.get("lowpass_kernel_size")) != 31:
        raise AssertionError(f"{config_path}: lowpass kernel size is not 31")
    if float(reflection_target.get("lowpass_sigma")) != 5.0:
        raise AssertionError(f"{config_path}: lowpass sigma is not 5.0")
    expected_aux_weight = 0.2 if mode == "residual_lowpass_aux" else 0.0
    assert_close(
        float(reflection_target.get("lowpass_aux_weight", 0.0)),
        expected_aux_weight,
        1e-12,
        "lowpass_aux_weight mismatch",
    )
    optim_g = cfg["train"]["optim_g"]
    expected_baseball_lr = 5e-6 if mode == "residual_lowpass_aux" else 1e-5
    expected_other_lr = 1e-5 if mode == "residual_lowpass_aux" else 2e-5
    assert_close(float(optim_g["baseball_lr"]), expected_baseball_lr, 1e-12, "baseball_lr mismatch")
    assert_close(float(optim_g["other_lr"]), expected_other_lr, 1e-12, "other_lr mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xreflection-root",
        type=Path,
        default=Path("/mnt/a/ljz/DIP/XReflection"),
    )
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    verify_lowpass_helpers(args.xreflection_root)
    if args.config:
        verify_config(args.config)


if __name__ == "__main__":
    main()
