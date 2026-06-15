#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from public_benchmark.adapters import build_adapter


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "public_benchmark" / "course_local5_open_models.yml"


def verify_worker(config_path: Path, model_name: str) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20260612)
    adapter = build_adapter(config["models"][model_name], device=device, runtime=config["runtime"])

    input_tensor = torch.rand(1, 3, 65, 67, dtype=torch.float32)
    prediction = adapter.predict(input_tensor)
    if prediction.shape != input_tensor.shape:
        raise AssertionError(f"{model_name}: shape {prediction.shape} != {input_tensor.shape}")
    if prediction.dtype != torch.float32:
        raise AssertionError(f"{model_name}: expected float32, got {prediction.dtype}")
    if not torch.isfinite(prediction).all():
        raise AssertionError(f"{model_name}: prediction contains non-finite values")

    if model_name in {"ibcln_official", "dsit_official"}:
        repeated = adapter.predict(input_tensor)
        if not torch.equal(prediction, repeated):
            raise AssertionError(f"{model_name}: repeated inference is not deterministic")

    if model_name == "dsit_official":
        if adapter.checkpoint_epoch != 66 or adapter.checkpoint_iterations != 330000:
            raise AssertionError(
                f"DSIT checkpoint metadata mismatch: "
                f"epoch={adapter.checkpoint_epoch} iterations={adapter.checkpoint_iterations}"
            )
    print(f"[OK] {model_name}: shape={tuple(prediction.shape)} dtype={prediction.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify official open-model public benchmark adapters.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--worker-model",
        choices=["dsrnet_official", "dsit_official", "ibcln_official"],
    )
    args = parser.parse_args()
    config_path = args.config.resolve()

    if args.worker_model:
        verify_worker(config_path, args.worker_model)
        return

    for model_name in ("dsrnet_official", "dsit_official", "ibcln_official"):
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path), "--worker-model", model_name],
            cwd=ROOT,
            check=True,
        )
    print("[OK] official open-model adapters verified")


if __name__ == "__main__":
    main()
