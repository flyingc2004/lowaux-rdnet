#!/usr/bin/env python3
"""Summarize lightweight lambda_low sweep runs without loading checkpoints."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PSNR_RE = re.compile(r"psnr=([0-9]+(?:\.[0-9]+)?)")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data or {}


def parse_psnr(path: Path) -> float | None:
    match = PSNR_RE.search(path.as_posix())
    if not match:
        return None
    return float(match.group(1))


def best_checkpoint(run_dir: Path) -> tuple[Path | None, float | None, int]:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None, None, 0

    checkpoints = [
        path
        for path in checkpoint_dir.rglob("*.ckpt")
        if path.name != "last.ckpt"
    ]
    if not checkpoints:
        last = checkpoint_dir / "last.ckpt"
        return (last, None, 1) if last.exists() else (None, None, 0)

    def sort_key(path: Path) -> tuple[int, float, float]:
        psnr = parse_psnr(path)
        return (1 if psnr is not None else 0, psnr or float("-inf"), path.stat().st_mtime)

    best = max(checkpoints, key=sort_key)
    return best, parse_psnr(best), len(checkpoints)


def discover_run_names(experiments_root: Path, config_dir: Path, run_prefix: str) -> list[str]:
    names = set()
    if experiments_root.exists():
        names.update(path.name for path in experiments_root.iterdir() if path.is_dir() and path.name.startswith(run_prefix))
    if config_dir.exists():
        names.update(path.stem for path in config_dir.glob(f"{run_prefix}*.yml"))
    return sorted(names)


def summarize(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, Any]] = []
    for run_name in discover_run_names(args.experiments_root, args.config_dir, args.run_prefix):
        config_path = args.config_dir / f"{run_name}.yml"
        cfg = read_yaml(config_path) if config_path.exists() else {}
        reflection_target = cfg.get("train", {}).get("reflection_target", {})
        lightning = cfg.get("lightning", {})
        pretrain = cfg.get("path", {}).get("pretrain_network_g", "")

        ckpt, psnr, count = best_checkpoint(args.experiments_root / run_name)
        rows.append(
            {
                "run_name": run_name,
                "lambda_low": reflection_target.get("lowpass_aux_weight", ""),
                "checkpoint_psnr": psnr,
                "best_checkpoint": str(ckpt) if ckpt else "",
                "checkpoint_count": count,
                "config_path": str(config_path) if config_path.exists() else "",
                "max_epochs": lightning.get("max_epochs", ""),
                "limit_train_batches": lightning.get("limit_train_batches", ""),
                "limit_val_batches": lightning.get("limit_val_batches", ""),
                "pretrain_network_g": pretrain,
            }
        )

    rows.sort(
        key=lambda row: (
            row["checkpoint_psnr"] is not None,
            row["checkpoint_psnr"] if row["checkpoint_psnr"] is not None else float("-inf"),
            -float(row["lambda_low"]) if row["lambda_low"] != "" else float("-inf"),
        ),
        reverse=True,
    )

    output_rows: list[dict[str, str]] = []
    for rank, row in enumerate(rows, start=1):
        output_rows.append(
            {
                "rank": str(rank),
                "run_name": str(row["run_name"]),
                "lambda_low": str(row["lambda_low"]),
                "checkpoint_psnr": "" if row["checkpoint_psnr"] is None else f"{row['checkpoint_psnr']:.4f}",
                "best_checkpoint": str(row["best_checkpoint"]),
                "checkpoint_count": str(row["checkpoint_count"]),
                "config_path": str(row["config_path"]),
                "max_epochs": str(row["max_epochs"]),
                "limit_train_batches": str(row["limit_train_batches"]),
                "limit_val_batches": str(row["limit_val_batches"]),
                "pretrain_network_g": str(row["pretrain_network_g"]),
            }
        )
    return output_rows


def write_csv(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "run_name",
        "lambda_low",
        "checkpoint_psnr",
        "best_checkpoint",
        "checkpoint_count",
        "config_path",
        "max_epochs",
        "limit_train_batches",
        "limit_val_batches",
        "pretrain_network_g",
    ]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", type=Path, default=REPO_ROOT / "XReflection" / "experiments")
    parser.add_argument("--config-dir", type=Path, default=REPO_ROOT / "results" / "rdnet_configs")
    parser.add_argument("--run-prefix", default="rdnet_r4_lowlambda_")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "lambda_low_sweep" / "r4_small_summary.csv")
    args = parser.parse_args()

    rows = summarize(args)
    write_csv(rows, args.output)

    print(f"Wrote {args.output}")
    if not rows:
        print("No matching runs found.")
        return
    for row in rows[:10]:
        print(
            f"#{row['rank']} lambda={row['lambda_low']} "
            f"psnr={row['checkpoint_psnr'] or 'NA'} run={row['run_name']}"
        )


if __name__ == "__main__":
    main()
