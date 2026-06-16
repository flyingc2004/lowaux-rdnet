#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_TYPES = [
    "DSRTestDataset",
    "DSRTestDataset",
    "DSRDataset",
    "RRWScenePairDataset",
]
EXPECTED_RATIOS = [0.15, 0.15, 0.55, 0.15]
EXPECTED_SPLITS = {"train": 147, "val": 15, "holdout": 5}


def read_manifest(path: Path):
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def verify_manifests(manifest_dir: Path, rrw_root: Path):
    split_rows = {
        split: read_manifest(manifest_dir / f"{split}.csv")
        for split in EXPECTED_SPLITS
    }
    split_ids = {}
    for split, expected_count in EXPECTED_SPLITS.items():
        rows = split_rows[split]
        if len(rows) != expected_count:
            raise AssertionError(
                f"{split} has {len(rows)} scenes, expected {expected_count}"
            )
        split_ids[split] = {row["scene_id"] for row in rows}
        for row in rows:
            if not (rrw_root / row["scene_dir"]).is_dir():
                raise AssertionError(f"Missing RRW scene: {row['scene_id']}")
            if not (rrw_root / row["target_path"]).is_file():
                raise AssertionError(f"Missing RRW target: {row['scene_id']}")

    for left, right in (("train", "val"), ("train", "holdout"), ("val", "holdout")):
        if split_ids[left] & split_ids[right]:
            raise AssertionError(f"RRW {left}/{right} scene overlap")

    with (manifest_dir / "summary.json").open("r", encoding="utf-8") as file:
        summary = json.load(file)
    if summary["valid_scenes"] != 167:
        raise AssertionError("RRW summary does not contain 167 valid scenes")
    if summary["missing_target_scenes"] != ["ref_hf3/wild_out7"]:
        raise AssertionError("RRW missing-target scene list changed")
    return split_rows


def assert_sample(sample, label):
    for key in ("input", "target_t", "target_r"):
        tensor = sample[key]
        if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
            raise AssertionError(f"{label} {key} must be CPU float32")
    if sample["input"].shape != sample["target_t"].shape:
        raise AssertionError(f"{label} input/target_t shapes differ")
    if sample["input"].shape != sample["target_r"].shape:
        raise AssertionError(f"{label} input/target_r shapes differ")
    if not torch.allclose(
        sample["target_r"], sample["input"] - sample["target_t"], atol=1e-7, rtol=0
    ):
        raise AssertionError(f"{label} target_r is not input-target_t")


def verify_config(config_path: Path, xreflection_root: Path):
    with config_path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    train = cfg["datasets"]["train"]
    if int(train.get("size", -1)) != 7932:
        raise AssertionError("R5 train dataset size is not 7932")

    fused = train["fused_datasets"]
    types = [dataset["type"] for dataset in fused]
    ratios = [float(dataset["ratio"]) for dataset in fused]
    if types != EXPECTED_TYPES:
        raise AssertionError(f"Unexpected R5 dataset types: {types}")
    if not np.allclose(ratios, EXPECTED_RATIOS, atol=1e-12, rtol=0):
        raise AssertionError(f"Unexpected R5 dataset ratios: {ratios}")
    if abs(sum(ratios) - 1.0) > 1e-12:
        raise AssertionError("R5 dataset ratios do not sum to 1")
    if any(dataset["type"] == "SharpReflectionDataset" for dataset in fused):
        raise AssertionError("R5 must not include SharpReflectionDataset")

    forbidden = ("/postcard", "/objects", "/wild", "/mnt/a/ljz/CG", "/syn13700")
    for dataset in fused:
        for key in ("datadir", "manifest"):
            value = str(dataset.get(key, "")).lower()
            if any(token.lower() in value for token in forbidden):
                raise AssertionError(f"Forbidden R5 training source: {value}")

    expected_reflection = {
        "mode": "residual_lowpass_aux",
        "lowpass_kernel_size": 31,
        "lowpass_sigma": 5.0,
        "lowpass_aux_weight": 0.2,
    }
    if cfg["train"]["reflection_target"] != expected_reflection:
        raise AssertionError(
            f"Unexpected R5 reflection config: {cfg['train']['reflection_target']}"
        )

    sys.path.insert(0, str(xreflection_root))
    from xreflection.data.rdnet_dataset import RRWScenePairDataset

    rrw_opt = dict(fused[3])
    rrw_opt["transform_size"] = 96
    rrw_dataset = RRWScenePairDataset(rrw_opt)
    if len(rrw_dataset) != 147 * 8:
        raise AssertionError(f"RRW train samples are {len(rrw_dataset)}, expected 1176")
    assert_sample(rrw_dataset[0], "RRW")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xreflection-root",
        type=Path,
        default=REPO_ROOT / "XReflection",
    )
    parser.add_argument("--rrw-root", type=Path, default=REPO_ROOT / "RRW")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=REPO_ROOT / "results" / "rdnet_data" / "r5_rrw",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "rdnet_configs"
        / "rdnet_r5_rrw_only_from_r4_e1.yml",
    )
    args = parser.parse_args()

    split_rows = verify_manifests(args.manifest_dir, args.rrw_root)
    verify_config(args.config, args.xreflection_root)
    print(
        "R5 data verification passed: "
        f"splits={{{', '.join(f'{k}: {len(v)}' for k, v in split_rows.items())}}}, "
        f"ratios={EXPECTED_RATIOS}"
    )


if __name__ == "__main__":
    main()
