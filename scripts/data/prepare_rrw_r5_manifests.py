#!/usr/bin/env python3
import argparse
import csv
import json
import random
from pathlib import Path


FIELDS = ("collection", "scene_id", "scene_dir", "target_path", "frame_count")
IMAGE_EXTENSIONS = {".jpg", ".jpeg"}


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def discover_scenes(rrw_root: Path):
    scenes = []
    missing_targets = []
    for collection in sorted(path for path in rrw_root.iterdir() if path.is_dir()):
        if collection.name.startswith("."):
            continue
        gt_dirs = sorted(
            path
            for path in collection.iterdir()
            if path.is_dir() and path.name.lower().startswith("gt")
        )
        gt_by_stem = {}
        for gt_dir in gt_dirs:
            for target in sorted(gt_dir.glob("*.png")):
                gt_by_stem[target.stem.lower()] = target

        scene_dirs = sorted(
            path
            for path in collection.iterdir()
            if path.is_dir() and not path.name.lower().startswith("gt")
        )
        for scene_dir in scene_dirs:
            frames = sorted(
                path
                for path in scene_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            target = gt_by_stem.get(f"{scene_dir.name}_GT".lower())
            scene_id = f"{collection.name}/{scene_dir.name}"
            if target is None:
                missing_targets.append(scene_id)
                continue
            if not frames:
                raise ValueError(f"RRW scene has no JPEG frames: {scene_id}")
            scenes.append(
                {
                    "collection": collection.name,
                    "scene_id": scene_id,
                    "scene_dir": relative_posix(scene_dir, rrw_root),
                    "target_path": relative_posix(target, rrw_root),
                    "frame_count": len(frames),
                }
            )
    return scenes, missing_targets


def split_scenes(scenes, seed: int):
    shuffled = list(scenes)
    random.Random(seed).shuffle(shuffled)
    return {
        "train": sorted(shuffled[:147], key=lambda row: row["scene_id"]),
        "val": sorted(shuffled[147:162], key=lambda row: row["scene_id"]),
        "holdout": sorted(shuffled[162:167], key=lambda row: row["scene_id"]),
    }


def write_outputs(output_dir: Path, splits, summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        with (output_dir / f"{split}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=True)
        file.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rrw-root", type=Path, default=Path("/mnt/a/ljz/RRW"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/a/ljz/DIP/ERRNet/results/rdnet_data/r5_rrw"),
    )
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rrw_root = args.rrw_root.resolve()
    scenes, missing_targets = discover_scenes(rrw_root)
    if len(scenes) != 167:
        raise ValueError(f"Expected 167 valid RRW scenes, found {len(scenes)}")
    if missing_targets != ["ref_hf3/wild_out7"]:
        raise ValueError(f"Unexpected missing-target scenes: {missing_targets}")

    splits = split_scenes(scenes, args.seed)
    split_ids = {name: {row["scene_id"] for row in rows} for name, rows in splits.items()}
    if split_ids["train"] & split_ids["val"]:
        raise AssertionError("RRW train and val scenes overlap")
    if split_ids["train"] & split_ids["holdout"]:
        raise AssertionError("RRW train and holdout scenes overlap")
    if split_ids["val"] & split_ids["holdout"]:
        raise AssertionError("RRW val and holdout scenes overlap")

    summary = {
        "rrw_root": str(rrw_root),
        "seed": args.seed,
        "valid_scenes": len(scenes),
        "missing_target_scenes": missing_targets,
        "split_scene_counts": {name: len(rows) for name, rows in splits.items()},
        "split_frame_counts": {
            name: sum(int(row["frame_count"]) for row in rows)
            for name, rows in splits.items()
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    if not args.dry_run:
        write_outputs(args.output_dir.resolve(), splits, summary)
        print(f"Wrote RRW R5 manifests to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
