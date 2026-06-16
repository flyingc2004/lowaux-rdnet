#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/ljz-public-benchmark-mpl"

import yaml


METRICS = ["PSNR", "SSIM", "NCC", "LMSE"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}


def parse_csv(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolved_model_spec(model_spec: dict) -> dict:
    resolved = dict(model_spec)
    if "checkpoint" in resolved:
        resolved["checkpoint"] = str(Path(resolved["checkpoint"]).resolve())
    if "checkpoints" in resolved:
        resolved["checkpoints"] = {
            name: str(Path(path).resolve())
            for name, path in resolved["checkpoints"].items()
        }
    return resolved


def model_weight_summary(model_spec: dict) -> str:
    resolved = resolved_model_spec(model_spec)
    if "checkpoint" in resolved:
        return resolved["checkpoint"]
    summary = {"checkpoints": resolved.get("checkpoints", {})}
    if "alpha" in resolved:
        summary["alpha"] = resolved["alpha"]
    return json.dumps(summary, sort_keys=True)


def resolve_selected(
    config: dict,
    models: str | None,
    datasets: str | None,
    manifest_datasets: set[str],
) -> tuple[list[str], list[str]]:
    selected_models = parse_csv(models) or list(config["models"])
    selected_datasets = parse_csv(datasets) or list(config["datasets"])
    missing_models = [name for name in selected_models if name not in config["models"]]
    known_datasets = set(config["datasets"]) | manifest_datasets
    missing_datasets = [name for name in selected_datasets if name not in known_datasets]
    if missing_models:
        raise ValueError(f"Unknown models: {missing_models}")
    if missing_datasets:
        raise ValueError(f"Unknown datasets: {missing_datasets}")
    return selected_models, selected_datasets


def resize_long_edge(img, max_long_edge: int | None):
    from PIL import Image

    if max_long_edge is None or max(img.size) <= max_long_edge:
        return img
    width, height = img.size
    scale = max_long_edge / float(max(width, height))
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return img.resize(new_size, Image.Resampling.BICUBIC)


def standard_dataset_pairs(dataset_name: str, spec: dict, max_samples: int | None) -> list[dict]:
    root = Path(spec["root"]).resolve()
    input_dir = root / spec.get("input_subdir", "blended")
    target_dir = root / spec.get("target_subdir", "transmission_layer")
    if not input_dir.is_dir() or not target_dir.is_dir():
        raise FileNotFoundError(f"Expected paired directories under {root}")
    input_names = {path.name for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    target_names = {path.name for path in target_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    if input_names != target_names:
        raise RuntimeError(
            f"{dataset_name} input/GT mismatch: "
            f"missing_gt={sorted(input_names - target_names)[:5]} "
            f"missing_input={sorted(target_names - input_names)[:5]}"
        )
    names = sorted(input_names)
    if max_samples is not None:
        names = names[:max_samples]
    return [
        {
            "dataset": dataset_name,
            "sample_id": Path(name).stem,
            "input_path": str((input_dir / name).resolve()),
            "target_path": str((target_dir / name).resolve()),
            "max_long_edge": spec.get("max_long_edge"),
        }
        for name in names
    ]


def manifest_pairs(path: Path, max_samples: int | None) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pair = {
                "dataset": row["dataset"],
                "sample_id": row["sample_id"],
                "input_path": str(Path(row["input_path"]).resolve()),
                "target_path": str(Path(row["target_path"]).resolve()),
                "max_long_edge": int(row["max_long_edge"]) if row.get("max_long_edge") else None,
            }
            grouped.setdefault(pair["dataset"], []).append(pair)
    for name, pairs in grouped.items():
        pairs.sort(key=lambda pair: pair["sample_id"])
        if max_samples is not None:
            grouped[name] = pairs[:max_samples]
    return grouped


def build_pair_map(config: dict, selected_datasets: list[str], manifests: list[Path], max_samples: int | None) -> dict:
    pair_map = {
        name: standard_dataset_pairs(name, config["datasets"][name], max_samples)
        for name in selected_datasets if name in config["datasets"]
    }
    for manifest in manifests:
        for name, pairs in manifest_pairs(manifest, max_samples).items():
            if name in selected_datasets:
                pair_map[name] = pairs
    missing = [name for name in selected_datasets if name not in pair_map]
    if missing:
        raise RuntimeError(f"No manifest pairs provided for datasets: {missing}")
    for name, pairs in pair_map.items():
        if not pairs:
            raise RuntimeError(f"No pairs found for {name}")
    return pair_map


def pil_to_tensor(img):
    import numpy as np
    import torch

    array = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def tensor_to_metric_array(tensor):
    return tensor.squeeze(0).permute(1, 2, 0).numpy().astype("float32") * 255.0


def save_prediction(path: Path, prediction) -> None:
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    display = np.rint(prediction.squeeze(0).permute(1, 2, 0).numpy() * 255.0)
    Image.fromarray(np.clip(display, 0, 255).astype("uint8")).save(path)


def evaluate_model(args, config: dict, model_name: str, pair_map: dict[str, list[dict]]) -> None:
    import torch
    from PIL import Image
    from public_benchmark.adapters import build_adapter
    from util.index import quality_assess

    output_dir = Path(args.output_dir).resolve()
    result_root = output_dir / ".model_results" / model_name
    result_root.mkdir(parents=True, exist_ok=True)
    model_spec = config["models"][model_name]
    metadata_path = result_root / "metadata.json"
    resolved_spec = resolved_model_spec(model_spec)
    expected_metadata = {
        "model": model_name,
        "adapter": model_spec["adapter"],
    }
    if "checkpoint" in resolved_spec:
        expected_metadata["checkpoint"] = resolved_spec["checkpoint"]
    else:
        expected_metadata["checkpoints"] = resolved_spec["checkpoints"]
    if args.resume and metadata_path.exists():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing_metadata != expected_metadata:
            raise RuntimeError(
                f"Resume metadata mismatch for {model_name}: "
                f"{existing_metadata} != {expected_metadata}"
            )
    metadata_path.write_text(json.dumps(expected_metadata, indent=2) + "\n", encoding="utf-8")

    pending = []
    for dataset_name, pairs in pair_map.items():
        result_path = result_root / f"{dataset_name}.json"
        if not (args.resume and result_path.exists()):
            pending.append(dataset_name)
            continue
        existing_rows = json.loads(result_path.read_text(encoding="utf-8"))
        expected_ids = [pair["sample_id"] for pair in pairs]
        existing_ids = [row["sample_id"] for row in existing_rows]
        if existing_ids != expected_ids:
            pending.append(dataset_name)
            print(f"[RERUN] {model_name} :: {dataset_name} sample list changed", flush=True)
    for dataset_name in pair_map:
        if dataset_name not in pending:
            print(f"[SKIP] {model_name} :: {dataset_name}", flush=True)
    if not pending:
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adapter = build_adapter(config["models"][model_name], device=device, runtime=config["runtime"])

    for dataset_name, pairs in pair_map.items():
        if dataset_name not in pending:
            continue
        result_path = result_root / f"{dataset_name}.json"

        rows = []
        for index, pair in enumerate(pairs, start=1):
            input_img = Image.open(pair["input_path"]).convert("RGB")
            target_img = Image.open(pair["target_path"]).convert("RGB")
            input_img = resize_long_edge(input_img, pair["max_long_edge"])
            target_img = resize_long_edge(target_img, pair["max_long_edge"])
            if input_img.size != target_img.size:
                raise RuntimeError(
                    f"{dataset_name}/{pair['sample_id']} input/GT size mismatch: "
                    f"{input_img.size} vs {target_img.size}"
                )

            input_tensor = pil_to_tensor(input_img)
            target_tensor = pil_to_tensor(target_img)
            prediction = adapter.predict(input_tensor)
            if prediction.dtype != torch.float32:
                prediction = prediction.float()
            prediction = prediction.clamp(0.0, 1.0)
            if prediction.shape != target_tensor.shape:
                raise RuntimeError(
                    f"{model_name}/{dataset_name}/{pair['sample_id']} prediction shape "
                    f"{tuple(prediction.shape)} != target {tuple(target_tensor.shape)}"
                )

            assessed = quality_assess(
                tensor_to_metric_array(prediction),
                tensor_to_metric_array(target_tensor),
            )
            rows.append({
                "model": model_name,
                "dataset": dataset_name,
                "sample_id": pair["sample_id"],
                **{metric: float(assessed[metric]) for metric in METRICS},
            })
            save_prediction(
                output_dir / "predictions" / model_name / dataset_name / f"{pair['sample_id']}.png",
                prediction,
            )
            print(
                f"[IMG] {model_name} {dataset_name} {index}/{len(pairs)} "
                f"PSNR={assessed['PSNR']:.4f}",
                flush=True,
            )

        result_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] {model_name} :: {dataset_name}", flush=True)


def average(rows: list[dict], metric: str) -> float:
    return sum(float(row[metric]) for row in rows) / len(rows)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_results(
    config: dict,
    output_dir: Path,
    selected_models: list[str],
    selected_datasets: list[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    image_rows = []
    for model_name in selected_models:
        for dataset_name in selected_datasets:
            path = output_dir / ".model_results" / model_name / f"{dataset_name}.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing benchmark result: {path}")
            image_rows.extend(json.loads(path.read_text(encoding="utf-8")))
    image_rows.sort(key=lambda row: (
        selected_models.index(row["model"]),
        selected_datasets.index(row["dataset"]),
        row["sample_id"],
    ))

    dataset_rows = []
    for model_name in selected_models:
        for dataset_name in selected_datasets:
            rows = [
                row for row in image_rows
                if row["model"] == model_name and row["dataset"] == dataset_name
            ]
            dataset_rows.append({
                "model": model_name,
                "label": config["models"][model_name]["label"],
                "dataset": dataset_name,
                "samples": len(rows),
                **{f"mean_{metric}": average(rows, metric) for metric in METRICS},
            })

    macro_datasets = [
        name for name in config["benchmark"]["macro_datasets"]
        if name in selected_datasets
    ]
    macro_rows = []
    if macro_datasets:
        for model_name in selected_models:
            rows = [
                row for row in dataset_rows
                if row["model"] == model_name and row["dataset"] in macro_datasets
            ]
            macro_rows.append({
                "model": model_name,
                "label": config["models"][model_name]["label"],
                "datasets": len(rows),
                **{f"mean_{metric}": average(rows, f"mean_{metric}") for metric in METRICS},
            })
    return image_rows, dataset_rows, macro_rows


def delta_rows(macro_rows: list[dict], baseline: str) -> list[dict]:
    indexed = {row["model"]: row for row in macro_rows}
    if baseline not in indexed:
        return []
    base = indexed[baseline]
    return [
        {
            "model": row["model"],
            "label": row["label"],
            "baseline": baseline,
            **{
                f"delta_{metric}": float(row[f"mean_{metric}"]) - float(base[f"mean_{metric}"])
                for metric in METRICS
            },
        }
        for row in macro_rows
    ]


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Model | Mean PSNR | Mean SSIM | Mean NCC | Mean LMSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {} | {:.5f} | {:.5f} | {:.5f} | {:.6f} |".format(
                row["label"],
                row["mean_PSNR"],
                row["mean_SSIM"],
                row["mean_NCC"],
                row["mean_LMSE"],
            )
        )
    return "\n".join(lines)


def contribution_line(indexed: dict[str, dict], start: str, end: str, label: str) -> str:
    if start not in indexed or end not in indexed:
        return f"- {label}: not evaluated"
    return "- {}: PSNR {:+.5f} dB, SSIM {:+.5f}, NCC {:+.5f}, LMSE {:+.6f}".format(
        label,
        indexed[end]["mean_PSNR"] - indexed[start]["mean_PSNR"],
        indexed[end]["mean_SSIM"] - indexed[start]["mean_SSIM"],
        indexed[end]["mean_NCC"] - indexed[start]["mean_NCC"],
        indexed[end]["mean_LMSE"] - indexed[start]["mean_LMSE"],
    )


def write_reports(config: dict, output_dir: Path, image_rows: list[dict], dataset_rows: list[dict], macro_rows: list[dict]) -> None:
    write_csv(
        output_dir / "image_metrics.csv",
        image_rows,
        ["model", "dataset", "sample_id"] + METRICS,
    )
    write_csv(
        output_dir / "dataset_summary.csv",
        dataset_rows,
        ["model", "label", "dataset", "samples"] + [f"mean_{metric}" for metric in METRICS],
    )
    write_csv(
        output_dir / "macro_summary.csv",
        macro_rows,
        ["model", "label", "datasets"] + [f"mean_{metric}" for metric in METRICS],
    )
    for baseline in ("errnet_e0", "rdnet_reproduced"):
        rows = delta_rows(macro_rows, baseline)
        if rows:
            write_csv(
                output_dir / f"deltas_vs_{baseline}.csv",
                rows,
                ["model", "label", "baseline"] + [f"delta_{metric}" for metric in METRICS],
            )

    indexed = {row["model"]: row for row in macro_rows}
    table_groups = config["benchmark"].get("paper_tables", {"All Evaluated Models": list(indexed)})
    text = [
        "# Public Benchmark Paper Tables",
        "",
    ]
    for title, model_names in table_groups.items():
        rows = [indexed[name] for name in model_names if name in indexed]
        if rows:
            text.extend([f"## {title}", "", markdown_table(rows), ""])
    paper_notes = config["benchmark"].get("paper_notes", [])
    if paper_notes:
        text.extend(["## Interpretation Notes", ""])
        text.extend(f"- {note}" for note in paper_notes)
        text.append("")
    contributions = config["benchmark"].get("contributions", [
        {
            "start": "errnet_e0",
            "end": "rdnet_reproduced",
            "label": "ERRNet E0 -> RDNet reproduced (architecture replacement)",
        },
        {
            "start": "rdnet_reproduced",
            "end": "r4",
            "label": "RDNet reproduced -> R4 (low-pass auxiliary supervision)",
        },
        {
            "start": "errnet_e0",
            "end": "r5_final",
            "label": "ERRNet E0 -> R5 (overall improvement)",
        },
    ])
    text.extend(["## Contribution Decomposition", ""])
    text.extend(
        contribution_line(indexed, item["start"], item["end"], item["label"])
        for item in contributions
    )
    text.append("")
    (output_dir / "paper_tables.md").write_text("\n".join(text), encoding="utf-8")


def make_visualizations(config: dict, output_dir: Path, pair_map: dict, selected_models: list[str]) -> None:
    from PIL import Image, ImageDraw

    requested = [
        name for name in config["benchmark"]["visualization_models"]
        if name in selected_models
    ]
    count = int(config["benchmark"].get("visualization_samples_per_dataset", 3))
    for dataset_name, pairs in pair_map.items():
        for pair in pairs[:count]:
            panels = [
                ("Input", Path(pair["input_path"])),
                ("GT", Path(pair["target_path"])),
            ]
            panels.extend([
                (
                    config["models"][name]["label"],
                    output_dir / "predictions" / name / dataset_name / f"{pair['sample_id']}.png",
                )
                for name in requested
            ])
            if any(not path.exists() for _, path in panels):
                continue
            images = []
            for label, path in panels:
                img = Image.open(path).convert("RGB")
                if label in {"Input", "GT"}:
                    img = resize_long_edge(img, pair["max_long_edge"])
                scale = min(1.0, 320.0 / img.height)
                img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))
                images.append((label, img))
            width = sum(img.width for _, img in images)
            canvas = Image.new("RGB", (width, max(img.height for _, img in images) + 28), "white")
            draw = ImageDraw.Draw(canvas)
            x = 0
            for label, img in images:
                canvas.paste(img, (x, 28))
                draw.text((x + 4, 6), label, fill="black")
                x += img.width
            path = output_dir / "visualizations" / dataset_name / f"{pair['sample_id']}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(path)


def git_revision(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() or None


def package_versions() -> dict:
    import PIL
    import numpy
    import skimage
    import torch

    return {
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "scikit_image": skimage.__version__,
        "pillow": PIL.__version__,
    }


def write_manifest(
    args,
    config: dict,
    output_dir: Path,
    selected_models: list[str],
    selected_datasets: list[str],
    pair_map: dict,
) -> None:
    from public_benchmark.adapters import (
        DSITAdapter,
        DSRNetAdapter,
        ERRNetAdapter,
        ERRNetFusionAdapter,
        IBCLNAdapter,
        RDNetAdapter,
    )

    root = Path(__file__).resolve().parent
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "config": str(Path(args.config).resolve()),
        "config_snapshot": config,
        "models": {
            name: resolved_model_spec(config["models"][name])
            for name in selected_models
        },
        "datasets": selected_datasets,
        "pairs": pair_map,
        "macro_datasets": [
            name for name in config["benchmark"]["macro_datasets"]
            if name in selected_datasets
        ],
        "adapter_behavior": {
            "errnet": ERRNetAdapter.behavior,
            "errnet_fusion": ERRNetFusionAdapter.behavior,
            "rdnet": RDNetAdapter.behavior,
            "dsrnet": DSRNetAdapter.behavior,
            "dsit": DSITAdapter.behavior,
            "ibcln": IBCLNAdapter.behavior,
        },
        "metric_protocol": {
            "prediction": "clip float32 tensor to [0,1]; no quantization before metrics",
            "metric_input": "HWC float32 [0,255]",
            "metrics": "util.index.quality_assess",
            "aggregation": "image mean per dataset, then equal-weight macro mean",
        },
        "git_revisions": {
            "errnet": git_revision(root),
            "xreflection": git_revision(Path(config["runtime"]["xreflection_root"])),
            "dsrnet": git_revision(Path(config["runtime"]["dsrnet_root"]))
            if config["runtime"].get("dsrnet_root") else None,
            "dsit": git_revision(Path(config["runtime"]["dsit_root"]))
            if config["runtime"].get("dsit_root") else None,
            "ibcln": git_revision(Path(config["runtime"]["ibcln_root"]))
            if config["runtime"].get("ibcln_root") else None,
        },
        "versions": package_versions(),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def worker_loop(args, config_path: Path, tasks: queue.Queue, errors: list[str], lock: threading.Lock, gpu_id: str) -> None:
    while True:
        model_name = tasks.get()
        if model_name is None:
            tasks.task_done()
            return
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["MPLCONFIGDIR"] = "/tmp/ljz-public-benchmark-mpl"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config", str(config_path),
            "--output-dir", str(args.output_dir),
            "--models", model_name,
            "--datasets", args.datasets or "",
            "--worker-model", model_name,
        ]
        if args.max_samples is not None:
            command.extend(["--max-samples", str(args.max_samples)])
        if args.resume:
            command.append("--resume")
        for manifest in args.pair_manifest:
            command.extend(["--pair-manifest", manifest])
        completed = subprocess.run(command, text=True, check=False, env=env)
        if completed.returncode != 0:
            with lock:
                errors.append(f"{model_name} failed with exit code {completed.returncode}")
        tasks.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified public benchmark for reflection-removal models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-ids", default="4,5,6,7")
    parser.add_argument("--output-dir", default="results/public_benchmark/course_local5")
    parser.add_argument("--models", default=None)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--pair-manifest", action="append", default=[])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-model", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    manifests = [Path(path).resolve() for path in args.pair_manifest]
    config = load_config(config_path)
    manifest_datasets = set()
    for manifest in manifests:
        manifest_datasets.update(manifest_pairs(manifest, args.max_samples))
    selected_models, selected_datasets = resolve_selected(
        config, args.models, args.datasets, manifest_datasets
    )
    pair_map = build_pair_map(config, selected_datasets, manifests, args.max_samples)
    output_dir = Path(args.output_dir).resolve()
    args.datasets = ",".join(selected_datasets)

    if args.dry_run:
        print(f"config={config_path}")
        print(f"models={','.join(selected_models)}")
        print(f"datasets={','.join(selected_datasets)}")
        print(f"gpu_ids={args.gpu_ids}")
        print(f"output_dir={output_dir}")
        for name in selected_datasets:
            print(f"pairs[{name}]={len(pair_map[name])}")
        for name in selected_models:
            print(f"weights[{name}]={model_weight_summary(config['models'][name])}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)

    if args.worker_model:
        evaluate_model(args, config, args.worker_model, pair_map)
        return

    gpu_ids = parse_csv(args.gpu_ids)
    if not gpu_ids:
        raise ValueError("At least one GPU ID is required")
    tasks: queue.Queue = queue.Queue()
    for model_name in selected_models:
        tasks.put(model_name)
    errors: list[str] = []
    lock = threading.Lock()
    workers = []
    for gpu_id in gpu_ids:
        thread = threading.Thread(
            target=worker_loop,
            args=(args, config_path, tasks, errors, lock, gpu_id),
            daemon=True,
        )
        thread.start()
        workers.append(thread)
    for _ in workers:
        tasks.put(None)
    tasks.join()
    for thread in workers:
        thread.join()
    if errors:
        raise RuntimeError("; ".join(errors))

    image_rows, dataset_rows, macro_rows = aggregate_results(
        config, output_dir, selected_models, selected_datasets
    )
    write_reports(config, output_dir, image_rows, dataset_rows, macro_rows)
    make_visualizations(config, output_dir, pair_map, selected_models)
    write_manifest(args, config, output_dir, selected_models, selected_datasets, pair_map)
    print(f"PUBLIC BENCHMARK: {output_dir}")


if __name__ == "__main__":
    main()
