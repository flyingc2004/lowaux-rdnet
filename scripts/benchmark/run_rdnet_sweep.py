#!/usr/bin/env python3
import argparse
import csv
import json
import os
import queue
import re
import subprocess
import sys
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path


DEFAULT_DATASETS = ["ceilnet_table2", "real20", "postcard", "objects", "wild"]
METRICS = ["PSNR", "SSIM", "NCC", "LMSE"]
REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DATASETS = {
    "ceilnet_table2": {
        "path": "testdata_CEILNET_table2",
    },
    "real20": {
        "path": "real20",
        "max_long_edge": 512,
    },
    "postcard": {
        "path": "postcard",
    },
    "sir2_postcard_full": {
        "path": "SIR2/PostcardDataset",
    },
    "official_ceilnet_table2": {
        "path": "testdata_CEILNET_table2",
    },
    "official_real20": {
        "path": "real20",
        "max_long_edge": 512,
    },
    "official_sir2_objects": {
        "path": "objects",
    },
    "official_sir2_postcard": {
        "path": "postcard",
    },
    "official_sir2_wild": {
        "path": "wild",
    },
    "objects": {
        "path": "objects",
    },
    "wild": {
        "path": "wild",
    },
    "sir2_withgt": {
        "path": "sir2_withgt",
    },
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_name(path: str) -> str:
    checkpoint_path = Path(path)
    name = "{}__{}".format(checkpoint_path.parent.name, checkpoint_path.stem)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def checkpoint_label(root: Path, checkpoint: str) -> str:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    try:
        return str(checkpoint_path.resolve().relative_to(root))
    except ValueError:
        return str(checkpoint_path.resolve())


def read_existing_metrics(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        return [
            {
                "checkpoint": row["checkpoint"],
                "dataset": row["dataset"],
                **{metric: float(row[metric]) for metric in METRICS},
            }
            for row in csv.DictReader(f)
        ]


def resize_long_edge(img, max_long_edge: int | None):
    if max_long_edge is None:
        return img
    width, height = img.size
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return img
    scale = max_long_edge / float(long_edge)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return img.resize((new_w, new_h))


def tensor_from_pil(img):
    import numpy as np
    import torch

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor_to_rgb_array(tensor):
    tensor = tensor.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0)
    return (tensor.cpu().numpy() * 255.0).round().astype("uint8")


def pad_to_multiple(tensor, multiple: int = 32):
    import torch.nn.functional as F

    _, _, height, width = tensor.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, height, width
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate"), height, width


def normalize_state_key(key: str) -> str:
    prefixes = ["_orig_mod.", "module.", "model.", "net_g."]
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            raw = checkpoint["state_dict"]
            net_state = {k[len("net_g."):]: v for k, v in raw.items() if k.startswith("net_g.")}
            return net_state or raw
        for key in ("params_ema", "params", "net_g", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_rdnet_network(xreflection_root: Path, checkpoint: Path, cls_model: Path, focal_model: Path, device):
    import torch
    import yaml

    sys.path.insert(0, str(xreflection_root))
    from xreflection.archs import build_network

    with (xreflection_root / "options" / "train_rdnet.yml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    network_opt = deepcopy(cfg["network_g"])
    network_opt["pretrained_models"]["cls_model"] = str(cls_model)
    network_opt["pretrained_models"]["base_network"] = str(focal_model)

    net = build_network(network_opt)
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    raw_state = extract_state_dict(ckpt)
    raw_state = {normalize_state_key(k): v for k, v in raw_state.items()}
    model_state = net.state_dict()
    compatible = {
        key: value
        for key, value in raw_state.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    if not compatible:
        raise RuntimeError(f"No compatible RDNet weights found in {checkpoint}")
    loaded_ratio = len(compatible) / max(1, len(model_state))
    if loaded_ratio < 0.8:
        raise RuntimeError(
            "Loaded only {:.1%} of RDNet parameters from {}. "
            "Checkpoint format may be incompatible.".format(loaded_ratio, checkpoint)
        )
    missing, unexpected = net.load_state_dict(compatible, strict=False)
    print(
        "[LOAD] {} compatible={}/{} missing={} unexpected={}".format(
            checkpoint, len(compatible), len(model_state), len(missing), len(unexpected)
        ),
        flush=True,
    )
    return net.to(device).eval()


def run_single(args) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from util.index import quality_assess

    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    xreflection_root = Path(args.xreflection_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    cls_model = Path(args.cls_model).resolve()
    focal_model = Path(args.focal_model).resolve()
    data_root = Path(args.data_root).resolve()

    if args.dataset not in EVAL_DATASETS:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    spec = EVAL_DATASETS[args.dataset]
    dataset_dir = data_root / spec["path"]
    input_dir = dataset_dir / "blended"
    target_dir = dataset_dir / "transmission_layer"
    if not input_dir.is_dir() or not target_dir.is_dir():
        raise FileNotFoundError(f"Expected blended/transmission_layer under {dataset_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    net = load_rdnet_network(xreflection_root, checkpoint, cls_model, focal_model, device)

    checkpoint_name = safe_name(str(checkpoint))
    image_dir = output_dir / "images" / checkpoint_name / args.dataset
    metric_dir = output_dir / "image_metrics" / checkpoint_name
    image_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    filenames = sorted([p.name for p in input_dir.iterdir() if p.is_file()])
    if args.max_samples is not None:
        filenames = filenames[: args.max_samples]
    if not filenames:
        raise RuntimeError(f"No input images found in {input_dir}")

    image_rows = []
    for index, filename in enumerate(filenames, start=1):
        input_img = Image.open(input_dir / filename).convert("RGB")
        target_img = Image.open(target_dir / filename).convert("RGB")
        max_long_edge = spec.get("max_long_edge")
        input_img = resize_long_edge(input_img, max_long_edge)
        target_img = resize_long_edge(target_img, max_long_edge)

        input_tensor = tensor_from_pil(input_img)
        input_tensor, height, width = pad_to_multiple(input_tensor, multiple=32)
        with torch.no_grad():
            _, image_outputs = net(input_tensor.to(device))
            pred = image_outputs[-1][:, :3, :, :][:, :, :height, :width]

        pred_rgb = tensor_to_rgb_array(pred)
        target_rgb = np.asarray(target_img, dtype=np.uint8)
        if pred_rgb.shape[:2] != target_rgb.shape[:2]:
            min_h = min(pred_rgb.shape[0], target_rgb.shape[0])
            min_w = min(pred_rgb.shape[1], target_rgb.shape[1])
            pred_rgb = pred_rgb[:min_h, :min_w]
            target_rgb = target_rgb[:min_h, :min_w]

        metrics = quality_assess(pred_rgb.astype(np.float32), target_rgb.astype(np.float32))
        image_rows.append({"filename": filename, **metrics})
        Image.fromarray(pred_rgb).save(image_dir / f"{Path(filename).stem}.png")
        print(
            "[IMG] {}/{} {} PSNR={:.4f} SSIM={:.4f} NCC={:.4f} LMSE={:.4f}".format(
                index,
                len(filenames),
                filename,
                metrics["PSNR"],
                metrics["SSIM"],
                metrics["NCC"],
                metrics["LMSE"],
            ),
            flush=True,
        )

    image_csv = metric_dir / f"{args.dataset}.csv"
    with image_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename"] + METRICS)
        writer.writeheader()
        writer.writerows(image_rows)

    row = {
        "checkpoint": checkpoint_label(root, str(checkpoint)),
        "dataset": args.dataset,
        **{
            metric: sum(float(image_row[metric]) for image_row in image_rows) / len(image_rows)
            for metric in METRICS
        },
    }
    Path(args.row_json).write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")


def run_one(root: Path, output_dir: Path, checkpoint: str, dataset: str, gpu_id: str, args) -> dict:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint_name = safe_name(str(checkpoint_path))
    log_dir = output_dir / "logs" / checkpoint_name
    row_dir = output_dir / ".rows" / checkpoint_name
    log_dir.mkdir(parents=True, exist_ok=True)
    row_dir.mkdir(parents=True, exist_ok=True)
    row_json = row_dir / f"{dataset}.json"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    pythonpath = [str(args.xreflection_root), str(root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single",
        "--root",
        str(root),
        "--checkpoint",
        str(checkpoint_path),
        "--dataset",
        dataset,
        "--output-dir",
        str(output_dir),
        "--row-json",
        str(row_json),
        "--data-root",
        str(args.data_root),
        "--xreflection-root",
        str(args.xreflection_root),
        "--cls-model",
        str(args.cls_model),
        "--focal-model",
        str(args.focal_model),
    ]
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])

    completed = subprocess.run(command, cwd=str(root), env=env, text=True, capture_output=True, check=False)
    merged = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
    (log_dir / f"{dataset}.log").write_text(merged + "\n", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"{checkpoint_name}/{dataset} failed with code {completed.returncode}:\n{merged[-4000:]}")
    return json.loads(row_json.read_text(encoding="utf-8"))


def worker(root: Path, output_dir: Path, gpu_id: str, tasks: queue.Queue, rows: list[dict], errors: list[str], lock: threading.Lock, args) -> None:
    while True:
        task = tasks.get()
        if task is None:
            tasks.task_done()
            return
        checkpoint, dataset = task
        try:
            print(f"[RUN][GPU {gpu_id}] {checkpoint} :: {dataset}", flush=True)
            row = run_one(root, output_dir, checkpoint, dataset, gpu_id, args)
            with lock:
                rows.append(row)
            print(
                "[OK][GPU {}] {} :: {} -> PSNR={:.4f}, SSIM={:.4f}, NCC={:.4f}, LMSE={:.4f}".format(
                    gpu_id,
                    checkpoint,
                    dataset,
                    row["PSNR"],
                    row["SSIM"],
                    row["NCC"],
                    row["LMSE"],
                ),
                flush=True,
            )
        except Exception as exc:
            with lock:
                errors.append(str(exc))
            print(f"[FAIL][GPU {gpu_id}] {checkpoint} :: {dataset}: {exc}", flush=True)
        finally:
            tasks.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RDNet checkpoints with ERRNet metric definitions.")
    parser.add_argument("checkpoints", nargs="*", help="RDNet Lightning checkpoints to evaluate.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="Comma-separated test datasets.")
    parser.add_argument("--gpu-ids", default="4,5,6,7", help="Comma-separated physical CUDA device IDs.")
    parser.add_argument("--data-root", default="external/datasets/course_local5", type=Path, help="course-local-five dataset root.")
    parser.add_argument("--xreflection-root", default="XReflection", type=Path, help="XReflection repository root.")
    parser.add_argument("--cls-model", default="external/weights/cls_model.pth", type=Path, help="RDNet cls_model.pth.")
    parser.add_argument("--focal-model", default="external/weights/focal.pth", type=Path, help="RDNet focal.pth.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to results/rdnet_sweep/<timestamp>.")
    parser.add_argument("--resume", action="store_true", help="Skip checkpoint/dataset pairs already present in output-dir/metrics.csv.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit images per dataset for smoke tests.")
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--root", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dataset", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--row-json", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else REPO_ROOT
    if not args.data_root.is_absolute():
        args.data_root = (root / args.data_root).resolve()
    if not args.xreflection_root.is_absolute():
        args.xreflection_root = (root / args.xreflection_root).resolve()
    if not args.cls_model.is_absolute():
        args.cls_model = (root / args.cls_model).resolve()
    if not args.focal_model.is_absolute():
        args.focal_model = (root / args.focal_model).resolve()

    if args.single:
        if not args.checkpoint or not args.dataset or not args.row_json:
            raise ValueError("--single requires --checkpoint, --dataset, and --row-json")
        run_single(args)
        return

    datasets = parse_csv(args.datasets)
    gpu_ids = parse_csv(args.gpu_ids)
    if not args.checkpoints:
        raise ValueError("At least one checkpoint is required.")
    if not gpu_ids:
        raise ValueError("At least one GPU ID is required.")
    for dataset in datasets:
        if dataset not in EVAL_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset}")

    if args.output_dir is None:
        output_dir = root / "results" / "rdnet_sweep" / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "metrics.csv"
    rows = read_existing_metrics(csv_path) if args.resume else []
    completed_pairs = {(row["checkpoint"], row["dataset"]) for row in rows}

    tasks = queue.Queue()
    for checkpoint in args.checkpoints:
        label = checkpoint_label(root, checkpoint)
        for dataset in datasets:
            if (label, dataset) in completed_pairs:
                print(f"[SKIP] {checkpoint} :: {dataset}", flush=True)
                continue
            tasks.put((checkpoint, dataset))

    errors: list[str] = []
    lock = threading.Lock()
    workers = []
    for gpu_id in gpu_ids:
        thread = threading.Thread(target=worker, args=(root, output_dir, gpu_id, tasks, rows, errors, lock, args), daemon=True)
        thread.start()
        workers.append(thread)
    for _ in workers:
        tasks.put(None)
    tasks.join()
    for thread in workers:
        thread.join()

    checkpoint_order = {checkpoint_label(root, checkpoint): index for index, checkpoint in enumerate(args.checkpoints)}
    dataset_order = {dataset: index for index, dataset in enumerate(datasets)}
    rows.sort(key=lambda row: (checkpoint_order.get(row["checkpoint"], 999), dataset_order[row["dataset"]]))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "dataset"] + METRICS)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    summary_rows = []
    for checkpoint in dict.fromkeys(row["checkpoint"] for row in rows):
        checkpoint_rows = [row for row in rows if row["checkpoint"] == checkpoint]
        summary_rows.append({
            "checkpoint": checkpoint,
            **{
                f"mean_{metric}": sum(row[metric] for row in checkpoint_rows) / len(checkpoint_rows)
                for metric in METRICS
            },
        })
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint"] + [f"mean_{metric}" for metric in METRICS])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nCSV: {csv_path}")
    print(f"SUMMARY: {summary_path}")
    if errors:
        (output_dir / "errors.log").write_text("\n\n".join(errors) + "\n", encoding="utf-8")
        raise RuntimeError(f"{len(errors)} RDNet evaluation tasks failed. See {output_dir / 'errors.log'}")
    (output_dir / "errors.log").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
