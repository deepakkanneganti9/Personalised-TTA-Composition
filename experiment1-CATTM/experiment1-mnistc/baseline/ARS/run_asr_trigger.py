from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tta_fl.data import get_clean_mnist_datasets, get_corrupted_mnist_dataset, set_seed
from tta_fl.model import MNISTCNN


@dataclass
class ASRConfig:
    data_dir: str
    model_ckpt: str
    mnistc_root: str
    severity: Sequence[int]
    batch_size: int
    num_workers: int
    seed: int
    device: str
    window_size: int
    mu_c: float
    alpha0: float
    reset_on_trigger: bool
    clean_windows: int
    benign_class_windows: int
    benign_mild_windows: int
    harmful_windows_per_corruption: int
    benign_digits: Sequence[int]
    harmful_corruptions: Sequence[str]
    output_csv: str
    output_config: str
    output_log: str
    output_summary_csv: str
    output_threshold_json: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASR-style concentration trigger on the mixed MNIST window stream.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--model_ckpt", default="artifacts/part1_light/global_model.pt")
    parser.add_argument("--mnistc_root", default="data/mnist_c")
    parser.add_argument("--severity", nargs="+", type=int, default=[3])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--mu_c", type=float, default=0.99)
    parser.add_argument("--alpha0", type=float, default=0.1)
    parser.add_argument("--reset_on_trigger", action="store_true", default=True)
    parser.add_argument("--clean_windows", type=int, default=-1)
    parser.add_argument("--benign_class_windows", type=int, default=-1)
    parser.add_argument("--benign_mild_windows", type=int, default=-1)
    parser.add_argument("--harmful_windows_per_corruption", type=int, default=-1)
    parser.add_argument("--benign_digits", nargs="+", type=int, default=[1, 7])
    parser.add_argument(
        "--harmful_corruptions",
        nargs="+",
        default=["fog", "impulse_noise", "translate", "zigzag"],
    )
    parser.add_argument("--output_csv", default="ASR/outputs/asr_trigger_log.csv")
    parser.add_argument("--output_config", default="ASR/configs/run_config.json")
    parser.add_argument("--output_log", default="ASR/logs/run.log")
    parser.add_argument("--output_summary_csv", default="ASR/outputs/asr_trigger_summary.csv")
    parser.add_argument("--output_threshold_json", default="ASR/outputs/asr_thresholds.json")
    return parser.parse_args()


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_model(model_ckpt: str | Path, device: torch.device) -> torch.nn.Module:
    model = MNISTCNN().to(device)
    state_dict = torch.load(model_ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_subset(dataset: Dataset, size: int, seed: int) -> Subset:
    indices = np.arange(len(dataset))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    subset_size = min(size, len(indices))
    return Subset(dataset, indices[:subset_size].tolist())


def dataset_indices_for_digits(dataset: Dataset, digits: Sequence[int]) -> List[int]:
    targets = dataset.targets.tolist()
    digit_set = set(int(digit) for digit in digits)
    return [index for index, label in enumerate(targets) if int(label) in digit_set]


def build_sequential_windows(dataset: Dataset, num_windows: int, window_size: int) -> List[Subset]:
    available_windows = len(dataset) // window_size
    if num_windows < 0 or num_windows > available_windows:
        num_windows = available_windows
    return [
        Subset(dataset, list(range(window_index * window_size, (window_index + 1) * window_size)))
        for window_index in range(num_windows)
    ]


def build_class_mix_windows(
    dataset: Dataset,
    digits: Sequence[int],
    num_windows: int,
    window_size: int,
    seed: int,
) -> List[Subset]:
    rng = np.random.default_rng(seed)
    digit_indices = dataset_indices_for_digits(dataset, digits)
    rng.shuffle(digit_indices)
    available_windows = len(digit_indices) // window_size
    if num_windows < 0 or num_windows > available_windows:
        num_windows = available_windows
    windows: List[Subset] = []
    for window_index in range(num_windows):
        start = window_index * window_size
        end = start + window_size
        windows.append(Subset(dataset, digit_indices[start:end]))
    return windows


def compute_window_metrics(
    model: torch.nn.Module,
    subset: Subset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    logits_batches: List[torch.Tensor] = []
    entropy_batches: List[torch.Tensor] = []
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            entropy_batches.append(-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1).cpu())
            logits_batches.append(logits.cpu())
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    all_logits = torch.cat(logits_batches, dim=0)
    mean_logits = all_logits.mean(dim=0)
    p_hat = torch.softmax(mean_logits, dim=0)
    c_t = float((p_hat * torch.log(p_hat.clamp_min(1e-12))).sum().item())
    mean_entropy = float(torch.cat(entropy_batches, dim=0).mean().item())
    accuracy = correct / total if total else 0.0
    return mean_entropy, c_t, float(accuracy)


def build_mixed_stream_windows(config: ASRConfig) -> List[Tuple[Subset, str, str, int]]:
    _, clean_test_dataset = get_clean_mnist_datasets(PROJECT_ROOT / config.data_dir)
    stream_windows: List[Tuple[Subset, str, str, int]] = []

    clean_windows = build_sequential_windows(clean_test_dataset, config.clean_windows, config.window_size)
    for window_subset in clean_windows:
        stream_windows.append((window_subset, "clean", "clean_mnist", 0))

    benign_class_windows = build_class_mix_windows(
        clean_test_dataset,
        digits=config.benign_digits,
        num_windows=config.benign_class_windows,
        window_size=config.window_size,
        seed=config.seed,
    )
    for window_subset in benign_class_windows:
        stream_windows.append((window_subset, "benign", "class_mixture_shift", 0))

    mild_dataset = get_corrupted_mnist_dataset(
        data_dir=PROJECT_ROOT / config.data_dir,
        corruption="gaussian_noise",
        severity=1,
        split="test",
        mnist_c_root=None,
    )
    mild_windows = build_sequential_windows(mild_dataset, config.benign_mild_windows, config.window_size)
    for window_subset in mild_windows:
        stream_windows.append((window_subset, "benign", "mild_gaussian_noise", 1))

    severity = config.severity[0]
    for corruption_name in config.harmful_corruptions:
        harmful_dataset = get_corrupted_mnist_dataset(
            data_dir=PROJECT_ROOT / config.data_dir,
            corruption=corruption_name,
            severity=severity,
            split="test",
            mnist_c_root=PROJECT_ROOT / config.mnistc_root,
        )
        harmful_windows = build_sequential_windows(
            harmful_dataset,
            config.harmful_windows_per_corruption,
            config.window_size,
        )
        for window_subset in harmful_windows:
            stream_windows.append((window_subset, "harmful", corruption_name, severity))

    return stream_windows


def open_csv_writer(output_csv: Path) -> Tuple[object, csv.DictWriter]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    handle = output_csv.open("w", newline="")
    fieldnames = [
        "window_index",
        "interval",
        "window_type",
        "source_name",
        "window_sample_size",
        "mean_entropy",
        "accuracy_before_tta",
        "trigger_threshold",
        "trigger_decision",
        "timestamp_utc",
        "C_t",
        "C_bar_prev",
        "C_bar_new",
        "mu_c",
        "trigger",
        "corruption",
        "severity",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    handle.flush()
    return handle, writer


def save_run_config(config: ASRConfig, output_config: Path) -> None:
    output_config.parent.mkdir(parents=True, exist_ok=True)
    with output_config.open("w") as handle:
        json.dump(asdict(config), handle, indent=2)


def save_summary(rows: List[dict], output_summary_csv: Path, output_threshold_json: Path, config: ASRConfig) -> None:
    summary_rows: List[dict] = []
    for window_type in ("clean", "benign", "harmful"):
        matching_rows = [row for row in rows if row["window_type"] == window_type]
        if not matching_rows:
            continue
        summary_rows.append(
            {
                "window_type": window_type,
                "num_windows": len(matching_rows),
                "num_triggers": sum(int(row["trigger"]) for row in matching_rows),
                "mean_C_t": float(np.mean([row["C_t"] for row in matching_rows])),
                "mean_C_bar_prev": float(np.mean([row["C_bar_prev"] for row in matching_rows])),
                "mean_entropy": float(np.mean([row["mean_entropy"] for row in matching_rows])),
                "mean_accuracy_before_tta": float(np.mean([row["accuracy_before_tta"] for row in matching_rows])),
            }
        )

    with output_summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    thresholds_payload = {
        "mu_c": config.mu_c,
        "alpha0": config.alpha0,
        "reset_on_trigger": config.reset_on_trigger,
        "window_size": config.window_size,
        "stream_mode": "mixed_window",
        "trigger_logic": "trigger if C_t > C_bar_prev; reset C_bar to -log(alpha0 * C) on trigger",
        "common_csv_prefix_columns": [
            "window_index",
            "interval",
            "window_type",
            "source_name",
            "window_sample_size",
        ],
    }
    with output_threshold_json.open("w") as handle:
        json.dump(thresholds_payload, handle, indent=2)


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    args = parse_args()
    config = ASRConfig(
        data_dir=args.data_dir,
        model_ckpt=args.model_ckpt,
        mnistc_root=args.mnistc_root,
        severity=tuple(args.severity),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        window_size=args.window_size,
        mu_c=args.mu_c,
        alpha0=args.alpha0,
        reset_on_trigger=args.reset_on_trigger,
        clean_windows=args.clean_windows,
        benign_class_windows=args.benign_class_windows,
        benign_mild_windows=args.benign_mild_windows,
        harmful_windows_per_corruption=args.harmful_windows_per_corruption,
        benign_digits=tuple(args.benign_digits),
        harmful_corruptions=tuple(args.harmful_corruptions),
        output_csv=args.output_csv,
        output_config=args.output_config,
        output_log=args.output_log,
        output_summary_csv=args.output_summary_csv,
        output_threshold_json=args.output_threshold_json,
    )

    setup_logging(PROJECT_ROOT / config.output_log)
    logging.info("Starting ASR trigger run")
    logging.info("This implementation stops at trigger detection and does not apply TTA")
    logging.info("Using mixed window stream to match the existing CATTM experiment")

    set_seed(config.seed)
    device = resolve_device(config.device)
    model = load_model(PROJECT_ROOT / config.model_ckpt, device)
    num_classes = 10
    c_bar_init = float(-math.log(config.alpha0 * num_classes))
    logging.info("Initial C_bar set to %.6f using -log(alpha0 * C) with alpha0=%.6f and C=%d", c_bar_init, config.alpha0, num_classes)

    save_run_config(config, PROJECT_ROOT / config.output_config)
    writer_handle, writer = open_csv_writer(PROJECT_ROOT / config.output_csv)

    stream_windows = build_mixed_stream_windows(config)
    logging.info("Streaming %d windows with window_size=%d", len(stream_windows), config.window_size)
    c_bar_prev = c_bar_init
    rows: List[dict] = []

    try:
        for step_index, (window_subset, window_type, source_name, severity) in enumerate(stream_windows):
            mean_entropy, c_t, accuracy_before_tta = compute_window_metrics(
                model=model,
                subset=window_subset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                device=device,
            )
            trigger = bool(c_t > c_bar_prev)
            if trigger and config.reset_on_trigger:
                c_bar_new = c_bar_init
            else:
                c_bar_new = config.mu_c * c_bar_prev + (1.0 - config.mu_c) * c_t

            start_sample = step_index * config.window_size
            end_sample = start_sample + config.window_size - 1
            row = {
                "window_index": step_index + 1,
                "interval": f"{start_sample}-{end_sample}",
                "window_type": window_type,
                "source_name": source_name,
                "window_sample_size": config.window_size,
                "mean_entropy": mean_entropy,
                "accuracy_before_tta": accuracy_before_tta,
                "trigger_threshold": c_bar_prev,
                "trigger_decision": int(trigger),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "C_t": c_t,
                "C_bar_prev": c_bar_prev,
                "C_bar_new": c_bar_new,
                "mu_c": config.mu_c,
                "trigger": int(trigger),
                "corruption": source_name,
                "severity": severity,
            }
            writer.writerow(row)
            writer_handle.flush()
            rows.append(row)
            c_bar_prev = c_bar_new
    finally:
        writer_handle.close()

    save_summary(
        rows=rows,
        output_summary_csv=PROJECT_ROOT / config.output_summary_csv,
        output_threshold_json=PROJECT_ROOT / config.output_threshold_json,
        config=config,
    )
    logging.info("Saved trigger log to %s", PROJECT_ROOT / config.output_csv)
    logging.info("Finished")


if __name__ == "__main__":
    main()
