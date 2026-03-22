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
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

CIFAR_NUM_CLASSES = 100

try:
    from tta_fl.data import get_clean_mnist_datasets, get_corrupted_mnist_dataset, set_seed
    from tta_fl.model import MNISTCNN
except ImportError:
    get_clean_mnist_datasets = None
    get_corrupted_mnist_dataset = None
    set_seed = None
    MNISTCNN = None


@dataclass
class ASRConfig:
    mode: str
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
    cifar_step5_window_csv: str
    cifar_output_per_window_csv: str
    cifar_output_per_corruption_csv: str
    cifar_output_per_corruption_json: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASR-style concentration trigger on the mixed MNIST window stream.")
    parser.add_argument("--mode", choices=["mnist_stream", "cifar_saved_windows"], default="cifar_saved_windows")
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
    parser.add_argument("--output_csv", default="ARS/outputs/asr_trigger_log.csv")
    parser.add_argument("--output_config", default="ARS/configs/run_config.json")
    parser.add_argument("--output_log", default="ARS/logs/run.log")
    parser.add_argument("--output_summary_csv", default="ARS/outputs/asr_trigger_summary.csv")
    parser.add_argument("--output_threshold_json", default="ARS/outputs/asr_thresholds.json")
    parser.add_argument(
        "--cifar_step5_window_csv",
        default="outputs_step1_step2_cifar_30k_5r_2e_fixed/pools/step5_window_scan.csv",
    )
    parser.add_argument("--cifar_output_per_window_csv", default="ARS/outputs/cifar_asr_per_window.csv")
    parser.add_argument("--cifar_output_per_corruption_csv", default="ARS/outputs/cifar_asr_per_corruption.csv")
    parser.add_argument("--cifar_output_per_corruption_json", default="ARS/outputs/cifar_asr_per_corruption.json")
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
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def load_model(model_ckpt: str | Path, device: torch.device) -> torch.nn.Module:
    if MNISTCNN is None:
        raise ImportError("MNIST dependencies are unavailable. Use --mode cifar_saved_windows.")
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
    if get_clean_mnist_datasets is None or get_corrupted_mnist_dataset is None:
        raise ImportError("MNIST dependencies are unavailable. Use --mode cifar_saved_windows.")
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


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def attach_oracle_labels_to_step5_rows(rows: List[dict]) -> List[dict]:
    mean_accuracy_threshold = float(np.mean([float(row["window_accuracy"]) for row in rows]))
    labeled_rows: List[dict] = []
    for row in rows:
        pool_name = str(row["pool_name"])
        if pool_name in {"visible_failure", "hidden_failure"}:
            oracle_trigger = 1
        elif pool_name in {"stable", "mild_variability"}:
            oracle_trigger = 0
        else:
            oracle_trigger = int(float(row["window_accuracy"]) < mean_accuracy_threshold)
        updated = dict(row)
        updated["oracle_trigger"] = oracle_trigger
        updated["oracle_mean_accuracy_threshold"] = mean_accuracy_threshold
        labeled_rows.append(updated)
    return labeled_rows


def compute_binary_metrics(rows: Sequence[dict]) -> Dict[str, float]:
    y_true = np.asarray([int(row["oracle_trigger"]) for row in rows], dtype=np.int64)
    y_pred = np.asarray([int(row["trigger"]) for row in rows], dtype=np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def load_step5_rows_for_cifar(window_csv_path: Path) -> List[dict]:
    with window_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No Step 5 windows found in {window_csv_path}.")
    return [row for row in attach_oracle_labels_to_step5_rows(rows) if str(row["corruption_type"]) != "clean"]


def load_logits_for_step5_window(row: dict, pools_dir: Path, logits_cache: Dict[str, np.ndarray]) -> np.ndarray:
    corruption_type = str(row["corruption_type"])
    if corruption_type not in logits_cache:
        logits_path = pools_dir / f"scan_{corruption_type}_test_logits.npy"
        if not logits_path.exists():
            raise FileNotFoundError(f"Expected logits for {corruption_type} at {logits_path}")
        logits_cache[corruption_type] = np.load(logits_path, mmap_mode="r")
    start_index = int(float(row["start_index"]))
    end_index = int(float(row["end_index_exclusive"]))
    return np.asarray(logits_cache[corruption_type][start_index:end_index], dtype=np.float64)


def compute_c_t_from_logits(window_logits: np.ndarray) -> float:
    if window_logits.ndim != 2 or window_logits.shape[1] != CIFAR_NUM_CLASSES:
        raise ValueError(f"Expected logits [N,{CIFAR_NUM_CLASSES}], got {window_logits.shape}")
    mean_logits = window_logits.mean(axis=0)
    shifted = mean_logits - np.max(mean_logits)
    probs = np.exp(shifted)
    probs = probs / np.sum(probs)
    return float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))


def run_asr_on_cifar_saved_windows(config: ASRConfig) -> None:
    step5_csv_path = PROJECT_ROOT / config.cifar_step5_window_csv
    per_window_csv = PROJECT_ROOT / config.cifar_output_per_window_csv
    per_corruption_csv = PROJECT_ROOT / config.cifar_output_per_corruption_csv
    per_corruption_json = PROJECT_ROOT / config.cifar_output_per_corruption_json
    thresholds_json = PROJECT_ROOT / config.output_threshold_json
    per_window_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = load_step5_rows_for_cifar(step5_csv_path)
    grouped_rows: Dict[str, List[dict]] = {}
    for row in rows:
        grouped_rows.setdefault(str(row["corruption_type"]), []).append(row)

    pools_dir = step5_csv_path.parent
    logits_cache: Dict[str, np.ndarray] = {}
    num_classes = CIFAR_NUM_CLASSES
    c_bar_init = float(-math.log(config.alpha0 * num_classes))

    per_window_rows: List[dict] = []
    per_corruption_rows: List[dict] = []

    for corruption_type, corruption_rows in sorted(grouped_rows.items()):
        ordered_rows = sorted(corruption_rows, key=lambda row: int(float(row["window_index"])))
        c_bar_prev = c_bar_init
        traced_rows: List[dict] = []
        for row in ordered_rows:
            window_logits = load_logits_for_step5_window(row, pools_dir=pools_dir, logits_cache=logits_cache)
            shifted = window_logits - np.max(window_logits, axis=1, keepdims=True)
            probs = np.exp(shifted)
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            mean_entropy = float(np.mean(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)))
            c_t = compute_c_t_from_logits(window_logits)
            trigger = int(c_t > c_bar_prev)
            if trigger and config.reset_on_trigger:
                c_bar_new = c_bar_init
            else:
                c_bar_new = float(config.mu_c * c_bar_prev + (1.0 - config.mu_c) * c_t)
            traced = {
                "corruption_type": corruption_type,
                "window_index": int(float(row["window_index"])),
                "severity": int(float(row["severity"])),
                "mean_entropy": mean_entropy,
                "window_accuracy": float(row["window_accuracy"]),
                "oracle_trigger": int(row["oracle_trigger"]),
                "pool_name": row["pool_name"],
                "C_t": c_t,
                "C_bar_prev": c_bar_prev,
                "C_bar_new": c_bar_new,
                "trigger": trigger,
                "mu_c": config.mu_c,
                "alpha0": config.alpha0,
                "trigger_threshold": c_bar_prev,
            }
            traced_rows.append(traced)
            per_window_rows.append(traced)
            c_bar_prev = c_bar_new

        metrics = compute_binary_metrics(traced_rows)
        per_corruption_rows.append(
            {
                "corruption_type": corruption_type,
                "mean_accuracy": float(np.mean([float(row["window_accuracy"]) for row in traced_rows])),
                "mean_entropy": float(np.mean([float(row["mean_entropy"]) for row in traced_rows])),
                "mean_C_t": float(np.mean([float(row["C_t"]) for row in traced_rows])),
                "mean_C_bar_prev": float(np.mean([float(row["C_bar_prev"]) for row in traced_rows])),
                "initial_C_bar": c_bar_init,
                "oracle_positive_windows": int(sum(int(row["oracle_trigger"]) for row in traced_rows)),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            }
        )

    with per_window_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_window_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_window_rows)

    with per_corruption_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_corruption_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_corruption_rows)

    per_corruption_json.write_text(
        json.dumps(
            {
                "mode": config.mode,
                "step5_window_csv": str(step5_csv_path),
                "mu_c": config.mu_c,
                "alpha0": config.alpha0,
                "reset_on_trigger": config.reset_on_trigger,
                "window_size": config.window_size,
                "per_corruption_rows": per_corruption_rows,
                "per_window_csv": str(per_window_csv),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    thresholds_json.write_text(
        json.dumps(
            {
                "mu_c": config.mu_c,
                "alpha0": config.alpha0,
                "reset_on_trigger": config.reset_on_trigger,
                "window_size": config.window_size,
                "trigger_logic": "trigger if C_t > C_bar_prev; reset C_bar to -log(alpha0 * C) on trigger",
                "score_for_ars": "C_t compared against evolving C_bar_prev",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    args = parse_args()
    config = ASRConfig(
        mode=args.mode,
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
        cifar_step5_window_csv=args.cifar_step5_window_csv,
        cifar_output_per_window_csv=args.cifar_output_per_window_csv,
        cifar_output_per_corruption_csv=args.cifar_output_per_corruption_csv,
        cifar_output_per_corruption_json=args.cifar_output_per_corruption_json,
    )

    setup_logging(PROJECT_ROOT / config.output_log)
    logging.info("Starting ASR trigger run")
    logging.info("This implementation stops at trigger detection and does not apply TTA")
    save_run_config(config, PROJECT_ROOT / config.output_config)
    if config.mode == "cifar_saved_windows":
        logging.info("Running ARS on saved CIFAR-10 / CIFAR-10-C windows for fair trigger comparison")
        run_asr_on_cifar_saved_windows(config)
        logging.info("Saved CIFAR ARS outputs to %s", PROJECT_ROOT / config.cifar_output_per_corruption_csv)
    else:
        logging.info("Using mixed window stream to match the existing CATTM experiment")
        if set_seed is None:
            raise ImportError("MNIST dependencies are unavailable. Use --mode cifar_saved_windows.")
        set_seed(config.seed)
        device = resolve_device(config.device)
        model = load_model(PROJECT_ROOT / config.model_ckpt, device)
        num_classes = CIFAR_NUM_CLASSES
        c_bar_init = float(-math.log(config.alpha0 * num_classes))
        logging.info("Initial C_bar set to %.6f using -log(alpha0 * C) with alpha0=%.6f and C=%d", c_bar_init, config.alpha0, num_classes)

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
