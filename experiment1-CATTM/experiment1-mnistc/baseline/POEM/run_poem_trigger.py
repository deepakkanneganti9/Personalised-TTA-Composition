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


ACTION_DELAY_SAMPLES = 100


@dataclass
class PoemConfig:
    data_dir: str
    model_ckpt: str
    mnistc_root: str
    corruption: Sequence[str]
    severity: Sequence[int]
    source_holdout_size: int
    source_holdout_split: str
    test_size: int
    batch_size: int
    num_workers: int
    wealth_threshold: float
    eps_threshold: float
    martingale_clip: float
    martingale_gamma: float
    seed: int
    device: str
    rolling_window: int
    action_delay_samples: int
    action_delay_steps: int
    sanity_clean_stream: bool
    window_size: int
    use_mixed_stream: bool
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
    source_entropy_cache: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="POEM trigger-only drift detection adapted to the existing mixed-window MNIST experiment.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--model_ckpt", default="artifacts/part1_light/global_model.pt")
    parser.add_argument("--mnistc_root", default="data/mnist_c")
    parser.add_argument("--corruption", nargs="+", default=["fog"])
    parser.add_argument("--severity", nargs="+", type=int, default=[3])
    parser.add_argument("--source_holdout_size", type=int, default=10000)
    parser.add_argument("--source_holdout_split", choices=["train", "test"], default="train")
    parser.add_argument("--test_size", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--wealth_threshold", type=float, default=10.0)
    parser.add_argument("--eps_threshold", type=float, default=0.05)
    parser.add_argument("--martingale_clip", type=float, default=1.8)
    parser.add_argument("--martingale_gamma", type=float, default=1.0 / (8.0 * math.sqrt(3.0)))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rolling_window", type=int, default=200)
    parser.add_argument("--action_delay_samples", type=int, default=ACTION_DELAY_SAMPLES)
    parser.add_argument("--sanity_clean_stream", action="store_true")
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--use_mixed_stream", action="store_true", default=True)
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
    parser.add_argument("--output_csv", default="poem/outputs/poem_trigger_log.csv")
    parser.add_argument("--output_config", default="poem/configs/run_config.json")
    parser.add_argument("--output_log", default="poem/logs/run.log")
    parser.add_argument("--output_summary_csv", default="poem/outputs/poem_trigger_summary.csv")
    parser.add_argument("--output_threshold_json", default="poem/outputs/poem_thresholds.json")
    parser.add_argument("--source_entropy_cache", default="poem/outputs/source_holdout_window_entropies.npy")
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


def compute_batch_entropies(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)


def evaluate_window_entropy_and_accuracy(
    model: torch.nn.Module,
    subset: Subset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[float, float]:
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    entropy_values: List[np.ndarray] = []
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            entropy_values.append(compute_batch_entropies(logits).cpu().numpy())
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    window_entropy = float(np.concatenate(entropy_values, axis=0).mean())
    accuracy = correct / total if total else 0.0
    return window_entropy, float(accuracy)


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
    windows: List[Subset] = []
    for window_index in range(num_windows):
        start = window_index * window_size
        end = start + window_size
        windows.append(Subset(dataset, list(range(start, end))))
    return windows


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


class EmpiricalCDF:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.sort(values.astype(np.float64))
        self.size = self.values.size
        self.probabilities = np.linspace(1.0 / self.size, 1.0, self.size, dtype=np.float64)

    def __call__(self, z: float) -> float:
        return float(np.interp(z, self.values, self.probabilities, left=0.0, right=1.0))

    def inverse(self, q: float) -> float:
        q = float(np.clip(q, 0.0, 1.0))
        return float(np.interp(q, self.probabilities, self.values, left=self.values[0], right=self.values[-1]))


def sf_ogd_update(u_t: float, eps_t: float, gradients: List[float], clip_value: float, gamma: float) -> float:
    v_t = u_t - 0.5
    e_tau = clip_value * np.sign(v_t)
    should_clip = e_tau * eps_t > 0.0 and abs(eps_t) > clip_value
    grad_t = 0.0 if should_clip else float(v_t / (1.0 + eps_t * v_t))
    gradients.append(grad_t)
    if grad_t == 0.0:
        return eps_t

    grad_norm = math.sqrt(sum(gradient * gradient for gradient in gradients))
    eps_next = eps_t + gamma * grad_t / max(grad_norm, 1e-12)
    return float(np.clip(eps_next, -2.0, 2.0))


def linear_betting_factor(u_t: float, eps_t: float) -> float:
    bet = 1.0 + eps_t * (u_t - 0.5)
    return float(np.clip(bet, 1e-12, 2.0))


def load_source_holdout_windows(config: PoemConfig) -> List[Subset]:
    train_dataset, test_dataset = get_clean_mnist_datasets(PROJECT_ROOT / config.data_dir)
    source_dataset = train_dataset if config.source_holdout_split == "train" else test_dataset
    source_subset = build_subset(source_dataset, config.source_holdout_size, config.seed)
    return build_sequential_windows(source_subset, -1, config.window_size)


def get_or_create_source_entropy_cache(
    config: PoemConfig,
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    cache_path = PROJECT_ROOT / config.source_entropy_cache
    metadata_path = cache_path.with_suffix(".json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_signature = {
        "model_ckpt": config.model_ckpt,
        "source_holdout_size": config.source_holdout_size,
        "source_holdout_split": config.source_holdout_split,
        "window_size": config.window_size,
        "seed": config.seed,
    }
    if cache_path.exists() and metadata_path.exists():
        with metadata_path.open() as handle:
            cached_signature = json.load(handle)
        if cached_signature == cache_signature:
            logging.info("Loading cached source window entropies from %s", cache_path)
            return np.load(cache_path)

    logging.info("Computing source holdout window entropies from existing global model")
    source_windows = load_source_holdout_windows(config)
    source_entropies = np.asarray(
        [
            evaluate_window_entropy_and_accuracy(
                model=model,
                subset=window_subset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                device=device,
            )[0]
            for window_subset in source_windows
        ],
        dtype=np.float64,
    )
    np.save(cache_path, source_entropies)
    with metadata_path.open("w") as handle:
        json.dump(cache_signature, handle, indent=2)
    return source_entropies


def build_mixed_stream_windows(config: PoemConfig) -> List[Tuple[Subset, str, str, int]]:
    _, clean_test_dataset = get_clean_mnist_datasets(PROJECT_ROOT / config.data_dir)
    stream_windows: List[Tuple[Subset, str, str, int]] = []

    if config.sanity_clean_stream:
        clean_subset = build_subset(clean_test_dataset, config.test_size, config.seed)
        for window_subset in build_sequential_windows(clean_subset, -1, config.window_size):
            stream_windows.append((window_subset, "clean", "clean_mnist", 0))
        return stream_windows

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

    for corruption_name in config.harmful_corruptions:
        harmful_dataset = get_corrupted_mnist_dataset(
            data_dir=PROJECT_ROOT / config.data_dir,
            corruption=corruption_name,
            severity=config.severity[0],
            split="test",
            mnist_c_root=PROJECT_ROOT / config.mnistc_root,
        )
        harmful_windows = build_sequential_windows(
            harmful_dataset,
            config.harmful_windows_per_corruption,
            config.window_size,
        )
        for window_subset in harmful_windows:
            stream_windows.append((window_subset, "harmful", corruption_name, config.severity[0]))

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
        "entropy_z",
        "u",
        "wealth",
        "epsilon",
        "beta_a",
        "beta_b",
        "wealth_trigger",
        "epsilon_trigger",
        "trigger",
        "corruption",
        "severity",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    handle.flush()
    return handle, writer


def save_run_config(config: PoemConfig, output_config: Path) -> None:
    output_config.parent.mkdir(parents=True, exist_ok=True)
    with output_config.open("w") as handle:
        json.dump(asdict(config), handle, indent=2)


def save_summary(rows: List[dict], output_summary_csv: Path, output_threshold_json: Path, config: PoemConfig) -> None:
    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
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
                "mean_wealth": float(np.mean([row["wealth"] for row in matching_rows])),
                "mean_epsilon": float(np.mean([row["epsilon"] for row in matching_rows])),
                "mean_entropy": float(np.mean([row["mean_entropy"] for row in matching_rows])),
                "mean_accuracy_before_tta": float(np.mean([row["accuracy_before_tta"] for row in matching_rows])),
            }
        )

    with output_summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    thresholds_payload = {
        "wealth_threshold": config.wealth_threshold,
        "eps_threshold": config.eps_threshold,
        "eps_threshold_note": "retained for backward compatibility; not used by the corrected trigger logic",
        "martingale_clip": config.martingale_clip,
        "martingale_gamma": config.martingale_gamma,
        "action_delay_samples": config.action_delay_samples,
        "window_size": config.window_size,
        "action_delay_steps": config.action_delay_steps,
        "stream_mode": "mixed_window",
        "trigger_logic": "wealth_only_after_delay",
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


def stream_and_log(
    config: PoemConfig,
    model: torch.nn.Module,
    Fs: EmpiricalCDF,
    device: torch.device,
    writer: csv.DictWriter,
    writer_handle: object,
) -> List[dict]:
    stream_windows = build_mixed_stream_windows(config)
    logging.info("Streaming %d windows with window_size=%d", len(stream_windows), config.window_size)
    wealth = 1.0
    epsilon = 0.0
    gradients: List[float] = []
    rows: List[dict] = []

    for step_index, (window_subset, window_type, source_name, severity) in enumerate(stream_windows):
        mean_entropy, accuracy_before_tta = evaluate_window_entropy_and_accuracy(
            model=model,
            subset=window_subset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            device=device,
        )
        u = Fs(mean_entropy)
        betting_factor = linear_betting_factor(u, epsilon)
        wealth = float(np.clip(wealth * betting_factor, 1e-12, 1e12))
        epsilon_next = sf_ogd_update(
            u_t=u,
            eps_t=epsilon,
            gradients=gradients,
            clip_value=config.martingale_clip,
            gamma=config.martingale_gamma,
        )

        wealth_trigger = wealth >= config.wealth_threshold
        epsilon_trigger = 0
        trigger = bool(wealth_trigger)
        if step_index < config.action_delay_steps:
            trigger = False

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
            "trigger_threshold": config.wealth_threshold,
            "trigger_decision": int(trigger),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "entropy_z": mean_entropy,
            "u": float(u),
            "wealth": wealth,
            "epsilon": epsilon,
            "beta_a": float("nan"),
            "beta_b": float("nan"),
            "wealth_trigger": int(wealth_trigger),
            "epsilon_trigger": int(epsilon_trigger),
            "trigger": int(trigger),
            "corruption": source_name,
            "severity": severity,
        }
        writer.writerow(row)
        writer_handle.flush()
        rows.append(row)
        epsilon = epsilon_next

    return rows


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    args = parse_args()
    action_delay_steps = max(1, math.ceil(args.action_delay_samples / args.window_size))
    config = PoemConfig(
        data_dir=args.data_dir,
        model_ckpt=args.model_ckpt,
        mnistc_root=args.mnistc_root,
        corruption=tuple(args.corruption),
        severity=tuple(args.severity),
        source_holdout_size=args.source_holdout_size,
        source_holdout_split=args.source_holdout_split,
        test_size=args.test_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        wealth_threshold=args.wealth_threshold,
        eps_threshold=args.eps_threshold,
        martingale_clip=args.martingale_clip,
        martingale_gamma=args.martingale_gamma,
        seed=args.seed,
        device=args.device,
        rolling_window=args.rolling_window,
        action_delay_samples=args.action_delay_samples,
        action_delay_steps=action_delay_steps,
        sanity_clean_stream=args.sanity_clean_stream,
        window_size=args.window_size,
        use_mixed_stream=args.use_mixed_stream,
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
        source_entropy_cache=args.source_entropy_cache,
    )

    setup_logging(PROJECT_ROOT / config.output_log)
    logging.info("Starting POEM trigger run")
    logging.info("This implementation stops at trigger detection and does not apply TTA")
    logging.info("Using mixed window stream to match the existing CATTM experiment")
    logging.info("Using POEM martingale b(u)=1+epsilon(u-0.5) with SF-OGD epsilon updates")

    set_seed(config.seed)
    device = resolve_device(config.device)
    model = load_model(PROJECT_ROOT / config.model_ckpt, device)
    source_window_entropies = get_or_create_source_entropy_cache(config, model, device)
    Fs = EmpiricalCDF(source_window_entropies)

    save_run_config(config, PROJECT_ROOT / config.output_config)
    writer_handle, writer = open_csv_writer(PROJECT_ROOT / config.output_csv)
    try:
        rows = stream_and_log(config, model, Fs, device, writer, writer_handle)
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
