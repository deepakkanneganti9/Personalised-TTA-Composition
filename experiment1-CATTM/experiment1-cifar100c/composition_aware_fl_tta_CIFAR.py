#!/usr/bin/env python3
"""
Composition-aware trigger mechanism for federated-learning TTA on CIFAR-100.

This file currently implements the CIFAR-100 variants of Step 1 and Step 2:
the full mathematical trigger mechanism, including:
  - window statistics
  - multiple historical contexts
  - closest-context matching
  - proxy/input scores
  - separate proxy/input persistence
  - visible and hidden degradation branches
  - final trigger decision

The script is intentionally written as a single-file experiment entrypoint so
later FL training / evaluation steps can be added without changing structure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


CIFAR_NUM_CLASSES = 100
CIFAR_DATASET_NAME = "CIFAR-100"
CIFAR_C_DATASET_NAME = "CIFAR-100-C"
CLEAN_CONTEXT_NAME = "clean_cifar100"
CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD = (0.2675, 0.2565, 0.2761)


# ---------------------------------------------------------------------------
# Step 1 - Composition-aware trigger mechanism
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set seeds for reproducible trigger experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


@dataclass
class TriggerConfig:
    """Configuration for the updated two-branch composition-aware trigger."""

    epsilon: float = 1e-8
    gamma_H: float = 3.0
    gamma_I: float = 2.5
    w_H: float = 0.5
    w_O: float = 0.5
    tau_proxy: float = 0.45
    tau_input: float = 0.45
    K: int = 3
    lambda_hidden: float = 1.0
    trigger_threshold: float = 0.18
    disable_hidden_branch: bool = False
    disable_input_persistence: bool = False
    disable_proxy_persistence: bool = False

    def validate(self) -> None:
        if not math.isclose(self.w_H + self.w_O, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("w_H + w_O must equal 1.")
        if self.K <= 0:
            raise ValueError("K must be positive.")
        if self.gamma_H <= 0 or self.gamma_I <= 0:
            raise ValueError("gamma_H and gamma_I must be positive.")


@dataclass
class HistoricalContext:
    """
    Historical operating context.

    Stores the clean or mildly corrupted operating regime used to match
    incoming windows against the closest known context.
    """

    name: str
    mean_entropy: float
    std_entropy: float
    output_distribution: List[float]
    input_feature_mean: List[float]
    input_feature_std: List[float]
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class WindowStats:
    """Window-level statistics used by the trigger."""

    window_index: int
    avg_entropy: float
    output_distribution: List[float]
    input_feature_mean: List[float]
    num_samples: int
    predicted_labels: List[int]
    max_probabilities: List[float]
    entropies: List[float]


@dataclass
class TriggerResult:
    """All intermediate values for one processed window."""

    window_index: int
    matched_context: str
    avg_entropy: float
    output_distribution: List[float]
    input_feature_mean: List[float]
    delta_input: float
    delta_entropy: float
    delta_entropy_norm: float
    delta_output: float
    proxy_score: float
    input_score: float
    proxy_abnormal: int
    input_abnormal: int
    proxy_persistence: float
    input_persistence: float
    visible_branch_score: float
    hidden_branch_score: float
    final_score: float
    trigger: bool


@dataclass
class FederatedConfig:
    """Step 2 configuration for clean-CIFAR-100 federated training."""

    data_root: str
    rounds: int = 5
    local_epochs: int = 2
    batch_size: int = 256
    eval_batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    num_threads: int = 1
    train_sample_cap: Optional[int] = None
    client_sample_cap: Optional[int] = None
    val_split: int = 5000


@dataclass
class HistoricalBankConfig:
    """Step 3 configuration for clean historical-bank construction."""

    window_size: int = 128


@dataclass
class CorruptionContextConfig:
    """Step 4 configuration for building known mild corruption contexts."""

    mnist_c_root: str
    context_names: List[str] = field(default_factory=lambda: ["brightness", "motion_blur"])
    split: str = "test"
    max_samples_per_context: Optional[int] = 10000


@dataclass
class PoolMiningConfig:
    """Step 5 configuration for scanning corrupted data and mining pools."""

    mnist_c_root: str
    split: str = "test"
    corruption_names: Optional[List[str]] = None
    max_samples_per_corruption: Optional[int] = None
    window_size: int = 128
    severity_bucket_size: int = 10000


@dataclass
class StreamConfig:
    """Step 6 configuration for deployment-like stream construction."""

    random_seed: int = 7
    windows_per_segment: int = 4


@dataclass
class OracleConfig:
    """Step 7 configuration for oracle trigger labels."""

    relative_accuracy_drop_threshold: float = 0.15


@dataclass
class TriggerRunConfig:
    """Step 8 configuration for running the trigger over saved streams."""

    use_saved_pool_window_stats: bool = True


@dataclass
class EvaluationConfig:
    """Step 9 configuration for baseline comparison and plotting."""

    entropy_threshold: float = 0.75
    accuracy_thresholds: List[float] = field(default_factory=lambda: [0.50, 0.60, 0.70, 0.80, 0.90])
    final_score_thresholds: List[float] = field(default_factory=lambda: [0.10, 0.15, 0.20, 0.25, 0.30])
    poem_wealth_threshold: float = 10.0
    poem_martingale_clip: float = 1.8
    poem_martingale_gamma: float = 1.0 / (8.0 * math.sqrt(3.0))
    poem_action_delay_samples: int = 100
    asr_mu_c: float = 0.99
    asr_alpha0: float = 0.1
    asr_reset_on_trigger: bool = True


@dataclass
class ReportingConfig:
    """Step 10 configuration for final report generation."""

    report_filename: str = "step10_report.txt"


class SimpleCIFARCNN(nn.Module):
    """Simple CNN suitable for CIFAR-100 and lightweight FL experiments."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, CIFAR_NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def compute_sample_entropy(probabilities: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    1A. Entropy per sample.

    H_t = -sum_c p_t(c) log(p_t(c) + epsilon)
    """
    return -(probabilities * torch.log(probabilities + epsilon)).sum(dim=1)


def compute_output_distribution(
    predicted_labels: torch.Tensor,
    num_classes: int = CIFAR_NUM_CLASSES,
) -> torch.Tensor:
    """
    1C. Window output distribution from argmax predictions.
    """
    if predicted_labels.numel() == 0:
        raise ValueError("predicted_labels must not be empty.")
    hist = torch.bincount(predicted_labels, minlength=num_classes).float()
    return hist / predicted_labels.numel()


def compute_input_features(images: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """
    1D. Input representation per image:
      - mean intensity
      - std intensity
      - center of mass x
      - center of mass y
      - bounding-box width
      - bounding-box height

    For CIFAR-style RGB inputs, center-of-mass and bounding-box features are
    extracted from a luminance projection so the trigger formulas remain the same.

    Accepts shapes [N, 3, H, W], [N, 1, H, W], or [N, H, W].
    """
    if images.ndim == 4 and images.shape[1] == 3:
        rgb = images.float()
        if float(rgb.min().item()) < 0.0 or float(rgb.max().item()) > 1.0:
            mean = torch.tensor(CIFAR_MEAN, device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
            std = torch.tensor(CIFAR_STD, device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
            rgb = torch.clamp(rgb * std + mean, 0.0, 1.0)
        images = 0.2989 * rgb[:, 0] + 0.5870 * rgb[:, 1] + 0.1140 * rgb[:, 2]
    elif images.ndim == 4 and images.shape[1] == 1:
        images = images[:, 0]
    if images.ndim != 3:
        raise ValueError("images must have shape [N, 3, H, W], [N, 1, H, W], or [N, H, W].")

    images = images.float()
    n, h, w = images.shape
    flat = images.view(n, -1)

    mean_intensity = flat.mean(dim=1)
    std_intensity = flat.std(dim=1, unbiased=False)

    y_coords = torch.arange(h, device=images.device, dtype=images.dtype).view(1, h, 1)
    x_coords = torch.arange(w, device=images.device, dtype=images.dtype).view(1, 1, w)
    mass = images.sum(dim=(1, 2)) + 1e-8
    center_y = (images * y_coords).sum(dim=(1, 2)) / mass
    center_x = (images * x_coords).sum(dim=(1, 2)) / mass

    mask = images > threshold
    any_fg = mask.view(n, -1).any(dim=1)

    rows = mask.any(dim=2)
    cols = mask.any(dim=1)

    row_idx = torch.arange(h, device=images.device).view(1, h)
    col_idx = torch.arange(w, device=images.device).view(1, w)

    row_min = torch.where(rows, row_idx, h).min(dim=1).values
    row_max = torch.where(rows, row_idx, -1).max(dim=1).values
    col_min = torch.where(cols, col_idx, w).min(dim=1).values
    col_max = torch.where(cols, col_idx, -1).max(dim=1).values

    bbox_height = torch.where(any_fg, (row_max - row_min + 1).float(), torch.zeros_like(mean_intensity))
    bbox_width = torch.where(any_fg, (col_max - col_min + 1).float(), torch.zeros_like(mean_intensity))

    return torch.stack(
        [mean_intensity, std_intensity, center_x, center_y, bbox_width, bbox_height],
        dim=1,
    )


def compute_window_stats(
    logits: torch.Tensor,
    images: torch.Tensor,
    window_index: int,
    epsilon: float,
) -> WindowStats:
    """
    1B, 1C, 1D: compute all window-level statistics from logits and images.
    """
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("logits must have shape [N, C] with N > 0.")

    probabilities = F.softmax(logits, dim=1)
    entropies = compute_sample_entropy(probabilities, epsilon)
    predicted_labels = probabilities.argmax(dim=1)
    output_distribution = compute_output_distribution(predicted_labels, num_classes=int(logits.shape[1]))
    input_features = compute_input_features(images)
    max_probabilities = probabilities.max(dim=1).values

    return WindowStats(
        window_index=window_index,
        avg_entropy=float(entropies.mean().item()),
        output_distribution=output_distribution.cpu().tolist(),
        input_feature_mean=input_features.mean(dim=0).cpu().tolist(),
        num_samples=int(logits.shape[0]),
        predicted_labels=predicted_labels.cpu().tolist(),
        max_probabilities=max_probabilities.cpu().tolist(),
        entropies=entropies.cpu().tolist(),
    )


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def compute_input_features_numpy(images: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    Numpy equivalent of compute_input_features, used to avoid heavy CPU-side
    torch ops during large post-processing stages.
    """
    if images.ndim == 4 and images.shape[1] == 3:
        rgb = images.astype(np.float32, copy=False)
        if float(rgb.min()) < 0.0 or float(rgb.max()) > 1.0:
            mean = np.asarray(CIFAR_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
            std = np.asarray(CIFAR_STD, dtype=np.float32).reshape(1, 3, 1, 1)
            rgb = np.clip(rgb * std + mean, 0.0, 1.0)
        images = 0.2989 * rgb[:, 0] + 0.5870 * rgb[:, 1] + 0.1140 * rgb[:, 2]
    elif images.ndim == 4 and images.shape[1] == 1:
        images = images[:, 0]
    elif images.ndim != 3:
        raise ValueError("images must have shape [N, 3, H, W], [N, 1, H, W], or [N, H, W].")

    images = images.astype(np.float32, copy=False)
    n, h, w = images.shape
    flat = images.reshape(n, -1)
    mean_intensity = flat.mean(axis=1)
    std_intensity = flat.std(axis=1)

    y_coords = np.arange(h, dtype=np.float32).reshape(1, h, 1)
    x_coords = np.arange(w, dtype=np.float32).reshape(1, 1, w)
    mass = images.sum(axis=(1, 2)) + 1e-8
    center_y = (images * y_coords).sum(axis=(1, 2)) / mass
    center_x = (images * x_coords).sum(axis=(1, 2)) / mass

    mask = images > threshold
    bbox_height = np.zeros(n, dtype=np.float32)
    bbox_width = np.zeros(n, dtype=np.float32)
    for idx in range(n):
        coords = np.argwhere(mask[idx])
        if coords.size == 0:
            continue
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        bbox_height[idx] = float(y_max - y_min + 1)
        bbox_width[idx] = float(x_max - x_min + 1)

    return np.stack(
        [mean_intensity, std_intensity, center_x, center_y, bbox_width, bbox_height],
        axis=1,
    )


def compute_window_stats_from_numpy(
    logits: np.ndarray,
    images: np.ndarray,
    window_index: int,
    epsilon: float,
) -> WindowStats:
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("logits must have shape [N, C] with N > 0.")

    probabilities = softmax_numpy(logits.astype(np.float64, copy=False))
    entropies = -(probabilities * np.log(probabilities + epsilon)).sum(axis=1)
    predicted_labels = probabilities.argmax(axis=1)
    output_distribution = np.bincount(predicted_labels, minlength=int(logits.shape[1])).astype(np.float64)
    output_distribution /= max(len(predicted_labels), 1)
    input_features = compute_input_features_numpy(images)
    max_probabilities = probabilities.max(axis=1)

    return WindowStats(
        window_index=window_index,
        avg_entropy=float(entropies.mean()),
        output_distribution=output_distribution.tolist(),
        input_feature_mean=input_features.mean(axis=0).tolist(),
        num_samples=int(logits.shape[0]),
        predicted_labels=predicted_labels.tolist(),
        max_probabilities=max_probabilities.tolist(),
        entropies=entropies.tolist(),
    )


def normalized_input_distance(
    input_feature_mean: Sequence[float],
    context: HistoricalContext,
    epsilon: float,
) -> float:
    """
    1F and 1G: normalized Euclidean distance to a historical context.
    """
    current = np.asarray(input_feature_mean, dtype=np.float64)
    hist_mean = np.asarray(context.input_feature_mean, dtype=np.float64)
    hist_std = np.asarray(context.input_feature_std, dtype=np.float64)
    z = (current - hist_mean) / (hist_std + epsilon)
    return float(np.linalg.norm(z))


def match_closest_context(
    window: WindowStats,
    contexts: Sequence[HistoricalContext],
    epsilon: float,
) -> Tuple[HistoricalContext, float]:
    """
    1F. Choose the closest historical context using normalized input distance.
    """
    if not contexts:
        raise ValueError("At least one historical context is required.")

    distances = [
        normalized_input_distance(window.input_feature_mean, context, epsilon)
        for context in contexts
    ]
    best_idx = int(np.argmin(distances))
    return contexts[best_idx], float(distances[best_idx])


class CompositionAwareTrigger:
    """
    Step 1 trigger implementation.

    B_A handles visible degradation:
      persistent proxy abnormality + current proxy score + unfamiliar input.

    B_B handles hidden degradation / confidently wrong cases:
      persistent unfamiliar input + current input shift + calm proxy signal.
    """

    def __init__(self, config: TriggerConfig, historical_contexts: Sequence[HistoricalContext]) -> None:
        self.config = config
        self.config.validate()
        self.historical_contexts = list(historical_contexts)
        if not self.historical_contexts:
            raise ValueError("historical_contexts must not be empty.")

        self.proxy_history: Deque[int] = deque(maxlen=self.config.K)
        self.input_history: Deque[int] = deque(maxlen=self.config.K)

    def reset(self) -> None:
        self.proxy_history.clear()
        self.input_history.clear()

    def process_window(self, window: WindowStats) -> TriggerResult:
        """
        1H-1N: apply the exact updated trigger mechanism to one window.
        """
        cfg = self.config
        context, delta_input = match_closest_context(window, self.historical_contexts, cfg.epsilon)

        delta_entropy = max(
            0.0,
            (window.avg_entropy - context.mean_entropy) / (context.std_entropy + cfg.epsilon),
        )
        delta_entropy_norm = min(1.0, delta_entropy / cfg.gamma_H)

        current_output = np.asarray(window.output_distribution, dtype=np.float64)
        hist_output = np.asarray(context.output_distribution, dtype=np.float64)
        delta_output = 0.5 * float(np.abs(current_output - hist_output).sum())

        proxy_score = cfg.w_H * delta_entropy_norm + cfg.w_O * delta_output
        input_score = min(1.0, delta_input / cfg.gamma_I)

        proxy_abnormal = int(proxy_score > cfg.tau_proxy)
        input_abnormal = int(input_score > cfg.tau_input)

        self.proxy_history.append(proxy_abnormal)
        self.input_history.append(input_abnormal)

        proxy_persistence = float(np.mean(self.proxy_history)) if self.proxy_history else 0.0
        input_persistence = float(np.mean(self.input_history)) if self.input_history else 0.0

        if cfg.disable_proxy_persistence:
            proxy_persistence = 1.0 if proxy_abnormal else 0.0
        if cfg.disable_input_persistence:
            input_persistence = 1.0 if input_abnormal else 0.0

        visible_branch_score = proxy_persistence * proxy_score * input_score

        if cfg.disable_hidden_branch:
            hidden_branch_score = 0.0
        else:
            hidden_branch_score = input_persistence * input_score * (1.0 - proxy_score)

        final_score = max(visible_branch_score, cfg.lambda_hidden * hidden_branch_score)
        trigger = final_score > cfg.trigger_threshold

        return TriggerResult(
            window_index=window.window_index,
            matched_context=context.name,
            avg_entropy=window.avg_entropy,
            output_distribution=window.output_distribution,
            input_feature_mean=window.input_feature_mean,
            delta_input=delta_input,
            delta_entropy=delta_entropy,
            delta_entropy_norm=delta_entropy_norm,
            delta_output=delta_output,
            proxy_score=proxy_score,
            input_score=input_score,
            proxy_abnormal=proxy_abnormal,
            input_abnormal=input_abnormal,
            proxy_persistence=proxy_persistence,
            input_persistence=input_persistence,
            visible_branch_score=visible_branch_score,
            hidden_branch_score=hidden_branch_score,
            final_score=final_score,
            trigger=trigger,
        )


def build_historical_context(
    name: str,
    window_stats: Sequence[WindowStats],
    metadata: Optional[Dict[str, str]] = None,
) -> HistoricalContext:
    """
    Build one historical context profile from a collection of windows.

    This supports Step 1E by storing:
      - mean/std of window entropy
      - average output distribution
      - mean/std of window input feature vectors
    """
    if not window_stats:
        raise ValueError("window_stats must not be empty.")

    entropies = np.asarray([w.avg_entropy for w in window_stats], dtype=np.float64)
    output_distributions = np.asarray([w.output_distribution for w in window_stats], dtype=np.float64)
    input_features = np.asarray([w.input_feature_mean for w in window_stats], dtype=np.float64)

    return HistoricalContext(
        name=name,
        mean_entropy=float(entropies.mean()),
        std_entropy=float(entropies.std(ddof=0) + 1e-8),
        output_distribution=output_distributions.mean(axis=0).tolist(),
        input_feature_mean=input_features.mean(axis=0).tolist(),
        input_feature_std=(input_features.std(axis=0, ddof=0) + 1e-8).tolist(),
        metadata=metadata or {},
    )


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_csv_rows(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_output_dirs(base_dir: Path) -> Dict[str, Path]:
    """Create the output directory structure requested for the experiment."""
    dirs = {
        "root": base_dir,
        "checkpoints": base_dir / "checkpoints",
        "historical_bank": base_dir / "historical_bank",
        "pools": base_dir / "pools",
        "stream_definitions": base_dir / "stream_definitions",
        "window_results": base_dir / "window_results",
        "plots": base_dir / "plots",
        "metrics": base_dir / "metrics",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def make_cifar_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def load_clean_cifar100_datasets(data_root: Path, val_split: int) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Step 2 - load clean CIFAR-100 from the local dataset root.
    """
    transform = make_cifar_transform()
    train_dataset = datasets.CIFAR100(root=str(data_root), train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root=str(data_root), train=False, download=True, transform=transform)

    if val_split <= 0 or val_split >= len(train_dataset):
        raise ValueError("val_split must be in (0, len(train_dataset)).")

    full_indices = np.arange(len(train_dataset))
    train_indices = full_indices[:-val_split]
    val_indices = full_indices[-val_split:]
    return Subset(train_dataset, train_indices.tolist()), Subset(train_dataset, val_indices.tolist()), test_dataset


def subset_targets(dataset: Dataset) -> List[int]:
    """Extract labels from a dataset or subset without changing storage."""
    if isinstance(dataset, Subset):
        base_targets = subset_targets(dataset.dataset)
        return [int(base_targets[i]) for i in dataset.indices]
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("Dataset does not expose targets.")
    if torch.is_tensor(targets):
        return [int(x) for x in targets.tolist()]
    return [int(x) for x in targets]


def limit_dataset_samples(dataset: Dataset, cap: Optional[int], seed: int) -> Dataset:
    """
    Optionally reduce the full training pool before client assignment.

    This is useful when we want the total FL workload to be capped, while
    preserving the same biased client-split mechanism on the reduced pool.
    """
    if cap is None:
        return dataset
    if cap <= 0:
        raise ValueError("train_sample_cap must be positive when provided.")
    if cap >= len(dataset):
        return dataset

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(dataset), size=cap, replace=False)
    return Subset(dataset, sorted(int(x) for x in chosen.tolist()))


def assign_class_biased_clients(targets: Sequence[int], seed: int) -> List[List[int]]:
    """
    Create a mild-to-moderate non-IID split:
      client 1: mostly classes 0,1,2,3
      client 2: mostly classes 3,4,5,6
      client 3: mostly classes 6,7,8,9

    The split is overlapping in class support, but each sample is assigned to one client.
    """
    rng = np.random.default_rng(seed)
    num_classes = max(int(max(targets, default=0)) + 1, CIFAR_NUM_CLASSES)
    chunk = int(math.ceil(num_classes / 3))
    overlap = max(1, chunk // 4)
    dominant_classes = {}
    for client_id in range(3):
        start = max(0, client_id * chunk - (overlap if client_id > 0 else 0))
        end = min(num_classes, (client_id + 1) * chunk + (overlap if client_id < 2 else 0))
        dominant_classes[client_id] = set(range(start, end))

    weights = np.full((3, num_classes), 0.10, dtype=np.float64)
    for client_id, classes in dominant_classes.items():
        for class_id in classes:
            weights[client_id, class_id] = 0.70
    weights /= weights.sum(axis=0, keepdims=True)

    targets_array = np.asarray(targets, dtype=np.int64)
    probs = weights[:, targets_array].T
    cumulative = np.cumsum(probs, axis=1)
    random_values = rng.random(targets_array.shape[0])[:, None]
    assignments = (random_values > cumulative[:, :2]).sum(axis=1)

    client_indices = [np.where(assignments == client_id)[0].tolist() for client_id in range(3)]
    return client_indices


def limit_client_indices(client_indices: List[List[int]], cap: Optional[int], seed: int) -> List[List[int]]:
    if cap is None:
        return client_indices
    rng = np.random.default_rng(seed)
    limited = []
    for indices in client_indices:
        if len(indices) <= cap:
            limited.append(indices)
        else:
            chosen = rng.choice(indices, size=cap, replace=False)
            limited.append(sorted(int(x) for x in chosen.tolist()))
    return limited


def summarize_client_split(targets: Sequence[int], client_indices: Sequence[Sequence[int]]) -> Dict[str, Dict]:
    summary: Dict[str, Dict] = {}
    for client_id, indices in enumerate(client_indices, start=1):
        labels = [int(targets[idx]) for idx in indices]
        label_hist = np.bincount(labels, minlength=CIFAR_NUM_CLASSES).tolist() if labels else [0] * CIFAR_NUM_CLASSES
        summary[f"client_{client_id}"] = {
            "num_samples": len(indices),
            "label_histogram": label_hist,
            "dominant_ratio": float(max(label_hist) / max(len(indices), 1)),
        }
    return summary


def create_client_loaders(
    train_dataset: Dataset,
    client_indices: Sequence[Sequence[int]],
    batch_size: int,
    num_workers: int,
) -> List[DataLoader]:
    loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, list(indices))
        loaders.append(
            DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )
        )
    return loaders


def create_eval_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


class NumpyImageDataset(Dataset):
    """Dataset wrapper for local numpy corruption arrays."""

    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        if images.shape[0] != labels.shape[0]:
            raise ValueError("images and labels must have the same number of samples.")
        self.images = images
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image = self.images[index]
        if image.ndim == 3 and image.shape[-1] in (1, 3):
            image = np.transpose(image, (2, 0, 1))
        elif image.ndim == 2:
            image = image[None, :, :]
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        if image_tensor.ndim == 3 and image_tensor.shape[0] == 3:
            mean = torch.tensor(CIFAR_MEAN, dtype=image_tensor.dtype).view(3, 1, 1)
            std = torch.tensor(CIFAR_STD, dtype=image_tensor.dtype).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
        label = int(self.labels[index])
        return image_tensor, label


def train_one_local_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    client_name: str = "client",
) -> Dict[str, float]:
    """Train one client model locally for a fixed number of epochs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for epoch_idx in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_seen = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_seen += batch_size
            epoch_loss += float(loss.item()) * batch_size
            epoch_correct += int((logits.argmax(dim=1) == labels).sum().item())
            epoch_seen += batch_size

        print(
            f"    {client_name} epoch {epoch_idx + 1}/{epochs}: "
            f"loss={epoch_loss / max(epoch_seen, 1):.4f}, "
            f"acc={epoch_correct / max(epoch_seen, 1):.4f}",
            flush=True,
        )

    return {
        "train_loss": total_loss / max(total_seen, 1),
        "train_accuracy": total_correct / max(total_seen, 1),
        "num_samples_seen": total_seen,
    }


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        total_loss += float(criterion(logits, labels).item())
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_seen += labels.size(0)

    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
        "num_samples": total_seen,
    }


def get_model_state_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def load_model_state(model: nn.Module, state: Dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict(state)
    model.to(device)


def fedavg_states(states: Sequence[Dict[str, torch.Tensor]], weights: Sequence[float]) -> Dict[str, torch.Tensor]:
    if not states:
        raise ValueError("states must not be empty.")
    if len(states) != len(weights):
        raise ValueError("states and weights must have the same length.")

    total_weight = float(sum(weights))
    averaged: Dict[str, torch.Tensor] = {}
    for key in states[0].keys():
        sample_tensor = states[0][key]
        if sample_tensor.is_floating_point():
            accumulator = torch.zeros_like(sample_tensor, dtype=torch.float32)
            for state, weight in zip(states, weights):
                accumulator += state[key].to(torch.float32) * (weight / total_weight)
            averaged[key] = accumulator.to(sample_tensor.dtype)
        else:
            accumulator = torch.zeros_like(sample_tensor, dtype=torch.float64)
            for state, weight in zip(states, weights):
                accumulator += state[key].to(torch.float64) * (weight / total_weight)
            averaged[key] = accumulator.round().to(sample_tensor.dtype)
    return averaged


def save_checkpoint(state: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def run_step2_federated_training(
    output_dirs: Dict[str, Path],
    federated_config: FederatedConfig,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    """
    Step 2 - train the federated model on clean CIFAR-100 using FedAvg.
    """
    torch.set_num_threads(federated_config.num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    print(
        "Step 2: runtime setup "
        f"(device={device}, torch_threads={torch.get_num_threads()}, "
        f"batch_size={federated_config.batch_size}, eval_batch_size={federated_config.eval_batch_size})",
        flush=True,
    )

    train_dataset, val_dataset, test_dataset = load_clean_cifar100_datasets(
        data_root=Path(federated_config.data_root),
        val_split=federated_config.val_split,
    )
    train_dataset = limit_dataset_samples(train_dataset, federated_config.train_sample_cap, seed=seed)
    train_targets = subset_targets(train_dataset)
    client_indices = assign_class_biased_clients(train_targets, seed=seed)
    client_indices = limit_client_indices(client_indices, federated_config.client_sample_cap, seed=seed)

    split_summary = summarize_client_split(train_targets, client_indices)
    save_json(
        {
            "federated_config": asdict(federated_config),
            "effective_total_train_samples": len(train_dataset),
            "split_summary": split_summary,
            "client_indices": {f"client_{i + 1}": indices for i, indices in enumerate(client_indices)},
        },
        output_dirs["metrics"] / "step2_split_metadata.json",
    )

    client_loaders = create_client_loaders(
        train_dataset=train_dataset,
        client_indices=client_indices,
        batch_size=federated_config.batch_size,
        num_workers=federated_config.num_workers,
    )
    val_loader = create_eval_loader(
        val_dataset,
        batch_size=federated_config.eval_batch_size,
        num_workers=federated_config.num_workers,
    )
    test_loader = create_eval_loader(
        test_dataset,
        batch_size=federated_config.eval_batch_size,
        num_workers=federated_config.num_workers,
    )

    global_model = SimpleCIFARCNN().to(device)
    global_state = get_model_state_cpu(global_model)
    round_summaries: List[Dict[str, object]] = []

    print(
        "Step 2: starting federated training "
        f"({federated_config.rounds} rounds, {federated_config.local_epochs} local epochs, device={device})."
    )

    for round_idx in range(1, federated_config.rounds + 1):
        set_seed(seed + round_idx)
        print(f"Step 2: round {round_idx}/{federated_config.rounds}")
        client_states: List[Dict[str, torch.Tensor]] = []
        client_weights: List[float] = []
        client_metrics: Dict[str, Dict[str, float]] = {}

        for client_idx, loader in enumerate(client_loaders, start=1):
            client_name = f"client_{client_idx}"
            print(f"  training {client_name} on {len(loader.dataset)} samples", flush=True)
            client_model = SimpleCIFARCNN().to(device)
            load_model_state(client_model, global_state, device)
            metrics = train_one_local_model(
                model=client_model,
                loader=loader,
                device=device,
                epochs=federated_config.local_epochs,
                lr=federated_config.lr,
                weight_decay=federated_config.weight_decay,
                client_name=client_name,
            )
            client_state = get_model_state_cpu(client_model)
            client_states.append(client_state)
            client_weights.append(float(len(loader.dataset)))
            client_metrics[client_name] = metrics
            print(
                f"  client {client_idx}: "
                f"loss={metrics['train_loss']:.4f}, acc={metrics['train_accuracy']:.4f}, "
                f"samples={int(metrics['num_samples_seen'])}"
            )

            save_checkpoint(
                {
                    "round": round_idx,
                    "client_id": client_idx,
                    "model_state_dict": client_state,
                    "metrics": metrics,
                },
                output_dirs["checkpoints"] / f"client_{client_idx}_round_{round_idx}.pt",
            )

        global_state = fedavg_states(client_states, client_weights)
        load_model_state(global_model, global_state, device)
        val_metrics = evaluate_model(global_model, val_loader, device)
        test_metrics = evaluate_model(global_model, test_loader, device)

        save_checkpoint(
            {
                "round": round_idx,
                "model_state_dict": global_state,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            },
            output_dirs["checkpoints"] / f"global_round_{round_idx}.pt",
        )

        round_summaries.append(
            {
                "round": round_idx,
                "client_metrics": client_metrics,
                "global_val_metrics": val_metrics,
                "global_test_metrics": test_metrics,
            }
        )
        print(
            f"  global: val_acc={val_metrics['accuracy']:.4f}, "
            f"test_acc={test_metrics['accuracy']:.4f}"
        )

    final_summary = {
        "federated_config": asdict(federated_config),
        "split_summary": split_summary,
        "round_summaries": round_summaries,
        "final_global_val_metrics": round_summaries[-1]["global_val_metrics"],
        "final_global_test_metrics": round_summaries[-1]["global_test_metrics"],
    }
    save_json(final_summary, output_dirs["metrics"] / "step2_training_summary.json")
    save_checkpoint(
        {
            "model_state_dict": global_state,
            "summary": final_summary,
        },
        output_dirs["checkpoints"] / "global_model_final.pt",
    )
    return final_summary


def load_global_model_checkpoint(checkpoint_path: Path, device: torch.device) -> SimpleCIFARCNN:
    """Load the global composition model checkpoint saved after Step 2."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Global checkpoint not found at {checkpoint_path}. Run Step 2 first or provide the expected output directory."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = SimpleCIFARCNN().to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model_state"))
    if state_dict is None:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} must contain 'model_state_dict' or 'model_state'."
        )
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def collect_windowed_clean_statistics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    window_size: int,
    epsilon: float,
) -> Tuple[List[WindowStats], List[Dict[str, object]], Dict[str, np.ndarray], Dict[str, float]]:
    """
    Step 3 - evaluate the global model on clean validation windows and retain
    all requested artifacts for the clean historical bank.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    model.eval()
    all_logits: List[torch.Tensor] = []
    all_images: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    total_correct = 0
    total_seen = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)

        total_loss += float(criterion(logits, labels).item())
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_seen += labels.size(0)

        all_logits.append(logits.cpu())
        all_images.append(images.cpu())
        all_labels.append(labels.cpu())

    logits_array = torch.cat(all_logits, dim=0).numpy()
    images_array = torch.cat(all_images, dim=0).numpy()
    labels_array = torch.cat(all_labels, dim=0).numpy()

    probabilities = softmax_numpy(logits_array.astype(np.float64, copy=False))
    predicted_labels = probabilities.argmax(axis=1)
    max_probabilities = probabilities.max(axis=1)
    entropies = -(probabilities * np.log(probabilities + epsilon)).sum(axis=1)
    correctness = (predicted_labels == labels_array).astype(np.int64)
    input_features = compute_input_features_numpy(images_array)

    window_stats_list: List[WindowStats] = []
    window_rows: List[Dict[str, object]] = []

    for start in range(0, total_seen, window_size):
        end = min(start + window_size, total_seen)
        window_index = start // window_size
        window_logits = logits_array[start:end]
        window_images = images_array[start:end]
        window_labels = labels_array[start:end]
        window_preds = predicted_labels[start:end]
        window_probs = max_probabilities[start:end]
        window_entropies = entropies[start:end]
        window_correctness = correctness[start:end]
        window_input_features = input_features[start:end]

        window_stats = compute_window_stats_from_numpy(
            logits=window_logits,
            images=window_images,
            window_index=window_index,
            epsilon=epsilon,
        )
        window_stats_list.append(window_stats)

        window_accuracy = float(np.mean(window_correctness))
        row = {
            "window_index": window_index,
            "start_index": start,
            "end_index_exclusive": end,
            "num_samples": int(end - start),
            "window_accuracy": window_accuracy,
            "window_entropy": window_stats.avg_entropy,
            "matched_context_placeholder": CLEAN_CONTEXT_NAME,
        }
        for cls_idx, value in enumerate(window_stats.output_distribution):
            row[f"output_dist_{cls_idx}"] = float(value)
        for feat_idx, value in enumerate(window_stats.input_feature_mean):
            row[f"input_feat_{feat_idx}"] = float(value)
        row["predicted_labels_json"] = json.dumps(window_preds.tolist())
        row["true_labels_json"] = json.dumps(window_labels.tolist())
        row["max_probabilities_json"] = json.dumps([float(x) for x in window_probs.tolist()])
        row["correctness_flags_json"] = json.dumps([int(x) for x in window_correctness.tolist()])
        row["entropies_json"] = json.dumps([float(x) for x in window_entropies.tolist()])
        window_rows.append(row)

    arrays = {
        "logits": logits_array,
        "labels": labels_array,
        "predicted_labels": predicted_labels,
        "entropies": entropies,
        "max_probabilities": max_probabilities,
        "correctness_flags": correctness,
        "input_features": input_features,
    }
    metrics = {
        "clean_val_accuracy": total_correct / max(total_seen, 1),
        "clean_val_loss": total_loss / max(total_seen, 1),
        "num_validation_samples": total_seen,
        "num_windows": len(window_stats_list),
        "window_size": window_size,
    }
    return window_stats_list, window_rows, arrays, metrics


def save_window_arrays(arrays: Dict[str, np.ndarray], output_prefix: Path) -> Dict[str, str]:
    saved_paths: Dict[str, str] = {}
    for name, array in arrays.items():
        path = output_prefix.parent / f"{output_prefix.name}_{name}.npy"
        np.save(path, array)
        saved_paths[name] = str(path)
    return saved_paths


def resolve_cifar10_c_paths(cifar10_c_root: Path, corruption_name: str) -> Tuple[Path, Path]:
    images_path = cifar10_c_root / f"{corruption_name}.npy"
    labels_path = cifar10_c_root / "labels.npy"
    if not images_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Missing {CIFAR_C_DATASET_NAME} files for corruption='{corruption_name}' under {cifar10_c_root}."
        )
    return images_path, labels_path


def load_cifar10_c_dataset(
    cifar10_c_root: Path,
    corruption_name: str,
    split: str,
    max_samples: Optional[int] = None,
    severities: Optional[Sequence[int]] = None,
) -> Tuple[Dataset, Dict[str, object]]:
    """
    Load one local CIFAR-100-C corruption dataset using official severity blocks.
    """
    if split != "test":
        raise ValueError(f"Local {CIFAR_C_DATASET_NAME} only provides test corruption files, so split must be 'test'.")

    images_path, labels_path = resolve_cifar10_c_paths(cifar10_c_root, corruption_name)
    images = np.load(images_path)
    labels = np.load(labels_path)

    selected_severities = list(severities) if severities is not None else [1, 2, 3, 4, 5]
    severity_block = 10000
    image_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    index_chunks: List[np.ndarray] = []
    for severity in selected_severities:
        if severity < 1 or severity > 5:
            raise ValueError(f"{CIFAR_C_DATASET_NAME} severities must be in {{1,2,3,4,5}}.")
        start = (severity - 1) * severity_block
        end = severity * severity_block
        image_chunks.append(images[start:end])
        label_chunks.append(labels[start:end])
        index_chunks.append(np.arange(start, end, dtype=np.int64))

    images = np.concatenate(image_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    original_indices = np.concatenate(index_chunks, axis=0)

    if max_samples is not None and max_samples < len(labels):
        images = images[:max_samples]
        labels = labels[:max_samples]
        original_indices = original_indices[:max_samples]

    dataset = NumpyImageDataset(images=images, labels=labels)
    metadata = {
        "corruption_name": corruption_name,
        "split": split,
        "num_samples": int(labels.shape[0]),
        "images_path": str(images_path),
        "labels_path": str(labels_path),
        "selected_severities": selected_severities,
        "original_indices_start": int(original_indices[0]) if len(original_indices) else 0,
        "original_indices_end_exclusive": int(original_indices[-1] + 1) if len(original_indices) else 0,
        "severity_note": f"{CIFAR_C_DATASET_NAME} uses official 10,000-sample blocks for severities 1..5.",
    }
    return dataset, metadata


def list_available_cifar10_c_corruptions(cifar10_c_root: Path) -> List[str]:
    return sorted(
        path.stem
        for path in cifar10_c_root.glob("*.npy")
        if path.stem != "labels"
    )


def load_historical_context_bank(history_dir: Path) -> List[HistoricalContext]:
    bank_path = history_dir / "historical_context_bank.json"
    if not bank_path.exists():
        clean_path = history_dir / f"context_1_{CLEAN_CONTEXT_NAME}.json"
        if not clean_path.exists():
            raise FileNotFoundError(
                f"No historical bank found at {bank_path} or {clean_path}. Run Step 3 and Step 4 first."
            )
        with clean_path.open("r", encoding="utf-8") as handle:
            return [HistoricalContext(**json.load(handle))]

    with bank_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [HistoricalContext(**context) for context in payload["historical_contexts"]]


def infer_local_severity_bucket(sample_index: int, bucket_size: int) -> int:
    """
    Infer the CIFAR-100-C severity bucket from the 10,000-sample block index.
    """
    return int(sample_index // bucket_size) + 1


def run_step3_clean_historical_bank(
    output_dirs: Dict[str, Path],
    federated_config: FederatedConfig,
    historical_bank_config: HistoricalBankConfig,
    trigger_config: TriggerConfig,
    device: torch.device,
) -> Dict[str, object]:
    """
    Step 3 - build the clean historical bank and save clean validation windows.
    """
    print("Step 3: building clean historical bank from validation windows.")
    _, val_dataset, _ = load_clean_cifar100_datasets(
        data_root=Path(federated_config.data_root),
        val_split=federated_config.val_split,
    )
    val_loader = create_eval_loader(
        dataset=val_dataset,
        batch_size=federated_config.eval_batch_size,
        num_workers=federated_config.num_workers,
    )
    global_model = load_global_model_checkpoint(
        output_dirs["checkpoints"] / "global_model_final.pt",
        device=device,
    )

    window_stats_list, window_rows, arrays, clean_metrics = collect_windowed_clean_statistics(
        model=global_model,
        loader=val_loader,
        device=device,
        window_size=historical_bank_config.window_size,
        epsilon=trigger_config.epsilon,
    )
    clean_context = build_historical_context(
        name=CLEAN_CONTEXT_NAME,
        window_stats=window_stats_list,
        metadata={
            "type": "clean",
            "source_split": "validation",
            "window_size": str(historical_bank_config.window_size),
        },
    )

    history_dir = output_dirs["historical_bank"]
    save_json(asdict(clean_context), history_dir / f"context_1_{CLEAN_CONTEXT_NAME}.json")
    save_csv_rows(window_rows, history_dir / "clean_validation_window_summary.csv")
    array_paths = save_window_arrays(arrays, history_dir / "clean_validation")

    summary = {
        "clean_context_profile_path": str(history_dir / f"context_1_{CLEAN_CONTEXT_NAME}.json"),
        "window_summary_csv_path": str(history_dir / "clean_validation_window_summary.csv"),
        "array_paths": array_paths,
        "clean_validation_metrics": clean_metrics,
        "num_saved_windows": len(window_rows),
        "coverage_checklist": {
            "window_accuracy": True,
            "window_entropy": True,
            "window_output_distribution": True,
            "window_input_feature_vectors": True,
            "predicted_labels": True,
            "max_probabilities": True,
            "correctness_flags": True,
            f"historical_context_1_{CLEAN_CONTEXT_NAME}": True,
            "global_clean_validation_performance": True,
        },
    }
    save_json(summary, history_dir / "step3_clean_historical_bank_summary.json")
    return summary


def run_step4_known_corruption_contexts(
    output_dirs: Dict[str, Path],
    federated_config: FederatedConfig,
    historical_bank_config: HistoricalBankConfig,
    corruption_context_config: CorruptionContextConfig,
    trigger_config: TriggerConfig,
    device: torch.device,
) -> Dict[str, object]:
    """
    Step 4 - build additional historical contexts from mild known corruptions.
    """
    print(
        f"Step 4: building known corruption contexts from local {CIFAR_C_DATASET_NAME} "
        f"split='{corruption_context_config.split}'."
    )
    mnist_c_root = Path(corruption_context_config.mnist_c_root)
    global_model = load_global_model_checkpoint(
        output_dirs["checkpoints"] / "global_model_final.pt",
        device=device,
    )

    history_dir = output_dirs["historical_bank"]
    context_summaries: List[Dict[str, object]] = []
    historical_bank_index: List[Dict[str, object]] = []

    clean_context_path = history_dir / f"context_1_{CLEAN_CONTEXT_NAME}.json"
    if clean_context_path.exists():
        with clean_context_path.open("r", encoding="utf-8") as handle:
            historical_bank_index.append(json.load(handle))

    for idx, corruption_name in enumerate(corruption_context_config.context_names, start=2):
        dataset, dataset_meta = load_cifar10_c_dataset(
            cifar10_c_root=mnist_c_root,
            corruption_name=corruption_name,
            split=corruption_context_config.split,
            max_samples=corruption_context_config.max_samples_per_context,
            severities=[1],
        )
        loader = create_eval_loader(
            dataset=dataset,
            batch_size=federated_config.eval_batch_size,
            num_workers=federated_config.num_workers,
        )
        window_stats_list, window_rows, arrays, metrics = collect_windowed_clean_statistics(
            model=global_model,
            loader=loader,
            device=device,
            window_size=historical_bank_config.window_size,
            epsilon=trigger_config.epsilon,
        )
        context = build_historical_context(
            name=f"known_{corruption_name}",
            window_stats=window_stats_list,
            metadata={
                "type": "known_mild_corruption",
                "corruption_name": corruption_name,
                "source_split": corruption_context_config.split,
                "window_size": str(historical_bank_config.window_size),
                "severity_note": dataset_meta["severity_note"],
            },
        )

        context_json_path = history_dir / f"context_{idx}_{corruption_name}.json"
        window_csv_path = history_dir / f"{corruption_name}_{corruption_context_config.split}_window_summary.csv"
        save_json(asdict(context), context_json_path)
        save_csv_rows(window_rows, window_csv_path)
        array_paths = save_window_arrays(
            arrays,
            history_dir / f"{corruption_name}_{corruption_context_config.split}",
        )

        summary = {
            "context_name": context.name,
            "context_json_path": str(context_json_path),
            "window_summary_csv_path": str(window_csv_path),
            "array_paths": array_paths,
            "evaluation_metrics": metrics,
            "dataset_metadata": dataset_meta,
        }
        context_summaries.append(summary)
        historical_bank_index.append(asdict(context))

    combined_index = {
        "historical_contexts": historical_bank_index,
        "step4_context_summaries": context_summaries,
        "selection_note": (
            f"Step 4 used local per-corruption {CIFAR_C_DATASET_NAME} files and the official severity-1 block "
            "for each selected known corruption context."
        ),
    }
    save_json(combined_index, history_dir / "historical_context_bank.json")
    save_json(
        {
            "cifar10_c_root": str(mnist_c_root),
            "selected_contexts": corruption_context_config.context_names,
            "split": corruption_context_config.split,
            "max_samples_per_context": corruption_context_config.max_samples_per_context,
            "context_summaries": context_summaries,
        },
        history_dir / "step4_known_corruption_contexts_summary.json",
    )
    return combined_index


@torch.no_grad()
def scan_corruption_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    corruption_name: str,
    split: str,
    historical_contexts: Sequence[HistoricalContext],
    trigger_config: TriggerConfig,
    window_size: int,
    severity_bucket_size: int,
    sample_index_offset: int = 0,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, np.ndarray]]:
    """
    Step 5 - scan one corruption dataset, record per-sample statistics, and
    aggregate windows for later pool mining.
    """
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_images: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_images.append(images.cpu())
        all_labels.append(labels.cpu())

    logits_array = torch.cat(all_logits, dim=0).numpy()
    images_array = torch.cat(all_images, dim=0).numpy()
    labels_array = torch.cat(all_labels, dim=0).numpy()

    probabilities = softmax_numpy(logits_array.astype(np.float64, copy=False))
    predicted_labels = probabilities.argmax(axis=1)
    max_probabilities = probabilities.max(axis=1)
    entropies = -(probabilities * np.log(probabilities + trigger_config.epsilon)).sum(axis=1)
    input_features = compute_input_features_numpy(images_array)
    correctness = (predicted_labels == labels_array).astype(np.int64)

    sample_rows: List[Dict[str, object]] = []
    window_rows: List[Dict[str, object]] = []

    for local_idx in range(labels_array.shape[0]):
        feature_vec = input_features[local_idx].tolist()
        matched_context, distance = min(
            (
                (context, normalized_input_distance(feature_vec, context, trigger_config.epsilon))
                for context in historical_contexts
            ),
            key=lambda item: item[1],
        )
        global_sample_index = sample_index_offset + local_idx
        sample_rows.append(
            {
                "sample_id": f"{corruption_name}_{split}_{global_sample_index}",
                "corruption_type": corruption_name,
                "split": split,
                "sample_index": global_sample_index,
                "severity": infer_local_severity_bucket(global_sample_index, severity_bucket_size),
                "severity_is_inferred": 1,
                "true_label": int(labels_array[local_idx]),
                "predicted_label": int(predicted_labels[local_idx]),
                "is_correct": int(correctness[local_idx]),
                "entropy": float(entropies[local_idx]),
                "max_softmax_probability": float(max_probabilities[local_idx]),
                "matched_context": matched_context.name,
                "distance_from_closest_historical_context": float(distance),
                "input_features_json": json.dumps([float(x) for x in feature_vec]),
            }
        )

    for start in range(0, labels_array.shape[0], window_size):
        end = min(start + window_size, labels_array.shape[0])
        window_index = start // window_size
        window_logits = logits_array[start:end]
        window_images = images_array[start:end]
        window_labels = labels_array[start:end]
        window_preds = predicted_labels[start:end]
        window_correctness = correctness[start:end]
        window_entropies = entropies[start:end]
        window_probs = max_probabilities[start:end]
        window_input = input_features[start:end]

        window_stats = compute_window_stats_from_numpy(
            logits=window_logits,
            images=window_images,
            window_index=window_index,
            epsilon=trigger_config.epsilon,
        )
        matched_context, delta_input = match_closest_context(
            window_stats,
            historical_contexts,
            trigger_config.epsilon,
        )
        hist_output = np.asarray(matched_context.output_distribution, dtype=np.float64)
        current_output = np.asarray(window_stats.output_distribution, dtype=np.float64)
        output_deviation = 0.5 * float(np.abs(current_output - hist_output).sum())

        window_rows.append(
            {
                "window_id": f"{corruption_name}_{split}_window_{window_index}",
                "corruption_type": corruption_name,
                "split": split,
                "window_index": window_index,
                "start_index": sample_index_offset + start,
                "end_index_exclusive": sample_index_offset + end,
                "severity": infer_local_severity_bucket(sample_index_offset + start, severity_bucket_size),
                "severity_is_inferred": 1,
                "window_accuracy": float(np.mean(window_correctness)),
                "window_entropy": float(np.mean(window_entropies)),
                "window_max_probability": float(np.mean(window_probs)),
                "window_wrong_fraction": 1.0 - float(np.mean(window_correctness)),
                "matched_context": matched_context.name,
                "distance_from_closest_historical_context": float(delta_input),
                "output_distribution_deviation": float(output_deviation),
                "predicted_labels_json": json.dumps(window_preds.tolist()),
                "true_labels_json": json.dumps(window_labels.tolist()),
                "correctness_flags_json": json.dumps([int(x) for x in window_correctness.tolist()]),
                "entropies_json": json.dumps([float(x) for x in window_entropies.tolist()]),
                "max_probabilities_json": json.dumps([float(x) for x in window_probs.tolist()]),
                "input_feature_mean_json": json.dumps([float(x) for x in window_input.mean(axis=0).tolist()]),
            }
        )
        for cls_idx, value in enumerate(window_stats.output_distribution):
            window_rows[-1][f"output_dist_{cls_idx}"] = float(value)

    arrays = {
        "logits": logits_array,
        "labels": labels_array,
        "predicted_labels": predicted_labels,
        "entropies": entropies,
        "max_probabilities": max_probabilities,
        "correctness_flags": correctness,
        "input_features": input_features,
    }
    return sample_rows, window_rows, arrays


def assign_sample_pools(sample_rows: List[Dict[str, object]], thresholds: Dict[str, float]) -> List[Dict[str, object]]:
    for row in sample_rows:
        is_correct = row["is_correct"] == 1
        entropy = float(row["entropy"])
        max_prob = float(row["max_softmax_probability"])
        input_dist = float(row["distance_from_closest_historical_context"])

        if is_correct and entropy <= thresholds["low_entropy"] and input_dist <= thresholds["low_input_distance"]:
            pool_name = "stable"
        elif (
            is_correct
            and entropy <= thresholds["mild_entropy"]
            and input_dist <= thresholds["moderate_input_distance"]
        ):
            pool_name = "mild_variability"
        elif (
            not is_correct
            and entropy >= thresholds["high_entropy"]
            and input_dist >= thresholds["high_input_distance"]
        ):
            pool_name = "visible_failure"
        elif (
            not is_correct
            and entropy <= thresholds["hidden_entropy"]
            and max_prob >= thresholds["high_confidence"]
            and input_dist >= thresholds["high_input_distance"]
        ):
            pool_name = "hidden_failure"
        else:
            pool_name = "unassigned"
        row["pool_name"] = pool_name
    return sample_rows


def assign_window_pools(window_rows: List[Dict[str, object]], thresholds: Dict[str, float]) -> List[Dict[str, object]]:
    for row in window_rows:
        acc = float(row["window_accuracy"])
        entropy = float(row["window_entropy"])
        max_prob = float(row["window_max_probability"])
        input_dist = float(row["distance_from_closest_historical_context"])
        out_dev = float(row["output_distribution_deviation"])

        if acc >= thresholds["stable_accuracy"] and entropy <= thresholds["low_entropy"] and input_dist <= thresholds["low_input_distance"]:
            pool_name = "stable"
        elif (
            acc >= thresholds["mild_accuracy"]
            and entropy <= thresholds["mild_entropy"]
            and input_dist <= thresholds["moderate_input_distance"]
        ):
            pool_name = "mild_variability"
        elif (
            acc <= thresholds["failure_accuracy"]
            and (entropy >= thresholds["high_entropy"] or out_dev >= thresholds["high_output_deviation"])
            and input_dist >= thresholds["high_input_distance"]
        ):
            pool_name = "visible_failure"
        elif (
            acc <= thresholds["failure_accuracy"]
            and entropy <= thresholds["hidden_entropy"]
            and max_prob >= thresholds["high_confidence"]
            and input_dist >= thresholds["high_input_distance"]
        ):
            pool_name = "hidden_failure"
        else:
            pool_name = "unassigned"
        row["pool_name"] = pool_name
    return window_rows


def top_k_by_score(
    rows: Sequence[Dict[str, object]],
    score_fn,
    k: int,
    excluded_ids: Optional[set] = None,
    id_key: str = "window_id",
) -> List[Dict[str, object]]:
    excluded_ids = excluded_ids or set()
    candidates = [row for row in rows if row[id_key] not in excluded_ids]
    ranked = sorted(candidates, key=score_fn, reverse=True)
    return [dict(row) for row in ranked[:k]]


def ensure_required_window_pools(
    window_rows: List[Dict[str, object]],
    min_windows_per_pool: int = 8,
) -> List[Dict[str, object]]:
    """
    Step 5 fallback miner.

    If strict thresholding leaves a required pool empty or too small, refill it
    using ranked candidates based on the intended semantics of each pool.
    """
    assigned_by_id = {row["window_id"]: row["pool_name"] for row in window_rows if row["pool_name"] != "unassigned"}

    def acc(row): return float(row["window_accuracy"])
    def ent(row): return float(row["window_entropy"])
    def conf(row): return float(row["window_max_probability"])
    def dist(row): return float(row["distance_from_closest_historical_context"])
    def out(row): return float(row["output_distribution_deviation"])
    conf_values = [conf(r) for r in window_rows]
    ent_values = [ent(r) for r in window_rows]
    dist_values = [dist(r) for r in window_rows]
    conf_q70 = np.quantile(conf_values, 0.70) if conf_values else 0.0
    ent_q45 = np.quantile(ent_values, 0.45) if ent_values else 0.0
    dist_q50 = np.quantile(dist_values, 0.50) if dist_values else 0.0
    dist_q75 = np.quantile(dist_values, 0.75) if dist_values else 0.0

    pool_specs = {
        "stable": lambda row: (
            acc(row) >= 0.92
            and dist(row) <= dist_q50
        ),
        "mild_variability": lambda row: (
            acc(row) >= 0.85
            and dist(row) <= dist_q75
        ),
        "visible_failure": lambda row: (
            acc(row) <= 0.80
            and dist(row) >= dist_q50
        ),
        "hidden_failure": lambda row: (
            acc(row) <= 0.65
            and conf(row) >= conf_q70
            and ent(row) <= ent_q45
            and dist(row) >= dist_q50
        ),
    }
    score_fns = {
        "stable": lambda row: (acc(row), -ent(row), -dist(row)),
        "mild_variability": lambda row: (acc(row), -abs(dist(row) - 0.5 * dist_q75), -ent(row)),
        "visible_failure": lambda row: (-acc(row), ent(row), out(row), dist(row)),
        "hidden_failure": lambda row: (-acc(row), conf(row), -ent(row), dist(row)),
    }

    for pool_name in ["stable", "mild_variability", "visible_failure", "hidden_failure"]:
        current = [row for row in window_rows if row["pool_name"] == pool_name]
        if len(current) >= min_windows_per_pool:
            continue
        needed = min_windows_per_pool - len(current)
        candidates = [row for row in window_rows if pool_specs[pool_name](row)]
        selected = top_k_by_score(
            rows=candidates,
            score_fn=score_fns[pool_name],
            k=needed,
            excluded_ids=set(assigned_by_id.keys()),
        )
        for row in selected:
            assigned_by_id[row["window_id"]] = pool_name

    for row in window_rows:
        if row["window_id"] in assigned_by_id:
            row["pool_name"] = assigned_by_id[row["window_id"]]
    return window_rows


def build_pool_rows_from_historical_summary(
    summary_csv_path: Path,
    pool_name: str,
    corruption_type: str,
    split: str,
    matched_context: str,
) -> List[Dict[str, object]]:
    rows = load_csv_rows(summary_csv_path)
    built: List[Dict[str, object]] = []
    for row in rows:
        converted = dict(row)
        window_index = int(float(converted["window_index"]))
        max_probs = [float(x) for x in parse_json_list(converted["max_probabilities_json"])]
        input_feature_mean = [float(converted[f"input_feat_{idx}"]) for idx in range(6)]
        built_row = {
            "window_id": f"{corruption_type}_{split}_window_{window_index}",
            "corruption_type": corruption_type,
            "split": split,
            "window_index": window_index,
            "start_index": int(float(converted["start_index"])),
            "end_index_exclusive": int(float(converted["end_index_exclusive"])),
            "severity": 0,
            "severity_is_inferred": 0,
            "window_accuracy": float(converted["window_accuracy"]),
            "window_entropy": float(converted["window_entropy"]),
            "window_max_probability": float(np.mean(max_probs)) if max_probs else 0.0,
            "window_wrong_fraction": 1.0 - float(converted["window_accuracy"]),
            "matched_context": matched_context,
            "distance_from_closest_historical_context": 0.0,
            "output_distribution_deviation": 0.0,
            "predicted_labels_json": converted["predicted_labels_json"],
            "true_labels_json": converted["true_labels_json"],
            "correctness_flags_json": converted["correctness_flags_json"],
            "entropies_json": converted["entropies_json"],
            "max_probabilities_json": converted["max_probabilities_json"],
            "input_feature_mean_json": json.dumps(input_feature_mean),
            "pool_name": pool_name,
        }
        for cls_idx in range(CIFAR_NUM_CLASSES):
            built_row[f"output_dist_{cls_idx}"] = float(converted[f"output_dist_{cls_idx}"])
        built.append(built_row)
    return built


def summarize_pool_assignments(rows: List[Dict[str, object]], id_key: str) -> Dict[str, object]:
    summary: Dict[str, object] = {"counts": {}}
    for pool_name in ["stable", "mild_variability", "visible_failure", "hidden_failure", "unassigned"]:
        pool_rows = [row for row in rows if row["pool_name"] == pool_name]
        summary["counts"][pool_name] = len(pool_rows)
        summary[f"{pool_name}_{id_key}s"] = [row[id_key] for row in pool_rows]
    return summary


def load_csv_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def maybe_cast_float(value: object) -> object:
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def normalize_window_row_types(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    numeric_keys = {
        "window_index",
        "start_index",
        "end_index_exclusive",
        "severity",
        "severity_is_inferred",
        "window_accuracy",
        "window_entropy",
        "window_max_probability",
        "window_wrong_fraction",
        "distance_from_closest_historical_context",
        "output_distribution_deviation",
    }
    normalized: List[Dict[str, object]] = []
    for row in rows:
        converted = dict(row)
        for key in numeric_keys:
            if key in converted:
                converted[key] = maybe_cast_float(converted[key])
        normalized.append(converted)
    return normalized


def sample_windows_for_region(
    pool_rows: Sequence[Dict[str, object]],
    region_label: str,
    count: int,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    if not pool_rows:
        raise ValueError(f"No windows available for region '{region_label}'.")
    chosen_indices = rng.choice(len(pool_rows), size=count, replace=len(pool_rows) < count)
    selected: List[Dict[str, object]] = []
    for region_order, row_idx in enumerate(np.atleast_1d(chosen_indices).tolist()):
        row = dict(pool_rows[int(row_idx)])
        row["region_label"] = region_label
        row["region_order"] = region_order
        selected.append(row)
    return selected


def build_stream_definition(
    name: str,
    segment_plan: Sequence[Tuple[str, str]],
    windows_by_pool: Dict[str, List[Dict[str, object]]],
    windows_per_segment: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """
    Build one deployment-like stream by concatenating windows from mined pools.

    segment_plan items are (region_label, pool_name).
    """
    stream_windows: List[Dict[str, object]] = []
    segments: List[Dict[str, object]] = []

    for segment_idx, (region_label, pool_name) in enumerate(segment_plan):
        sampled = sample_windows_for_region(
            pool_rows=windows_by_pool[pool_name],
            region_label=region_label,
            count=windows_per_segment,
            rng=rng,
        )
        start_idx = len(stream_windows)
        for row in sampled:
            stream_window = dict(row)
            stream_window["stream_name"] = name
            stream_window["pool_name"] = pool_name
            stream_window["stream_window_index"] = len(stream_windows)
            stream_window["segment_index"] = segment_idx
            stream_windows.append(stream_window)
        segments.append(
            {
                "segment_index": segment_idx,
                "region_label": region_label,
                "pool_name": pool_name,
                "start_window_index": start_idx,
                "end_window_index_exclusive": len(stream_windows),
                "num_windows": windows_per_segment,
            }
        )

    return {
        "stream_name": name,
        "num_windows": len(stream_windows),
        "segments": segments,
        "windows": stream_windows,
    }


def run_step6_create_streams(
    output_dirs: Dict[str, Path],
    stream_config: StreamConfig,
) -> Dict[str, object]:
    """
    Step 6 - create streaming test scenarios from mined pools.
    """
    print("Step 6: creating deployment-like streaming scenarios from mined pools.")
    window_rows = normalize_window_row_types(load_csv_rows(output_dirs["pools"] / "step5_window_scan.csv"))
    windows_by_pool: Dict[str, List[Dict[str, object]]] = {
        "stable": [row for row in window_rows if row["pool_name"] == "stable"],
        "mild_variability": [row for row in window_rows if row["pool_name"] == "mild_variability"],
        "visible_failure": [row for row in window_rows if row["pool_name"] == "visible_failure"],
        "hidden_failure": [row for row in window_rows if row["pool_name"] == "hidden_failure"],
    }

    for pool_name, rows in windows_by_pool.items():
        if not rows:
            raise ValueError(
                f"Step 6 requires non-empty pool '{pool_name}'. "
                "If this pool is empty, Step 5 thresholds or corruption coverage need adjustment."
            )

    rng = np.random.default_rng(stream_config.random_seed)
    stream_specs = {
        "stream_a_benign_variability": [
            ("stable", "stable"),
            ("mild_variability", "mild_variability"),
            ("stable_recovery", "stable"),
        ],
        "stream_b_visible_degradation": [
            ("stable", "stable"),
            ("visible_failure", "visible_failure"),
            ("recovery", "stable"),
        ],
        "stream_c_hidden_degradation": [
            ("stable", "stable"),
            ("hidden_failure", "hidden_failure"),
            ("recovery", "stable"),
        ],
        "stream_d_mixed_long": [
            ("stable", "stable"),
            ("mild_variability", "mild_variability"),
            ("visible_failure", "visible_failure"),
            ("hidden_failure", "hidden_failure"),
            ("recovery", "stable"),
        ],
    }

    stream_summaries: Dict[str, object] = {}
    for stream_name, segment_plan in stream_specs.items():
        definition = build_stream_definition(
            name=stream_name,
            segment_plan=segment_plan,
            windows_by_pool=windows_by_pool,
            windows_per_segment=stream_config.windows_per_segment,
            rng=rng,
        )
        rows_for_csv = []
        for window in definition["windows"]:
            row = dict(window)
            rows_for_csv.append(row)

        save_json(
            definition,
            output_dirs["stream_definitions"] / f"{stream_name}.json",
        )
        save_csv_rows(
            rows_for_csv,
            output_dirs["stream_definitions"] / f"{stream_name}.csv",
        )
        stream_summaries[stream_name] = {
            "num_windows": definition["num_windows"],
            "segments": definition["segments"],
            "json_path": str(output_dirs["stream_definitions"] / f"{stream_name}.json"),
            "csv_path": str(output_dirs["stream_definitions"] / f"{stream_name}.csv"),
        }

    summary = {
        "stream_config": asdict(stream_config),
        "stream_summaries": stream_summaries,
        "window_size_note": (
            "Streams are constructed from Step 5 window pools, so the effective stream window size "
            "matches the configured experiment window size used during pool mining."
        ),
    }
    save_json(summary, output_dirs["stream_definitions"] / "step6_stream_summary.json")
    return summary


def should_trigger_oracle_from_region(region_label: str) -> int:
    """
    Step 7 main oracle:
      Stable and Mild Variability -> No Trigger
      Visible Failure and Hidden Failure -> Trigger
    """
    normalized = region_label.lower()
    if "visible_failure" in normalized or "hidden_failure" in normalized:
        return 1
    return 0


def parse_json_list(value: object) -> List[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return json.loads(value)
    raise ValueError(f"Unsupported JSON-list value type: {type(value)}")


def load_stream_windows(stream_json_path: Path) -> List[Dict[str, object]]:
    with stream_json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["windows"]


def stream_window_to_window_stats(window: Dict[str, object]) -> WindowStats:
    output_distribution = [
        float(window.get(f"output_dist_{cls_idx}", 0.0))
        for cls_idx in range(CIFAR_NUM_CLASSES)
    ]
    input_feature_mean = parse_json_list(window["input_feature_mean_json"])
    predicted_labels = [int(x) for x in parse_json_list(window["predicted_labels_json"])]
    max_probabilities = [float(x) for x in parse_json_list(window["max_probabilities_json"])]
    entropies = [float(x) for x in parse_json_list(window["entropies_json"])]
    return WindowStats(
        window_index=int(float(window["stream_window_index"])),
        avg_entropy=float(window["window_entropy"]),
        output_distribution=output_distribution,
        input_feature_mean=[float(x) for x in input_feature_mean],
        num_samples=len(predicted_labels),
        predicted_labels=predicted_labels,
        max_probabilities=max_probabilities,
        entropies=entropies,
    )


def run_step7_define_oracle_labels(
    output_dirs: Dict[str, Path],
    oracle_config: OracleConfig,
) -> Dict[str, object]:
    """
    Step 7 - define oracle trigger labels for each stream window.
    """
    print("Step 7: defining oracle trigger labels for all streams.")
    stream_summary_path = output_dirs["stream_definitions"] / "step6_stream_summary.json"
    if not stream_summary_path.exists():
        raise FileNotFoundError(
            f"Step 7 requires {stream_summary_path}. Run Step 6 first."
        )

    clean_summary_path = output_dirs["historical_bank"] / "step3_clean_historical_bank_summary.json"
    if not clean_summary_path.exists():
        raise FileNotFoundError(
            f"Step 7 requires {clean_summary_path}. Run Step 3 first."
        )
    with clean_summary_path.open("r", encoding="utf-8") as handle:
        clean_summary = json.load(handle)
    clean_baseline_accuracy = float(clean_summary["clean_validation_metrics"]["clean_val_accuracy"])
    accuracy_floor = clean_baseline_accuracy * (1.0 - oracle_config.relative_accuracy_drop_threshold)

    with stream_summary_path.open("r", encoding="utf-8") as handle:
        stream_summary = json.load(handle)

    stream_oracle_summaries: Dict[str, object] = {}
    for stream_name, meta in stream_summary["stream_summaries"].items():
        stream_json_path = Path(meta["json_path"])
        with stream_json_path.open("r", encoding="utf-8") as handle:
            stream_definition = json.load(handle)

        oracle_rows: List[Dict[str, object]] = []
        for window in stream_definition["windows"]:
            region_label = str(window["region_label"])
            window_accuracy = float(window["window_accuracy"])
            scenario_oracle = should_trigger_oracle_from_region(region_label)
            accuracy_oracle = int(window_accuracy < accuracy_floor)

            oracle_row = dict(window)
            oracle_row["oracle_trigger"] = scenario_oracle
            oracle_row["oracle_trigger_accuracy_aux"] = accuracy_oracle
            oracle_row["oracle_reason"] = (
                "scenario_region"
                if scenario_oracle == 1
                else "stable_or_mild_region"
            )
            oracle_row["clean_baseline_accuracy"] = clean_baseline_accuracy
            oracle_row["accuracy_drop_threshold"] = oracle_config.relative_accuracy_drop_threshold
            oracle_row["accuracy_floor"] = accuracy_floor
            oracle_rows.append(oracle_row)

        oracle_json_path = output_dirs["window_results"] / f"{stream_name}_oracle_labels.json"
        oracle_csv_path = output_dirs["window_results"] / f"{stream_name}_oracle_labels.csv"
        save_json(
            {
                "stream_name": stream_name,
                "oracle_config": asdict(oracle_config),
                "clean_baseline_accuracy": clean_baseline_accuracy,
                "accuracy_floor": accuracy_floor,
                "windows": oracle_rows,
            },
            oracle_json_path,
        )
        save_csv_rows(oracle_rows, oracle_csv_path)

        stream_oracle_summaries[stream_name] = {
            "num_windows": len(oracle_rows),
            "num_oracle_trigger": int(sum(int(row["oracle_trigger"]) for row in oracle_rows)),
            "num_accuracy_aux_trigger": int(sum(int(row["oracle_trigger_accuracy_aux"]) for row in oracle_rows)),
            "json_path": str(oracle_json_path),
            "csv_path": str(oracle_csv_path),
        }

    summary = {
        "oracle_config": asdict(oracle_config),
        "clean_baseline_accuracy": clean_baseline_accuracy,
        "accuracy_floor": accuracy_floor,
        "stream_oracle_summaries": stream_oracle_summaries,
        "main_oracle_note": (
            "Main oracle labels are scenario-based: Stable/Mild -> 0, Visible/Hidden -> 1. "
            "Accuracy-based oracle is saved as an auxiliary diagnostic only."
        ),
    }
    save_json(summary, output_dirs["window_results"] / "step7_oracle_summary.json")
    return summary


def run_step8_execute_trigger(
    output_dirs: Dict[str, Path],
    trigger_config: TriggerConfig,
    trigger_run_config: TriggerRunConfig,
) -> Dict[str, object]:
    """
    Step 8 - run the updated composition-aware trigger on all streams.
    """
    print("Step 8: running the composition-aware trigger on all streams.")
    historical_contexts = load_historical_context_bank(output_dirs["historical_bank"])
    stream_summary_path = output_dirs["stream_definitions"] / "step6_stream_summary.json"
    oracle_summary_path = output_dirs["window_results"] / "step7_oracle_summary.json"
    if not stream_summary_path.exists():
        raise FileNotFoundError(f"Step 8 requires {stream_summary_path}. Run Step 6 first.")
    if not oracle_summary_path.exists():
        raise FileNotFoundError(f"Step 8 requires {oracle_summary_path}. Run Step 7 first.")

    with stream_summary_path.open("r", encoding="utf-8") as handle:
        stream_summary = json.load(handle)

    step8_stream_summaries: Dict[str, object] = {}
    for stream_name, meta in stream_summary["stream_summaries"].items():
        stream_windows = load_stream_windows(Path(meta["json_path"]))
        oracle_path = output_dirs["window_results"] / f"{stream_name}_oracle_labels.json"
        with oracle_path.open("r", encoding="utf-8") as handle:
            oracle_payload = json.load(handle)
        oracle_by_index = {
            int(window["stream_window_index"]): window
            for window in oracle_payload["windows"]
        }

        trigger = CompositionAwareTrigger(
            config=trigger_config,
            historical_contexts=historical_contexts,
        )
        result_rows: List[Dict[str, object]] = []
        visible_branch_hits = 0
        hidden_branch_hits = 0

        for window in stream_windows:
            stats = stream_window_to_window_stats(window)
            result = asdict(trigger.process_window(stats))
            oracle_row = oracle_by_index[int(window["stream_window_index"])]

            merged = dict(window)
            merged.update(result)
            merged["oracle_trigger"] = int(oracle_row["oracle_trigger"])
            merged["oracle_trigger_accuracy_aux"] = int(oracle_row["oracle_trigger_accuracy_aux"])
            merged["oracle_reason"] = oracle_row["oracle_reason"]
            merged["branch_dominant"] = (
                "visible"
                if float(result["visible_branch_score"]) >= float(trigger_config.lambda_hidden * result["hidden_branch_score"])
                else "hidden"
            )
            if float(result["visible_branch_score"]) > 0:
                visible_branch_hits += 1
            if float(result["hidden_branch_score"]) > 0:
                hidden_branch_hits += 1
            result_rows.append(merged)

        json_path = output_dirs["window_results"] / f"{stream_name}_trigger_results.json"
        csv_path = output_dirs["window_results"] / f"{stream_name}_trigger_results.csv"
        save_json(
            {
                "stream_name": stream_name,
                "trigger_config": asdict(trigger_config),
                "trigger_run_config": asdict(trigger_run_config),
                "results": result_rows,
            },
            json_path,
        )
        save_csv_rows(result_rows, csv_path)

        step8_stream_summaries[stream_name] = {
            "num_windows": len(result_rows),
            "num_triggered": int(sum(int(bool(row["trigger"])) for row in result_rows)),
            "visible_branch_nonzero_windows": visible_branch_hits,
            "hidden_branch_nonzero_windows": hidden_branch_hits,
            "json_path": str(json_path),
            "csv_path": str(csv_path),
        }

    summary = {
        "trigger_config": asdict(trigger_config),
        "trigger_run_config": asdict(trigger_run_config),
        "step8_stream_summaries": step8_stream_summaries,
        "demonstration_note": (
            "Visible degradation is exposed through visible_branch_score, "
            "while hidden/confidently-wrong degradation is exposed through hidden_branch_score."
        ),
    }
    save_json(summary, output_dirs["window_results"] / "step8_trigger_summary.json")
    return summary


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def step5_window_row_to_window_stats(row: Dict[str, object]) -> WindowStats:
    """Convert one saved Step 5 window row back into the trigger's WindowStats structure."""
    output_distribution = [float(row.get(f"output_dist_{cls_idx}", 0.0)) for cls_idx in range(CIFAR_NUM_CLASSES)]
    input_feature_mean = [float(x) for x in parse_json_list(row["input_feature_mean_json"])]
    predicted_labels = [int(x) for x in parse_json_list(row["predicted_labels_json"])]
    max_probabilities = [float(x) for x in parse_json_list(row["max_probabilities_json"])]
    entropies = [float(x) for x in parse_json_list(row["entropies_json"])]
    return WindowStats(
        window_index=int(float(row["window_index"])),
        avg_entropy=float(row["window_entropy"]),
        output_distribution=output_distribution,
        input_feature_mean=input_feature_mean,
        num_samples=len(predicted_labels),
        predicted_labels=predicted_labels,
        max_probabilities=max_probabilities,
        entropies=entropies,
    )


def run_trigger_on_full_step5_windows(
    output_dirs: Dict[str, Path],
    trigger_config: TriggerConfig,
) -> List[Dict[str, object]]:
    """
    Run the exact composition-aware trigger on the full Step 5 saved windows.

    This keeps all methods on the same corruption windows for threshold calibration
    and corruption-wise comparisons.
    """
    historical_contexts = load_historical_context_bank(output_dirs["historical_bank"])
    step5_rows = normalize_window_row_types(load_csv_rows(output_dirs["pools"] / "step5_window_scan.csv"))
    grouped_rows: Dict[str, List[Dict[str, object]]] = {}
    for row in step5_rows:
        grouped_rows.setdefault(str(row["corruption_type"]), []).append(row)

    traced_rows: List[Dict[str, object]] = []
    for corruption_type in sorted(grouped_rows.keys()):
        trigger = CompositionAwareTrigger(
            config=trigger_config,
            historical_contexts=historical_contexts,
        )
        corruption_rows = sorted(
            grouped_rows[corruption_type],
            key=lambda row: int(float(row["window_index"])),
        )
        for row in corruption_rows:
            stats = step5_window_row_to_window_stats(row)
            result = asdict(trigger.process_window(stats))
            merged = dict(row)
            merged.update(result)
            merged["avg_entropy"] = float(row["window_entropy"])
            merged["proxy_statistic"] = float(result["proxy_persistence"]) * float(result["proxy_score"])
            merged["input_statistic"] = float(result["input_persistence"]) * float(result["input_score"])
            traced_rows.append(merged)
    return traced_rows


def attach_full_window_oracle_labels(
    rows: Sequence[Dict[str, object]],
    mean_accuracy_threshold: float,
) -> List[Dict[str, object]]:
    """
    Build a full-window evaluation oracle for Step 5.

    Where Step 5 already mined explicit benign/failure pools, we preserve those semantics.
    For ambiguous unassigned windows, we use the global mean accuracy threshold so that
    all methods are compared on the full saved window set.
    """
    labeled_rows: List[Dict[str, object]] = []
    for row in rows:
        pool_name = str(row["pool_name"])
        if pool_name in {"visible_failure", "hidden_failure"}:
            oracle_trigger = 1
            oracle_reason = "step5_failure_pool"
        elif pool_name in {"stable", "mild_variability"}:
            oracle_trigger = 0
            oracle_reason = "step5_benign_pool"
        else:
            oracle_trigger = int(float(row["window_accuracy"]) < mean_accuracy_threshold)
            oracle_reason = "mean_accuracy_threshold_for_unassigned"

        updated = dict(row)
        updated["oracle_trigger"] = oracle_trigger
        updated["oracle_reason"] = oracle_reason
        updated["mean_accuracy_threshold"] = mean_accuracy_threshold
        labeled_rows.append(updated)
    return labeled_rows


def make_threshold_grid(values: np.ndarray, num_points: int = 15) -> List[float]:
    """Compact threshold grid derived from the empirical score distribution."""
    if values.size == 0:
        return [0.0]
    quantiles = np.linspace(0.10, 0.90, num_points)
    grid = sorted({float(np.quantile(values, q)) for q in quantiles})
    if not grid:
        return [float(values.mean())]
    return grid


def evaluate_single_threshold_method(
    rows: Sequence[Dict[str, object]],
    score_key: str,
    threshold: float,
    *,
    greater_is_positive: bool,
    decision_key: str,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    decisions: List[int] = []
    for row in rows:
        score = float(row[score_key])
        if greater_is_positive:
            decisions.append(int(score > threshold))
        else:
            decisions.append(int(score < threshold))
    enriched = attach_decisions(rows, decision_key, decisions)
    metrics = compute_binary_metrics(enriched, decision_key)
    metrics["accuracy"] = safe_divide(
        metrics["tp"] + metrics["tn"],
        metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"],
    )
    return enriched, metrics


def sweep_best_threshold(
    rows: Sequence[Dict[str, object]],
    score_key: str,
    *,
    greater_is_positive: bool,
    metric_name: str = "f1",
) -> Tuple[float, Dict[str, float], List[Dict[str, object]]]:
    values = np.asarray([float(row[score_key]) for row in rows], dtype=np.float64)
    best_threshold = float(values.mean()) if values.size else 0.0
    best_metrics: Optional[Dict[str, float]] = None
    best_rows: List[Dict[str, object]] = []
    best_score = -1.0

    for threshold in make_threshold_grid(values):
        candidate_rows, candidate_metrics = evaluate_single_threshold_method(
            rows=rows,
            score_key=score_key,
            threshold=threshold,
            greater_is_positive=greater_is_positive,
            decision_key=f"{score_key}_decision",
        )
        if float(candidate_metrics[metric_name]) > best_score:
            best_score = float(candidate_metrics[metric_name])
            best_threshold = threshold
            best_metrics = candidate_metrics
            best_rows = candidate_rows

    if best_metrics is None:
        best_rows, best_metrics = evaluate_single_threshold_method(
            rows=rows,
            score_key=score_key,
            threshold=best_threshold,
            greater_is_positive=greater_is_positive,
            decision_key=f"{score_key}_decision",
        )
    return best_threshold, best_metrics, best_rows


def compute_corruption_wise_metrics(
    rows: Sequence[Dict[str, object]],
    decision_key: str,
    method_name: str,
    threshold_value: float,
    threshold_source: str,
) -> List[Dict[str, object]]:
    metric_rows: List[Dict[str, object]] = []
    corruption_types = sorted({str(row["corruption_type"]) for row in rows})
    for corruption_type in corruption_types:
        subset = [row for row in rows if str(row["corruption_type"]) == corruption_type]
        metrics = compute_binary_metrics(subset, decision_key)
        metric_rows.append(
            {
                "corruption_type": corruption_type,
                "method": method_name,
                "threshold_source": threshold_source,
                "threshold_value": threshold_value,
                "num_windows": len(subset),
                "mean_window_accuracy": float(np.mean([float(row["window_accuracy"]) for row in subset])),
                "mean_final_score": float(np.mean([float(row["final_score"]) for row in subset])),
                "accuracy": safe_divide(
                    metrics["tp"] + metrics["tn"],
                    metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"],
                ),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "false_positive_rate": metrics["false_positive_rate"],
                "false_negative_rate": metrics["false_negative_rate"],
                "hidden_failure_recall": metrics["hidden_failure_recall"],
                "visible_failure_recall": metrics["visible_failure_recall"],
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            }
        )
    return metric_rows


def compute_detection_delay(rows: Sequence[Dict[str, object]], decision_key: str) -> Optional[float]:
    """Detection delay from first oracle-positive window to first predicted-positive window within that region."""
    oracle_positive_indices = [i for i, row in enumerate(rows) if int(row["oracle_trigger"]) == 1]
    if not oracle_positive_indices:
        return None
    first_oracle = oracle_positive_indices[0]
    predicted_after = [i for i in oracle_positive_indices if int(bool(rows[i][decision_key])) == 1]
    if not predicted_after:
        return None
    return float(predicted_after[0] - first_oracle)


def compute_binary_metrics(rows: Sequence[Dict[str, object]], decision_key: str) -> Dict[str, float]:
    y_true = np.asarray([int(row["oracle_trigger"]) for row in rows], dtype=np.int64)
    y_pred = np.asarray([int(bool(row[decision_key])) for row in rows], dtype=np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    visible_mask = np.asarray(
        [
            1
            if "visible_failure" in str(row.get("region_label", row.get("pool_name", ""))).lower()
            else 0
            for row in rows
        ],
        dtype=np.int64,
    )
    hidden_mask = np.asarray(
        [
            1
            if "hidden_failure" in str(row.get("region_label", row.get("pool_name", ""))).lower()
            else 0
            for row in rows
        ],
        dtype=np.int64,
    )

    visible_recall = safe_divide(
        float(np.sum((visible_mask == 1) & (y_pred == 1))),
        float(np.sum(visible_mask == 1)),
    )
    hidden_recall = safe_divide(
        float(np.sum((hidden_mask == 1) & (y_pred == 1))),
        float(np.sum(hidden_mask == 1)),
    )

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    fpr = safe_divide(fp, fp + tn)
    fnr = safe_divide(fn, fn + tp)
    delay = compute_detection_delay(rows, decision_key)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "detection_delay": -1.0 if delay is None else delay,
        "hidden_failure_recall": hidden_recall,
        "visible_failure_recall": visible_recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def baseline_entropy_only(rows: Sequence[Dict[str, object]], entropy_threshold: float) -> List[int]:
    return [int(float(row["avg_entropy"]) > entropy_threshold) for row in rows]


def baseline_proxy_only(rows: Sequence[Dict[str, object]], trigger_threshold: float) -> List[int]:
    return [
        int(float(row["proxy_persistence"]) * float(row["proxy_score"]) > trigger_threshold)
        for row in rows
    ]


def baseline_input_only(rows: Sequence[Dict[str, object]], trigger_threshold: float) -> List[int]:
    return [
        int(float(row["input_persistence"]) * float(row["input_score"]) > trigger_threshold)
        for row in rows
    ]


def baseline_accuracy_and_final_score(
    rows: Sequence[Dict[str, object]],
    accuracy_threshold: float,
    final_score_threshold: float,
) -> List[int]:
    return [
        int(
            (float(row["window_accuracy"]) < accuracy_threshold)
            or (float(row["final_score"]) > final_score_threshold)
        )
        for row in rows
    ]


class EmpiricalCDF:
    """POEM source-side entropy calibrator, preserved from the reference logic."""

    def __init__(self, values: np.ndarray) -> None:
        if values.size == 0:
            raise ValueError("EmpiricalCDF requires at least one value.")
        self.values = np.sort(values.astype(np.float64))
        self.size = self.values.size
        self.probabilities = np.linspace(1.0 / self.size, 1.0, self.size, dtype=np.float64)

    def __call__(self, z: float) -> float:
        return float(np.interp(z, self.values, self.probabilities, left=0.0, right=1.0))


def sf_ogd_update(u_t: float, eps_t: float, gradients: List[float], clip_value: float, gamma: float) -> float:
    """POEM stochastic-feedback OGD update, kept aligned with the reference implementation."""
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
    """POEM linear betting factor."""
    bet = 1.0 + eps_t * (u_t - 0.5)
    return float(np.clip(bet, 1e-12, 2.0))


def load_clean_window_entropies(output_dirs: Dict[str, Path]) -> np.ndarray:
    """Use the saved Step 3 clean-validation windows as the POEM source calibration bank."""
    summary_csv_path = output_dirs["historical_bank"] / "clean_validation_window_summary.csv"
    if not summary_csv_path.exists():
        raise FileNotFoundError(
            f"POEM baseline requires {summary_csv_path}. Run Step 3 first."
        )
    rows = load_csv_rows(summary_csv_path)
    entropies = np.asarray([float(row["window_entropy"]) for row in rows], dtype=np.float64)
    if entropies.size == 0:
        raise ValueError("Clean historical-bank window summary has no entropy values for POEM calibration.")
    return entropies


def run_poem_on_rows(
    rows: Sequence[Dict[str, object]],
    clean_window_entropies: np.ndarray,
    evaluation_config: EvaluationConfig,
    window_size: int,
) -> List[Dict[str, object]]:
    """
    Run the POEM martingale on the exact same saved windows used by the proposed trigger.

    This keeps the POEM core logic intact while matching the current experiment's
    sample volume and windowing scheme for a fair comparison.
    """
    source_cdf = EmpiricalCDF(clean_window_entropies)
    wealth = 1.0
    epsilon_t = 0.0
    gradients: List[float] = []
    action_delay_steps = max(1, int(math.ceil(evaluation_config.poem_action_delay_samples / max(window_size, 1))))

    traced_rows: List[Dict[str, object]] = []
    for index, row in enumerate(rows):
        mean_entropy = float(row["avg_entropy"])
        u_t = source_cdf(mean_entropy)
        betting_factor = linear_betting_factor(u_t, epsilon_t)
        wealth *= betting_factor
        epsilon_next = sf_ogd_update(
            u_t=u_t,
            eps_t=epsilon_t,
            gradients=gradients,
            clip_value=evaluation_config.poem_martingale_clip,
            gamma=evaluation_config.poem_martingale_gamma,
        )
        trigger = int(index >= action_delay_steps and wealth >= evaluation_config.poem_wealth_threshold)
        updated = dict(row)
        updated["poem_uniformized_entropy"] = u_t
        updated["poem_betting_factor"] = betting_factor
        updated["poem_wealth"] = wealth
        updated["poem_epsilon"] = epsilon_t
        updated["poem_decision"] = trigger
        traced_rows.append(updated)
        epsilon_t = epsilon_next
    return traced_rows


def load_logits_for_window(
    row: Dict[str, object],
    output_dirs: Dict[str, Path],
    logits_cache: Dict[Tuple[str, str], np.ndarray],
) -> np.ndarray:
    """Load the exact Step 5 saved logits slice for one streamed window."""
    corruption_type = str(row["corruption_type"])
    split = str(row["split"])
    cache_key = (corruption_type, split)
    if cache_key not in logits_cache:
        candidate_paths = [
            output_dirs["pools"] / f"scan_{corruption_type}_{split}_logits.npy",
            output_dirs["historical_bank"] / f"{corruption_type}_{split}_logits.npy",
        ]
        if corruption_type == "clean" and split == "validation":
            candidate_paths.append(output_dirs["historical_bank"] / "clean_validation_logits.npy")

        logits_path = None
        for candidate in candidate_paths:
            if candidate.exists():
                logits_path = candidate
                break

        if logits_path is None:
            raise FileNotFoundError(
                "ARS baseline requires saved logits for the streamed window. "
                f"Tried: {[str(path) for path in candidate_paths]}"
            )
        logits_cache[cache_key] = np.load(logits_path, mmap_mode="r")
    start_index = int(row["start_index"])
    end_index = int(row["end_index_exclusive"])
    return np.asarray(logits_cache[cache_key][start_index:end_index], dtype=np.float64)


def compute_asr_c_t_from_logits(window_logits: np.ndarray) -> float:
    """Faithful ARS window statistic: C_t = sum softmax(mean_logits) * log softmax(mean_logits)."""
    if window_logits.ndim != 2 or window_logits.shape[1] != CIFAR_NUM_CLASSES:
        raise ValueError(f"Expected window logits with shape [N, {CIFAR_NUM_CLASSES}], got {window_logits.shape}.")
    mean_logits = window_logits.mean(axis=0)
    shifted = mean_logits - np.max(mean_logits)
    probs = np.exp(shifted)
    probs = probs / np.sum(probs)
    return float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))


def run_asr_on_rows(
    rows: Sequence[Dict[str, object]],
    output_dirs: Dict[str, Path],
    evaluation_config: EvaluationConfig,
) -> List[Dict[str, object]]:
    """
    Run the original ARS-style sequential concentration trigger on the same streamed windows.
    """
    num_classes = CIFAR_NUM_CLASSES
    c_bar_init = float(-math.log(evaluation_config.asr_alpha0 * num_classes))
    c_bar_prev = c_bar_init
    logits_cache: Dict[Tuple[str, str], np.ndarray] = {}
    traced_rows: List[Dict[str, object]] = []

    for row in rows:
        window_logits = load_logits_for_window(row=row, output_dirs=output_dirs, logits_cache=logits_cache)
        c_t = compute_asr_c_t_from_logits(window_logits)
        trigger = int(c_t > c_bar_prev)
        if trigger and evaluation_config.asr_reset_on_trigger:
            c_bar_new = c_bar_init
        else:
            c_bar_new = evaluation_config.asr_mu_c * c_bar_prev + (1.0 - evaluation_config.asr_mu_c) * c_t

        updated = dict(row)
        updated["asr_c_t"] = c_t
        updated["asr_c_bar_prev"] = c_bar_prev
        updated["asr_c_bar_new"] = c_bar_new
        updated["asr_mu_c"] = evaluation_config.asr_mu_c
        updated["asr_alpha0"] = evaluation_config.asr_alpha0
        updated["asr_decision"] = trigger
        traced_rows.append(updated)
        c_bar_prev = c_bar_new
    return traced_rows


def attach_decisions(rows: Sequence[Dict[str, object]], key: str, decisions: Sequence[int]) -> List[Dict[str, object]]:
    enriched = []
    for row, decision in zip(rows, decisions):
        updated = dict(row)
        updated[key] = int(decision)
        enriched.append(updated)
    return enriched


def plot_stream_results(
    rows: Sequence[Dict[str, object]],
    stream_name: str,
    plot_path: Path,
    trigger_threshold: float,
) -> None:
    indices = np.arange(len(rows))
    entropy = np.asarray([float(row["avg_entropy"]) for row in rows], dtype=np.float64)
    proxy_score = np.asarray([float(row["proxy_score"]) for row in rows], dtype=np.float64)
    input_score = np.asarray([float(row["input_score"]) for row in rows], dtype=np.float64)
    final_score = np.asarray([float(row["final_score"]) for row in rows], dtype=np.float64)
    trigger = np.asarray([int(bool(row["trigger"])) for row in rows], dtype=np.int64)
    oracle = np.asarray([int(row["oracle_trigger"]) for row in rows], dtype=np.int64)

    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    axes[0].plot(indices, entropy, color="tab:blue")
    axes[0].set_ylabel("Entropy")
    axes[0].set_title(stream_name)

    axes[1].plot(indices, proxy_score, color="tab:orange")
    axes[1].set_ylabel("Proxy")

    axes[2].plot(indices, input_score, color="tab:green")
    axes[2].set_ylabel("Input")

    axes[3].plot(indices, final_score, color="tab:red")
    axes[3].axhline(trigger_threshold, color="black", linestyle="--", linewidth=1)
    axes[3].set_ylabel("Final")

    axes[4].step(indices, oracle, where="mid", label="oracle", color="black")
    axes[4].step(indices, trigger, where="mid", label="proposed_trigger", color="tab:red")
    axes[4].set_ylabel("Trigger")
    axes[4].set_xlabel("Window Index")
    axes[4].legend(loc="upper right")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def run_joint_threshold_sweep(
    rows: Sequence[Dict[str, object]],
    evaluation_config: EvaluationConfig,
) -> List[Dict[str, object]]:
    sweep_rows: List[Dict[str, object]] = []
    for accuracy_threshold in evaluation_config.accuracy_thresholds:
        for final_score_threshold in evaluation_config.final_score_thresholds:
            decisions = baseline_accuracy_and_final_score(
                rows=rows,
                accuracy_threshold=accuracy_threshold,
                final_score_threshold=final_score_threshold,
            )
            method_rows = attach_decisions(rows, "joint_threshold_decision", decisions)
            metrics = compute_binary_metrics(method_rows, "joint_threshold_decision")
            sweep_row = {
                "accuracy_threshold": accuracy_threshold,
                "final_score_threshold": final_score_threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "accuracy": safe_divide(
                    metrics["tp"] + metrics["tn"],
                    metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"],
                ),
                "f1": metrics["f1"],
                "false_positive_rate": metrics["false_positive_rate"],
                "false_negative_rate": metrics["false_negative_rate"],
                "hidden_failure_recall": metrics["hidden_failure_recall"],
                "visible_failure_recall": metrics["visible_failure_recall"],
            }
            sweep_rows.append(sweep_row)
    return sweep_rows


def run_step9_evaluate_methods(
    output_dirs: Dict[str, Path],
    trigger_config: TriggerConfig,
    historical_bank_config: HistoricalBankConfig,
    evaluation_config: EvaluationConfig,
) -> Dict[str, object]:
    """
    Step 9 - evaluate only the proposed trigger and save plots.
    """
    print("Step 9: evaluating the proposed trigger only.")
    summary_path = output_dirs["window_results"] / "step8_trigger_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Step 9 requires {summary_path}. Run Step 8 first.")

    with summary_path.open("r", encoding="utf-8") as handle:
        step8_summary = json.load(handle)

    all_metric_rows: List[Dict[str, object]] = []
    stream_metrics_summary: Dict[str, object] = {}

    for stream_name, meta in step8_summary["step8_stream_summaries"].items():
        json_path = Path(meta["json_path"])
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload["results"]
        proposed_rows = attach_decisions(
            rows,
            "proposed_decision",
            [int(bool(row["trigger"])) for row in rows],
        )
        metrics = compute_binary_metrics(proposed_rows, "proposed_decision")
        metrics["accuracy"] = safe_divide(
            metrics["tp"] + metrics["tn"],
            metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"],
        )
        metrics["num_windows"] = len(rows)
        stream_metrics_summary[stream_name] = metrics
        metric_row = {"stream_name": stream_name, "method": "proposed"}
        metric_row.update(metrics)
        all_metric_rows.append(metric_row)

        plot_stream_results(
            rows=rows,
            stream_name=stream_name,
            plot_path=output_dirs["plots"] / f"{stream_name}_step9.png",
            trigger_threshold=trigger_config.trigger_threshold,
        )

    save_csv_rows(all_metric_rows, output_dirs["metrics"] / "step9_method_metrics.csv")

    summary = {
        "evaluation_config": asdict(evaluation_config),
        "stream_metrics_summary": stream_metrics_summary,
        "metrics_csv_path": str(output_dirs["metrics"] / "step9_method_metrics.csv"),
        "plots_dir": str(output_dirs["plots"]),
        "methods_evaluated": ["proposed"],
    }
    save_json(summary, output_dirs["metrics"] / "step9_evaluation_summary.json")
    return summary


def format_metric_table(rows: Sequence[Dict[str, object]]) -> str:
    if not rows:
        return "No metrics available."

    headers = [
        "stream_name",
        "method",
        "precision",
        "recall",
        "accuracy",
        "f1",
        "hidden_failure_recall",
        "visible_failure_recall",
    ]
    header_line = " | ".join(headers)
    separator = "-+-".join("-" * len(header) for header in headers)
    lines = [header_line, separator]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    str(row["stream_name"]),
                    str(row["method"]),
                    f"{float(row['precision']):.3f}",
                    f"{float(row['recall']):.3f}",
                    f"{float(row.get('accuracy', float('nan'))):.3f}",
                    f"{float(row['f1']):.3f}",
                    f"{float(row['hidden_failure_recall']):.3f}",
                    f"{float(row['visible_failure_recall']):.3f}",
                ]
            )
        )
    return "\n".join(lines)


def run_step10_finalize_outputs(
    output_dirs: Dict[str, Path],
    trigger_config: TriggerConfig,
    reporting_config: ReportingConfig,
) -> Dict[str, object]:
    """
    Step 10 - finalize outputs, save summaries, and print reproducible end report.
    """
    print("Step 10: finalizing outputs and writing experiment report.")
    metrics_dir = output_dirs["metrics"]
    window_results_dir = output_dirs["window_results"]

    training_summary_path = metrics_dir / "step2_training_summary.json"
    step9_summary_path = metrics_dir / "step9_evaluation_summary.json"
    step8_summary_path = window_results_dir / "step8_trigger_summary.json"
    step7_summary_path = window_results_dir / "step7_oracle_summary.json"

    final_payload: Dict[str, object] = {}

    if training_summary_path.exists():
        with training_summary_path.open("r", encoding="utf-8") as handle:
            training_summary = json.load(handle)
        final_payload["training_summary"] = training_summary
        global_clean_accuracy = float(training_summary["final_global_test_metrics"]["accuracy"])
    else:
        training_summary = None
        global_clean_accuracy = float("nan")

    if step9_summary_path.exists():
        with step9_summary_path.open("r", encoding="utf-8") as handle:
            step9_summary = json.load(handle)
        final_payload["step9_evaluation_summary"] = step9_summary
    else:
        step9_summary = None

    if step8_summary_path.exists():
        with step8_summary_path.open("r", encoding="utf-8") as handle:
            step8_summary = json.load(handle)
        final_payload["step8_trigger_summary"] = step8_summary
    else:
        step8_summary = None

    if step7_summary_path.exists():
        with step7_summary_path.open("r", encoding="utf-8") as handle:
            step7_summary = json.load(handle)
        final_payload["step7_oracle_summary"] = step7_summary
    else:
        step7_summary = None

    save_json(asdict(trigger_config), metrics_dir / "trigger_parameter_summary.json")

    metric_rows = load_csv_rows(metrics_dir / "step9_method_metrics.csv") if (metrics_dir / "step9_method_metrics.csv").exists() else []
    windows_per_stream: Dict[str, int] = {}
    trigger_counts: Dict[str, int] = {}
    hidden_detected = False

    if step8_summary is not None:
        for stream_name, meta in step8_summary["step8_stream_summaries"].items():
            windows_per_stream[stream_name] = int(meta["num_windows"])
            trigger_counts[stream_name] = int(meta["num_triggered"])

            result_path = Path(meta["json_path"])
            with result_path.open("r", encoding="utf-8") as handle:
                result_payload = json.load(handle)
            hidden_rows = [
                row for row in result_payload["results"]
                if "hidden_failure" in str(row["region_label"]).lower()
            ]
            if any(int(bool(row["trigger"])) == 1 for row in hidden_rows):
                hidden_detected = True

    report_lines = [
        "Composition-Aware FL-TTA Experiment Report",
        "",
        f"Global clean accuracy: {global_clean_accuracy:.4f}" if not math.isnan(global_clean_accuracy) else "Global clean accuracy: unavailable",
        f"Windows per stream: {json.dumps(windows_per_stream, indent=2)}",
        f"Trigger decisions per stream: {json.dumps(trigger_counts, indent=2)}",
        f"Hidden degradation successfully detected: {hidden_detected}",
        "",
        "Baseline comparison table:",
        format_metric_table(metric_rows),
        "",
        f"Artifacts root: {output_dirs['root']}",
    ]
    report_path = metrics_dir / reporting_config.report_filename
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    summary = {
        "global_clean_accuracy": global_clean_accuracy,
        "windows_per_stream": windows_per_stream,
        "trigger_counts": trigger_counts,
        "hidden_degradation_detected": hidden_detected,
        "trigger_parameter_summary_path": str(metrics_dir / "trigger_parameter_summary.json"),
        "text_report_path": str(report_path),
    }
    save_json(summary, metrics_dir / "step10_final_summary.json")

    print(
        "Final summary: "
        f"global_clean_accuracy={global_clean_accuracy:.4f} "
        f"hidden_detected={hidden_detected}"
        if not math.isnan(global_clean_accuracy)
        else f"Final summary: hidden_detected={hidden_detected}"
    )
    print(f"Windows per stream: {windows_per_stream}")
    print(f"Trigger decisions per stream: {trigger_counts}")
    if metric_rows:
        print("Baseline comparison:")
        for row in metric_rows:
            print(
                f"  {row['stream_name']} | {row['method']} | "
                f"F1={float(row['f1']):.3f} | "
                f"Recall={float(row['recall']):.3f} | "
                f"HiddenRecall={float(row['hidden_failure_recall']):.3f}"
            )

    return summary


def run_step5_scan_and_mine_pools(
    output_dirs: Dict[str, Path],
    federated_config: FederatedConfig,
    pool_mining_config: PoolMiningConfig,
    trigger_config: TriggerConfig,
    device: torch.device,
) -> Dict[str, object]:
    """
    Step 5 - scan corrupted data and mine the evaluation pools.
    """
    print("Step 5: scanning corrupted data and mining evaluation pools.")
    historical_contexts = load_historical_context_bank(output_dirs["historical_bank"])
    global_model = load_global_model_checkpoint(
        output_dirs["checkpoints"] / "global_model_final.pt",
        device=device,
    )

    mnist_c_root = Path(pool_mining_config.mnist_c_root)
    corruption_names = (
        pool_mining_config.corruption_names
        if pool_mining_config.corruption_names is not None
        else list_available_cifar10_c_corruptions(mnist_c_root)
    )

    all_sample_rows: List[Dict[str, object]] = []
    all_window_rows: List[Dict[str, object]] = []
    pool_array_paths: Dict[str, Dict[str, str]] = {}

    for corruption_idx, corruption_name in enumerate(corruption_names, start=1):
        print(
            f"  Step 5: scanning corruption {corruption_idx}/{len(corruption_names)} -> {corruption_name}",
            flush=True,
        )
        dataset, metadata = load_cifar10_c_dataset(
            cifar10_c_root=mnist_c_root,
            corruption_name=corruption_name,
            split=pool_mining_config.split,
            max_samples=pool_mining_config.max_samples_per_corruption,
            severities=[1, 2, 3, 4, 5],
        )
        loader = create_eval_loader(
            dataset=dataset,
            batch_size=federated_config.eval_batch_size,
            num_workers=federated_config.num_workers,
        )
        sample_rows, window_rows, arrays = scan_corruption_dataset(
            model=global_model,
            loader=loader,
            device=device,
            corruption_name=corruption_name,
            split=pool_mining_config.split,
            historical_contexts=historical_contexts,
            trigger_config=trigger_config,
            window_size=pool_mining_config.window_size,
            severity_bucket_size=pool_mining_config.severity_bucket_size,
            sample_index_offset=int(metadata["original_indices_start"]),
        )
        all_sample_rows.extend(sample_rows)
        all_window_rows.extend(window_rows)
        pool_array_paths[corruption_name] = save_window_arrays(
            arrays,
            output_dirs["pools"] / f"scan_{corruption_name}_{pool_mining_config.split}",
        )
        print(
            f"    completed {corruption_name}: samples={len(sample_rows)}, windows={len(window_rows)}",
            flush=True,
        )

    sample_entropy = np.asarray([float(row["entropy"]) for row in all_sample_rows], dtype=np.float64)
    sample_input_dist = np.asarray(
        [float(row["distance_from_closest_historical_context"]) for row in all_sample_rows],
        dtype=np.float64,
    )
    window_accuracy = np.asarray([float(row["window_accuracy"]) for row in all_window_rows], dtype=np.float64)
    window_entropy = np.asarray([float(row["window_entropy"]) for row in all_window_rows], dtype=np.float64)
    window_input_dist = np.asarray(
        [float(row["distance_from_closest_historical_context"]) for row in all_window_rows],
        dtype=np.float64,
    )
    window_conf = np.asarray([float(row["window_max_probability"]) for row in all_window_rows], dtype=np.float64)
    window_output_dev = np.asarray(
        [float(row["output_distribution_deviation"]) for row in all_window_rows],
        dtype=np.float64,
    )
    thresholds = {
        "low_entropy": float(np.quantile(sample_entropy, 0.25)),
        "mild_entropy": float(np.quantile(sample_entropy, 0.50)),
        "high_entropy": float(np.quantile(sample_entropy, 0.80)),
        "hidden_entropy": float(np.quantile(sample_entropy, 0.30)),
        "low_input_distance": float(np.quantile(sample_input_dist, 0.25)),
        "moderate_input_distance": float(np.quantile(sample_input_dist, 0.55)),
        "high_input_distance": float(np.quantile(sample_input_dist, 0.80)),
        "high_confidence": 0.85,
        "stable_accuracy": 0.90,
        "mild_accuracy": 0.75,
        "failure_accuracy": 0.50,
        "high_output_deviation": float(np.quantile(window_output_dev, 0.80)),
    }
    window_thresholds = {
        "low_entropy": float(np.quantile(window_entropy, 0.20)),
        "mild_entropy": float(np.quantile(window_entropy, 0.50)),
        "high_entropy": float(np.quantile(window_entropy, 0.80)),
        "hidden_entropy": max(float(np.quantile(window_entropy, 0.35)), 0.35),
        "low_input_distance": float(np.quantile(window_input_dist, 0.25)),
        "moderate_input_distance": float(np.quantile(window_input_dist, 0.55)),
        "high_input_distance": float(np.quantile(window_input_dist, 0.80)),
        "high_confidence": max(0.85, float(np.quantile(window_conf, 0.70))),
        "stable_accuracy": max(0.95, float(np.quantile(window_accuracy, 0.75))),
        "mild_accuracy": max(0.90, float(np.quantile(window_accuracy, 0.50))),
        "failure_accuracy": min(0.80, float(np.quantile(window_accuracy, 0.20))),
        "high_output_deviation": float(np.quantile(window_output_dev, 0.80)),
    }

    assign_sample_pools(all_sample_rows, thresholds)
    assign_window_pools(all_window_rows, window_thresholds)
    ensure_required_window_pools(all_window_rows, min_windows_per_pool=8)

    current_counts = summarize_pool_assignments(all_window_rows, "window_id")["counts"]
    if int(current_counts["stable"]) < 8:
        all_window_rows.extend(
            build_pool_rows_from_historical_summary(
                summary_csv_path=output_dirs["historical_bank"] / "clean_validation_window_summary.csv",
                pool_name="stable",
                corruption_type="clean",
                split="validation",
                matched_context=CLEAN_CONTEXT_NAME,
            )
        )
    if int(current_counts["mild_variability"]) < 8:
        for context_name in ["brightness", "motion_blur"]:
            csv_path = output_dirs["historical_bank"] / f"{context_name}_test_window_summary.csv"
            if csv_path.exists():
                all_window_rows.extend(
                    build_pool_rows_from_historical_summary(
                        summary_csv_path=csv_path,
                        pool_name="mild_variability",
                        corruption_type=context_name,
                        split="test",
                        matched_context=f"known_{context_name}",
                    )
                )

    ensure_required_window_pools(all_window_rows, min_windows_per_pool=8)

    save_csv_rows(all_sample_rows, output_dirs["pools"] / "step5_sample_scan.csv")
    save_csv_rows(all_window_rows, output_dirs["pools"] / "step5_window_scan.csv")

    sample_pool_summary = summarize_pool_assignments(all_sample_rows, "sample_id")
    window_pool_summary = summarize_pool_assignments(all_window_rows, "window_id")
    summary = {
        "pool_mining_config": asdict(pool_mining_config),
        "corruption_names_scanned": corruption_names,
        "sample_thresholds": thresholds,
        "window_thresholds": window_thresholds,
        "sample_pool_summary": sample_pool_summary,
        "window_pool_summary": window_pool_summary,
        "array_paths": pool_array_paths,
        "severity_note": (
            f"Severity values in Step 5 follow the official {CIFAR_C_DATASET_NAME} severity blocks "
            f"using bucket_size={pool_mining_config.severity_bucket_size}."
        ),
    }
    save_json(summary, output_dirs["pools"] / "step5_pool_assignments_summary.json")
    return summary


def create_demo_window(
    window_index: int,
    label_bias: int,
    entropy_scale: float,
    input_shift: Sequence[float],
    samples: int = 32,
) -> WindowStats:
    """
    Small synthetic helper so Step 1 can be executed and inspected in isolation.
    """
    logits = torch.randn(samples, CIFAR_NUM_CLASSES) / max(entropy_scale, 1e-4)
    logits[:, label_bias % CIFAR_NUM_CLASSES] += 2.5

    images = torch.rand(samples, 3, 32, 32) * 0.15
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    cx = 16.0 + float(input_shift[0])
    cy = 16.0 + float(input_shift[1])
    radius = 7.0 + float(input_shift[2])
    blob = (((xx - cx) ** 2 + (yy - cy) ** 2) < radius ** 2).float()
    color_blob = torch.stack([blob, 0.7 * blob, 0.4 * blob], dim=0)
    images = torch.clamp(images + 0.65 * color_blob.unsqueeze(0), 0.0, 1.0)

    return compute_window_stats(logits=logits, images=images, window_index=window_index, epsilon=1e-8)


def build_demo_contexts() -> List[HistoricalContext]:
    """
    Three minimal contexts matching the requested design:
      context 1 = clean CIFAR-100
      context 2 = mild known corruption A
      context 3 = mild known corruption B
    """
    clean_windows = [create_demo_window(i, label_bias=1, entropy_scale=1.8, input_shift=(0.0, 0.0, 0.0)) for i in range(4)]
    noise_windows = [create_demo_window(i, label_bias=1, entropy_scale=1.4, input_shift=(0.8, 0.5, 0.3)) for i in range(4, 8)]
    blur_windows = [create_demo_window(i, label_bias=1, entropy_scale=1.5, input_shift=(-0.7, 0.6, 0.2)) for i in range(8, 12)]

    return [
        build_historical_context(CLEAN_CONTEXT_NAME, clean_windows, {"type": "clean"}),
        build_historical_context("mild_corruption_a", noise_windows, {"type": "gaussian_noise_severity_1"}),
        build_historical_context("mild_corruption_b", blur_windows, {"type": "motion_blur_severity_1"}),
    ]


def run_demo(output_dir: Path, config: TriggerConfig) -> None:
    """
    Execute a small synthetic demo to verify Step 1 end-to-end.
    """
    contexts = build_demo_contexts()
    trigger = CompositionAwareTrigger(config=config, historical_contexts=contexts)

    demo_windows = [
        create_demo_window(0, label_bias=1, entropy_scale=1.8, input_shift=(0.1, 0.1, 0.0)),   # benign
        create_demo_window(1, label_bias=1, entropy_scale=0.55, input_shift=(3.0, 3.0, 3.5)),  # visible
        create_demo_window(2, label_bias=1, entropy_scale=3.5, input_shift=(3.5, 2.8, 3.0)),   # hidden-style
    ]

    results = [asdict(trigger.process_window(window)) for window in demo_windows]
    payload = {
        "config": asdict(config),
        "historical_contexts": [asdict(context) for context in contexts],
        "results": results,
    }
    save_json(payload, output_dir / "step1_demo_results.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Composition-aware FL-TTA experiment on CIFAR-100. Step 1 and Step 2 are CIFAR-ready."
    )
    parser.add_argument("--output-dir", type=str, default="outputs_experiment", help="Directory for experiment artifacts.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="Dataset",
        help="Local CIFAR-100 dataset root.",
    )
    parser.add_argument(
        "--mnist-c-root",
        type=str,
        default="Dataset/CIFAR-100-C/CIFAR-100-C",
        help="Local CIFAR-100-C root with one .npy file per corruption.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use: auto, cpu, or cuda.")
    parser.add_argument("--gamma-H", dest="gamma_H", type=float, default=3.0)
    parser.add_argument("--gamma-I", dest="gamma_I", type=float, default=2.5)
    parser.add_argument("--w-H", dest="w_H", type=float, default=0.5)
    parser.add_argument("--w-O", dest="w_O", type=float, default=0.5)
    parser.add_argument("--tau-proxy", dest="tau_proxy", type=float, default=0.45)
    parser.add_argument("--tau-input", dest="tau_input", type=float, default=0.45)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--lambda-hidden", dest="lambda_hidden", type=float, default=1.0)
    parser.add_argument("--trigger-threshold", dest="trigger_threshold", type=float, default=0.18)
    parser.add_argument("--disable-hidden-branch", action="store_true")
    parser.add_argument("--disable-input-persistence", action="store_true")
    parser.add_argument("--disable-proxy-persistence", action="store_true")
    parser.set_defaults(skip_demo=True)
    parser.add_argument(
        "--run-demo",
        dest="skip_demo",
        action="store_false",
        help="Run the optional synthetic Step 1 demo.",
    )
    parser.add_argument("--skip-step2", action="store_true", help="Skip federated training.")
    parser.add_argument("--skip-step3", action="store_true", help="Skip clean historical-bank construction.")
    parser.add_argument("--skip-step4", action="store_true", help="Skip known corruption-context construction.")
    parser.add_argument("--skip-step5", action="store_true", help="Skip corrupted-pool scanning and mining.")
    parser.add_argument("--skip-step6", action="store_true", help="Skip streaming scenario construction.")
    parser.add_argument("--skip-step7", action="store_true", help="Skip oracle trigger label construction.")
    parser.add_argument("--skip-step8", action="store_true", help="Skip trigger execution on streams.")
    parser.add_argument("--skip-step9", action="store_true", help="Skip baseline evaluation and plotting.")
    parser.add_argument("--skip-step10", action="store_true", help="Skip final reporting and summary export.")
    parser.add_argument("--rounds", type=int, default=5, help="FedAvg rounds.")
    parser.add_argument("--local-epochs", type=int, default=2, help="Local epochs per round.")
    parser.add_argument("--batch-size", type=int, default=64, help="Client batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=256, help="Evaluation batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Local optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Local optimizer weight decay.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--num-threads", type=int, default=1, help="Torch CPU threads.")
    parser.add_argument(
        "--train-sample-cap",
        type=int,
        default=None,
        help="Optional cap on the total FL training pool before client assignment.",
    )
    parser.add_argument(
        "--client-sample-cap",
        type=int,
        default=None,
        help="Optional cap on samples per client for faster runs.",
    )
    parser.add_argument("--val-split", type=int, default=5000, help="Validation split size from training set.")
    parser.add_argument("--window-size", type=int, default=128, help="Window size for Step 3 statistics.")
    parser.add_argument(
        "--context-corruptions",
        type=str,
        default="brightness,motion_blur",
        help="Comma-separated CIFAR-100-C corruption names to use as Step 4 historical contexts.",
    )
    parser.add_argument(
        "--mnist-c-split",
        type=str,
        default="test",
        choices=["test"],
        help="CIFAR-100-C split to use for Step 4.",
    )
    parser.add_argument(
        "--max-context-samples",
        type=int,
        default=10000,
        help="Number of samples to use per Step 4 context. Default maps to one severity block (10,000).",
    )
    parser.add_argument(
        "--pool-corruptions",
        type=str,
        default="",
        help="Optional comma-separated corruption names for Step 5. Empty means scan all available CIFAR-100-C corruptions.",
    )
    parser.add_argument(
        "--pool-split",
        type=str,
        default="test",
        choices=["test"],
        help="CIFAR-100-C split to scan for Step 5.",
    )
    parser.add_argument(
        "--max-pool-samples",
        type=int,
        default=None,
        help="Optional cap per corruption for Step 5 scanning.",
    )
    parser.add_argument(
        "--severity-bucket-size",
        type=int,
        default=10000,
        help="Severity block size for CIFAR-100-C labels in Step 5.",
    )
    parser.add_argument(
        "--windows-per-segment",
        type=int,
        default=4,
        help="Number of windows per region segment when building Step 6 streams.",
    )
    parser.add_argument(
        "--relative-accuracy-drop-threshold",
        type=float,
        default=0.15,
        help="Auxiliary oracle threshold for relative accuracy drop in Step 7.",
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=0.75,
        help="Entropy threshold for the Step 9 entropy-only baseline.",
    )
    parser.add_argument(
        "--accuracy-thresholds",
        type=str,
        default="0.50,0.60,0.70,0.80,0.90",
        help="Comma-separated accuracy thresholds for the joint Step 9 sweep.",
    )
    parser.add_argument(
        "--final-score-thresholds",
        type=str,
        default="0.10,0.15,0.20,0.25,0.30",
        help="Comma-separated final-score thresholds for the joint Step 9 sweep.",
    )
    parser.add_argument(
        "--poem-wealth-threshold",
        type=float,
        default=10.0,
        help="POEM martingale wealth threshold used in Step 9 fair-comparison evaluation.",
    )
    parser.add_argument(
        "--poem-martingale-clip",
        type=float,
        default=1.8,
        help="POEM clipping constant for the SF-OGD update.",
    )
    parser.add_argument(
        "--poem-martingale-gamma",
        type=float,
        default=1.0 / (8.0 * math.sqrt(3.0)),
        help="POEM SF-OGD learning-rate constant.",
    )
    parser.add_argument(
        "--poem-action-delay-samples",
        type=int,
        default=100,
        help="POEM action delay in samples; converted into window steps for the current setup.",
    )
    parser.add_argument(
        "--asr-mu-c",
        type=float,
        default=0.99,
        help="ARS smoothing factor for the running concentration threshold.",
    )
    parser.add_argument(
        "--asr-alpha0",
        type=float,
        default=0.1,
        help="ARS initial-threshold parameter used in c_bar_init = -log(alpha0 * C).",
    )
    parser.add_argument(
        "--disable-asr-reset-on-trigger",
        action="store_true",
        help="Disable the ARS threshold reset after each trigger.",
    )
    parser.add_argument(
        "--report-filename",
        type=str,
        default="step10_report.txt",
        help="Filename for the Step 10 text report under metrics/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    config = TriggerConfig(
        gamma_H=args.gamma_H,
        gamma_I=args.gamma_I,
        w_H=args.w_H,
        w_O=args.w_O,
        tau_proxy=args.tau_proxy,
        tau_input=args.tau_input,
        K=args.K,
        lambda_hidden=args.lambda_hidden,
        trigger_threshold=args.trigger_threshold,
        disable_hidden_branch=args.disable_hidden_branch,
        disable_input_persistence=args.disable_input_persistence,
        disable_proxy_persistence=args.disable_proxy_persistence,
    )
    config.validate()

    federated_config = FederatedConfig(
        data_root=args.data_root,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        num_threads=args.num_threads,
        train_sample_cap=args.train_sample_cap,
        client_sample_cap=args.client_sample_cap,
        val_split=args.val_split,
    )
    historical_bank_config = HistoricalBankConfig(window_size=args.window_size)
    corruption_context_config = CorruptionContextConfig(
        mnist_c_root=args.mnist_c_root,
        context_names=[name.strip() for name in args.context_corruptions.split(",") if name.strip()],
        split=args.mnist_c_split,
        max_samples_per_context=args.max_context_samples,
    )
    pool_mining_config = PoolMiningConfig(
        mnist_c_root=args.mnist_c_root,
        split=args.pool_split,
        corruption_names=[name.strip() for name in args.pool_corruptions.split(",") if name.strip()] or None,
        max_samples_per_corruption=args.max_pool_samples,
        window_size=args.window_size,
        severity_bucket_size=args.severity_bucket_size,
    )
    stream_config = StreamConfig(
        random_seed=args.seed,
        windows_per_segment=args.windows_per_segment,
    )
    oracle_config = OracleConfig(
        relative_accuracy_drop_threshold=args.relative_accuracy_drop_threshold,
    )
    trigger_run_config = TriggerRunConfig()
    evaluation_config = EvaluationConfig(
        entropy_threshold=args.entropy_threshold,
        accuracy_thresholds=parse_float_list(args.accuracy_thresholds),
        final_score_thresholds=parse_float_list(args.final_score_thresholds),
        poem_wealth_threshold=args.poem_wealth_threshold,
        poem_martingale_clip=args.poem_martingale_clip,
        poem_martingale_gamma=args.poem_martingale_gamma,
        poem_action_delay_samples=args.poem_action_delay_samples,
        asr_mu_c=args.asr_mu_c,
        asr_alpha0=args.asr_alpha0,
        asr_reset_on_trigger=not args.disable_asr_reset_on_trigger,
    )
    reporting_config = ReportingConfig(
        report_filename=args.report_filename,
    )

    output_dirs = build_output_dirs(Path(args.output_dir))
    save_json(
        {
            "trigger_config": asdict(config),
            "federated_config": asdict(federated_config),
            "historical_bank_config": asdict(historical_bank_config),
            "corruption_context_config": asdict(corruption_context_config),
            "pool_mining_config": asdict(pool_mining_config),
            "stream_config": asdict(stream_config),
            "oracle_config": asdict(oracle_config),
            "trigger_run_config": asdict(trigger_run_config),
            "evaluation_config": asdict(evaluation_config),
            "reporting_config": asdict(reporting_config),
            "seed": args.seed,
            "device": str(device),
        },
        output_dirs["metrics"] / "experiment_config.json",
    )

    if not args.skip_demo:
        run_demo(output_dir=output_dirs["metrics"], config=config)

    step2_summary = None
    if not args.skip_step2:
        step2_summary = run_step2_federated_training(
            output_dirs=output_dirs,
            federated_config=federated_config,
            seed=args.seed,
            device=device,
        )

    step3_summary = None
    if not args.skip_step3:
        step3_summary = run_step3_clean_historical_bank(
            output_dirs=output_dirs,
            federated_config=federated_config,
            historical_bank_config=historical_bank_config,
            trigger_config=config,
            device=device,
        )

    step4_summary = None
    if not args.skip_step4:
        step4_summary = run_step4_known_corruption_contexts(
            output_dirs=output_dirs,
            federated_config=federated_config,
            historical_bank_config=historical_bank_config,
            corruption_context_config=corruption_context_config,
            trigger_config=config,
            device=device,
        )

    step5_summary = None
    if not args.skip_step5:
        step5_summary = run_step5_scan_and_mine_pools(
            output_dirs=output_dirs,
            federated_config=federated_config,
            pool_mining_config=pool_mining_config,
            trigger_config=config,
            device=device,
        )

    step6_summary = None
    if not args.skip_step6:
        step6_summary = run_step6_create_streams(
            output_dirs=output_dirs,
            stream_config=stream_config,
        )

    step7_summary = None
    if not args.skip_step7:
        step7_summary = run_step7_define_oracle_labels(
            output_dirs=output_dirs,
            oracle_config=oracle_config,
        )

    step8_summary = None
    if not args.skip_step8:
        step8_summary = run_step8_execute_trigger(
            output_dirs=output_dirs,
            trigger_config=config,
            trigger_run_config=trigger_run_config,
        )

    step9_summary = None
    if not args.skip_step9:
        step9_summary = run_step9_evaluate_methods(
            output_dirs=output_dirs,
            trigger_config=config,
            historical_bank_config=historical_bank_config,
            evaluation_config=evaluation_config,
        )

    step10_summary = None
    if not args.skip_step10:
        step10_summary = run_step10_finalize_outputs(
            output_dirs=output_dirs,
            trigger_config=config,
            reporting_config=reporting_config,
        )

    print("Step 1 complete.")
    print(f"Experiment config saved to: {output_dirs['metrics'] / 'experiment_config.json'}")
    if not args.skip_demo:
        print(f"Demo results saved to: {output_dirs['metrics'] / 'step1_demo_results.json'}")
    if step2_summary is not None:
        print("Step 2 complete.")
        print(
            "Final global clean test accuracy: "
            f"{step2_summary['final_global_test_metrics']['accuracy']:.4f}"
        )
        print(f"Training summary saved to: {output_dirs['metrics'] / 'step2_training_summary.json'}")
    if step3_summary is not None:
        print("Step 3 complete.")
        print(
            "Clean validation accuracy: "
            f"{step3_summary['clean_validation_metrics']['clean_val_accuracy']:.4f}"
        )
        print(
            "Clean historical bank saved to: "
            f"{output_dirs['historical_bank'] / 'step3_clean_historical_bank_summary.json'}"
        )
    if step4_summary is not None:
        print("Step 4 complete.")
        print(
            "Known corruption contexts saved to: "
            f"{output_dirs['historical_bank'] / 'step4_known_corruption_contexts_summary.json'}"
        )
    if step5_summary is not None:
        print("Step 5 complete.")
        print(
            "Pool mining summary saved to: "
            f"{output_dirs['pools'] / 'step5_pool_assignments_summary.json'}"
        )
    if step6_summary is not None:
        print("Step 6 complete.")
        print(
            "Stream summary saved to: "
            f"{output_dirs['stream_definitions'] / 'step6_stream_summary.json'}"
        )
    if step7_summary is not None:
        print("Step 7 complete.")
        print(
            "Oracle summary saved to: "
            f"{output_dirs['window_results'] / 'step7_oracle_summary.json'}"
        )
    if step8_summary is not None:
        print("Step 8 complete.")
        print(
            "Trigger summary saved to: "
            f"{output_dirs['window_results'] / 'step8_trigger_summary.json'}"
        )
    if step9_summary is not None:
        print("Step 9 complete.")
        print(
            "Evaluation summary saved to: "
            f"{output_dirs['metrics'] / 'step9_evaluation_summary.json'}"
        )
    if step10_summary is not None:
        print("Step 10 complete.")
        print(
            "Final report saved to: "
            f"{output_dirs['metrics'] / reporting_config.report_filename}"
        )


if __name__ == "__main__":
    main()
