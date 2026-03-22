from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import composition_aware_fl_tta_CIFAR as exp


@dataclass
class DSSConfig:
    experiment_output_dir: str
    checkpoint_path: str
    cifar_c_root: str
    split: str
    max_samples_per_corruption: int
    window_size: int
    batch_size: int
    num_workers: int
    device: str
    seed: int
    base_tau: float
    threshold_offsets: Sequence[float]
    output_dir: str


class FeatureOnlyCNN(nn.Module):
    """Use the source model as the paper's fixed source feature extractor f."""

    def __init__(self, model: exp.SimpleCIFARCNN) -> None:
        super().__init__()
        self.features = model.features
        self.classifier = model.classifier[:-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone DSS trigger evaluation using the exact paper logic "
            "on the saved CIFAR Step 5/6 artifacts."
        )
    )
    parser.add_argument(
        "--experiment-output-dir",
        type=str,
        default="/Users/deepakkanneganti/Documents/Experiment CIFAR/outputs_step1_step2_cifar_30k_5r_2e_fixed",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/Users/deepakkanneganti/Documents/Experiment CIFAR/outputs_step1_step2_cifar_30k_5r_2e_fixed/checkpoints/global_model_final.pt",
    )
    parser.add_argument(
        "--cifar-c-root",
        type=str,
        default="/Users/deepakkanneganti/Documents/Experiment CIFAR/Dataset/CIFAR-10-C/CIFAR-10-C",
    )
    parser.add_argument("--split", type=str, default="test", choices=["test"])
    parser.add_argument("--max-samples-per-corruption", type=int, default=20000)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--base-tau",
        type=float,
        default=0.98,
        help="Default DSS similarity threshold used for the stream replay.",
    )
    parser.add_argument(
        "--threshold-offsets",
        type=str,
        default="0.00,0.005,0.010,0.015,0.020",
        help="Offsets added to each corruption's mean DSS for the local threshold sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/deepakkanneganti/Documents/Experiment CIFAR/DSS/outputs",
    )
    return parser.parse_args()


def parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def load_source_feature_extractor(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = exp.SimpleCIFARCNN().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    feature_extractor = FeatureOnlyCNN(model).to(device)
    feature_extractor.eval()
    return feature_extractor


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 1.0
    return float(np.dot(a, b) / denom)


def compute_binary_metrics(rows: Sequence[Dict[str, object]], decision_key: str) -> Dict[str, float]:
    y_true = np.asarray([int(row["oracle_trigger"]) for row in rows], dtype=np.int64)
    y_pred = np.asarray([int(bool(row[decision_key])) for row in rows], dtype=np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
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


def load_corruption_dataset(
    cifar_c_root: Path,
    corruption_name: str,
    split: str,
    max_samples: int,
) -> exp.NumpyImageDataset:
    dataset, _ = exp.load_cifar10_c_dataset(
        cifar10_c_root=cifar_c_root,
        corruption_name=corruption_name,
        split=split,
        max_samples=max_samples,
        severities=[1, 2, 3, 4, 5],
    )
    return dataset


@torch.no_grad()
def extract_feature_matrix(
    feature_extractor: nn.Module,
    dataset: exp.NumpyImageDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    feature_batches: List[np.ndarray] = []
    for images, _ in loader:
        features = feature_extractor(images.to(device)).cpu().numpy()
        feature_batches.append(features.astype(np.float32))
    return np.concatenate(feature_batches, axis=0)


def run_dss_sequence(
    feature_means: Sequence[np.ndarray],
    tau: float,
) -> List[Dict[str, float]]:
    """
    Exact paper trigger:
    DSS_t = cosine(E[v_f(t)], E[v_f(t-1)])
    trigger if DSS_t < tau.
    """
    records: List[Dict[str, float]] = []
    prev_mean: np.ndarray | None = None
    for index, current_mean in enumerate(feature_means):
        if prev_mean is None:
            dss = 1.0
            trigger = 0
        else:
            dss = cosine_similarity(current_mean, prev_mean)
            trigger = int(dss < tau)
        records.append(
            {
                "window_index": index,
                "dss_similarity": dss,
                "tau_threshold": tau,
                "dss_decision": trigger,
            }
        )
        prev_mean = current_mean
    return records


def rows_to_feature_means(
    rows: Sequence[Dict[str, object]],
    feature_matrix: np.ndarray,
) -> List[np.ndarray]:
    means: List[np.ndarray] = []
    for row in rows:
        start_index = int(row["start_index"])
        end_index = int(row["end_index_exclusive"])
        means.append(feature_matrix[start_index:end_index].mean(axis=0))
    return means


def save_json(data: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_csv_rows(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = DSSConfig(
        experiment_output_dir=args.experiment_output_dir,
        checkpoint_path=args.checkpoint_path,
        cifar_c_root=args.cifar_c_root,
        split=args.split,
        max_samples_per_corruption=args.max_samples_per_corruption,
        window_size=args.window_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        base_tau=args.base_tau,
        threshold_offsets=parse_float_list(args.threshold_offsets),
        output_dir=args.output_dir,
    )
    exp.set_seed(config.seed)
    device = get_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_extractor = load_source_feature_extractor(Path(config.checkpoint_path), device)
    experiment_output_dir = Path(config.experiment_output_dir)
    step5_csv = experiment_output_dir / "pools" / "step5_window_scan.csv"
    step8_summary_path = experiment_output_dir / "window_results" / "step8_trigger_summary.json"

    if not step5_csv.exists():
        raise FileNotFoundError(f"Expected Step 5 window scan at {step5_csv}")
    if not step8_summary_path.exists():
        raise FileNotFoundError(f"Expected Step 8 summary at {step8_summary_path}")

    step5_rows = exp.load_csv_rows(step5_csv)
    with step8_summary_path.open("r", encoding="utf-8") as handle:
        step8_summary = json.load(handle)

    grouped_step5_rows: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in step5_rows:
        if str(row["corruption_type"]) == "clean":
            continue
        grouped_step5_rows[str(row["corruption_type"])].append(row)
    for corruption_name in grouped_step5_rows:
        grouped_step5_rows[corruption_name].sort(key=lambda item: int(item["window_index"]))

    feature_cache: Dict[str, np.ndarray] = {}
    per_corruption_best_rows: List[Dict[str, object]] = []
    all_sweep_rows: List[Dict[str, object]] = []

    for corruption_name, corruption_rows in sorted(grouped_step5_rows.items()):
        dataset = load_corruption_dataset(
            cifar_c_root=Path(config.cifar_c_root),
            corruption_name=corruption_name,
            split=config.split,
            max_samples=config.max_samples_per_corruption,
        )
        feature_matrix = extract_feature_matrix(
            feature_extractor=feature_extractor,
            dataset=dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            device=device,
        )
        feature_cache[corruption_name] = feature_matrix

        feature_means = rows_to_feature_means(corruption_rows, feature_matrix)
        base_records = run_dss_sequence(feature_means, tau=config.base_tau)
        valid_dss = [record["dss_similarity"] for record in base_records[1:]] if len(base_records) > 1 else [1.0]
        mean_dss = float(np.mean(valid_dss))

        best_row: Dict[str, object] | None = None
        for offset in config.threshold_offsets:
            tau = float(np.clip(config.base_tau + offset, -1.0, 1.0))
            records = run_dss_sequence(feature_means, tau=tau)
            enriched_rows: List[Dict[str, object]] = []
            for row, record in zip(corruption_rows, records):
                enriched = dict(row)
                enriched.update(record)
                enriched["oracle_trigger"] = int(
                    str(row["pool_name"]) in {"visible_failure", "hidden_failure"}
                    or (
                        str(row["pool_name"]) == "unassigned"
                        and float(row["window_accuracy"])
                        < float(np.mean([float(item["window_accuracy"]) for item in corruption_rows]))
                    )
                )
                enriched_rows.append(enriched)
            metrics = compute_binary_metrics(enriched_rows, "dss_decision")
            sweep_row = {
                "corruption": corruption_name,
                "mean_window_accuracy": float(np.mean([float(row["window_accuracy"]) for row in corruption_rows])),
                "mean_dss_similarity": mean_dss,
                "tau_threshold": tau,
                **metrics,
            }
            all_sweep_rows.append(sweep_row)
            score = (metrics["f1"], metrics["precision"], metrics["recall"], metrics["accuracy"])
            if best_row is None or score > (
                float(best_row["f1"]),
                float(best_row["precision"]),
                float(best_row["recall"]),
                float(best_row["accuracy"]),
            ):
                best_row = sweep_row
        assert best_row is not None
        per_corruption_best_rows.append(best_row)

    save_csv_rows(all_sweep_rows, output_dir / "dss_per_corruption_sweep.csv")
    save_csv_rows(per_corruption_best_rows, output_dir / "dss_per_corruption_best.csv")

    stream_metric_rows: List[Dict[str, object]] = []
    stream_summary: Dict[str, object] = {}
    for stream_name, meta in step8_summary["step8_stream_summaries"].items():
        result_json_path = Path(meta["json_path"])
        with result_json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload["results"]

        stream_feature_means: List[np.ndarray] = []
        for row in rows:
            corruption_name = str(row["corruption_type"])
            if corruption_name not in feature_cache:
                dataset = load_corruption_dataset(
                    cifar_c_root=Path(config.cifar_c_root),
                    corruption_name=corruption_name,
                    split=str(row["split"]),
                    max_samples=config.max_samples_per_corruption,
                )
                feature_cache[corruption_name] = extract_feature_matrix(
                    feature_extractor=feature_extractor,
                    dataset=dataset,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    device=device,
                )
            start_index = int(row["start_index"])
            end_index = int(row["end_index_exclusive"])
            stream_feature_means.append(feature_cache[corruption_name][start_index:end_index].mean(axis=0))

        dss_rows = run_dss_sequence(stream_feature_means, tau=config.base_tau)
        traced_rows: List[Dict[str, object]] = []
        for row, dss_row in zip(rows, dss_rows):
            updated = dict(row)
            updated.update(dss_row)
            traced_rows.append(updated)

        metrics = compute_binary_metrics(traced_rows, "dss_decision")
        metric_row = {
            "stream_name": stream_name,
            "method": "dss",
            **metrics,
        }
        stream_metric_rows.append(metric_row)
        save_json(
            {
                "stream_name": stream_name,
                "dss_config": {
                    "tau_threshold": config.base_tau,
                    "window_size": config.window_size,
                    "paper_logic": "trigger if cosine(mean_feature_t, mean_feature_t_minus_1) < tau",
                },
                "results": traced_rows,
            },
            output_dir / f"{stream_name}_dss_results.json",
        )
        save_csv_rows(traced_rows, output_dir / f"{stream_name}_dss_results.csv")
        stream_summary[stream_name] = {
            "metrics": metrics,
            "json_path": str(output_dir / f"{stream_name}_dss_results.json"),
            "csv_path": str(output_dir / f"{stream_name}_dss_results.csv"),
        }

    save_csv_rows(stream_metric_rows, output_dir / "dss_stream_metrics.csv")
    save_json(
        {
            "config": asdict(config),
            "paper_logic_verified": (
                "DSS = cosine similarity between consecutive batch mean source features; "
                "trigger when DSS < tau."
            ),
            "per_corruption_best_csv_path": str(output_dir / "dss_per_corruption_best.csv"),
            "per_corruption_sweep_csv_path": str(output_dir / "dss_per_corruption_sweep.csv"),
            "stream_metrics_csv_path": str(output_dir / "dss_stream_metrics.csv"),
            "stream_summary": stream_summary,
        },
        output_dir / "dss_summary.json",
    )

    print("DSS evaluation complete.")
    print(f"Per-corruption sweep: {output_dir / 'dss_per_corruption_sweep.csv'}")
    print(f"Per-corruption best: {output_dir / 'dss_per_corruption_best.csv'}")
    print(f"Stream metrics: {output_dir / 'dss_stream_metrics.csv'}")


if __name__ == "__main__":
    main()
