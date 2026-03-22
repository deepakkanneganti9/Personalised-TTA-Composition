import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

torch.set_num_threads(1)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FL.train_fedavg_cifar10 import CIFAR10CNN, set_seed  # noqa: E402


DATA_ROOT = PROJECT_ROOT / "Data"


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def list_composition_dirs(artifacts_root: Path) -> List[Path]:
    composition_dirs = []
    for path in sorted(artifacts_root.iterdir()):
        if path.is_dir() and (path / "composition_model.pt").exists() and (path / "config.json").exists():
            composition_dirs.append(path)
    if not composition_dirs:
        raise FileNotFoundError(f"No composition artifact folders found in {artifacts_root}")
    return composition_dirs


def load_clean_cifar10_test(data_root: Path, sample_count: int) -> Subset:
    dataset = datasets.CIFAR10(
        root=str(data_root),
        train=False,
        download=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        ),
    )
    return Subset(dataset, list(range(min(sample_count, len(dataset)))))


def load_composition_model(weights_path: Path) -> CIFAR10CNN:
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model = CIFAR10CNN()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def evaluate_predictions(
    model: CIFAR10CNN,
    dataset: Subset,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []

    start = time.perf_counter()
    with torch.no_grad():
        for images, batch_labels in dataloader:
            logits = model(images)
            predictions.append(logits.argmax(dim=1).cpu())
            labels.append(batch_labels.cpu())
    latency = time.perf_counter() - start

    prediction_array = torch.cat(predictions).numpy()
    label_array = torch.cat(labels).numpy()
    return prediction_array, label_array, float(latency)


def compute_prediction_distribution(predictions: np.ndarray, num_classes: int = 10) -> Dict[str, List[float]]:
    counts = np.bincount(predictions, minlength=num_classes).astype(np.int64)
    distribution = (counts / max(1, counts.sum())).tolist()
    return {
        "counts": counts.tolist(),
        "probabilities": distribution,
    }


def compute_batch_accuracies(predictions: np.ndarray, labels: np.ndarray, num_batches: int = 10) -> List[float]:
    accuracies: List[float] = []
    for batch_indices in np.array_split(np.arange(len(labels)), num_batches):
        if len(batch_indices) == 0:
            accuracies.append(0.0)
            continue
        accuracies.append(float((predictions[batch_indices] == labels[batch_indices]).mean()))
    return accuracies


def stretch_score(value: float, floor: float) -> float:
    normalized = (value - floor) / max(1e-8, 1.0 - floor)
    normalized = float(np.clip(normalized, 0.0, 1.0))
    return 100.0 * (normalized ** 2)


def compute_quality_factor(batch_accuracies: Sequence[float]) -> float:
    accuracies = np.asarray(batch_accuracies, dtype=np.float64)
    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))
    min_acc = float(np.min(accuracies))
    base_quality = mean_acc * max(0.0, 1.0 - std_acc) * (0.75 + 0.25 * min_acc)
    return float(np.clip(stretch_score(base_quality, floor=0.40), 0.0, 100.0))


def compute_reliability_metrics(
    model: CIFAR10CNN,
    dataset: Subset,
    batch_size: int,
    reference_accuracy: float,
) -> Dict[str, object]:
    split_indices = np.array_split(np.arange(len(dataset)), 10)
    partial_sizes = [32, 64, 96, 128, 160, 192, 224, 256, 320, 384]
    partial_accuracies: List[float] = []

    for indices, partial_size in zip(split_indices, partial_sizes):
        selected = indices[: min(len(indices), partial_size)]
        if len(selected) == 0:
            partial_accuracies.append(0.0)
            continue
        subset = Subset(dataset, selected.tolist())
        predictions, labels, _ = evaluate_predictions(model, subset, batch_size=min(batch_size, len(subset)))
        partial_accuracies.append(float((predictions == labels).mean()))

    partial_array = np.asarray(partial_accuracies, dtype=np.float64)
    mean_partial = float(np.mean(partial_array))
    std_partial = float(np.std(partial_array))
    degradation = max(0.0, reference_accuracy - mean_partial)
    base_reliability = mean_partial * max(0.0, 1.0 - std_partial - 0.5 * degradation)
    return {
        "partial_batch_sizes": partial_sizes,
        "partial_accuracies": partial_accuracies,
        "variation_std": std_partial,
        "degradation": degradation,
        "reliability_score": float(np.clip(stretch_score(base_reliability, floor=0.35), 0.0, 100.0)),
    }


def copy_weights(source_path: Path, weights_root: Path, composition_name: str) -> Path:
    weights_root.mkdir(parents=True, exist_ok=True)
    target_path = weights_root / f"{composition_name}_composition_model.pt"
    shutil.copy2(source_path, target_path)
    return target_path


def build_row(
    composition_dir: Path,
    dataset: Subset,
    weights_root: Path,
    batch_size: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    summary = read_json(composition_dir / "summary.json")
    composition_name = summary["composition_name"]
    client_names = summary["client_names"]
    composition_size = len(client_names)

    model = load_composition_model(composition_dir / "composition_model.pt")
    predictions, labels, raw_latency = evaluate_predictions(model, dataset, batch_size=batch_size)

    prediction_distribution = compute_prediction_distribution(predictions)
    batch_accuracies = compute_batch_accuracies(predictions, labels, num_batches=10)
    raw_accuracy = float((predictions == labels).mean())
    quality_factor = compute_quality_factor(batch_accuracies)
    reliability = compute_reliability_metrics(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        reference_accuracy=raw_accuracy,
    )
    adjusted_latency = raw_latency * composition_size
    copied_weights_path = copy_weights(
        source_path=composition_dir / "composition_model.pt",
        weights_root=weights_root,
        composition_name=composition_name,
    )

    row = {
        "composition_id": composition_name,
        "client_names": json.dumps(client_names),
        "composition_size": composition_size,
        "prediction_distribution": json.dumps(prediction_distribution["probabilities"]),
        "latency": adjusted_latency,
        "quality_factor": quality_factor,
        "reliability_score": reliability["reliability_score"],
        "weights_file": str(copied_weights_path),
    }
    metadata = {
        "composition_id": composition_name,
        "artifact_dir": str(composition_dir),
        "client_names": client_names,
        "composition_size": composition_size,
        "prediction_distribution": prediction_distribution,
        "raw_latency_seconds": raw_latency,
        "adjusted_latency_seconds": adjusted_latency,
        "quality_batch_accuracies": batch_accuracies,
        "raw_accuracy": raw_accuracy,
        "quality_factor": quality_factor,
        "reliability": reliability,
        "weights_file": str(copied_weights_path),
    }
    return row, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-root",
        default=str(PROJECT_ROOT / "Performing composition" / "artifacts_expanded"),
    )
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "QOS_performing_C"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    artifacts_root = Path(args.artifacts_root).resolve()
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    weights_root = output_root / "weights"
    output_root.mkdir(parents=True, exist_ok=True)

    composition_dirs = list_composition_dirs(artifacts_root)
    dataset = load_clean_cifar10_test(data_root=data_root, sample_count=args.sample_count)

    rows: List[Dict[str, object]] = []
    metadata: Dict[str, object] = {
        "artifacts_root": str(artifacts_root),
        "data_root": str(data_root),
        "common_sample_count": len(dataset),
        "num_models": len(composition_dirs),
        "models": [],
    }

    for composition_dir in composition_dirs:
        row, model_metadata = build_row(
            composition_dir=composition_dir,
            dataset=dataset,
            weights_root=weights_root,
            batch_size=args.batch_size,
        )
        rows.append(row)
        metadata["models"].append(model_metadata)
        print(
            f"{row['composition_id']} | size={row['composition_size']} | "
            f"quality={row['quality_factor']:.2f} | reliability={row['reliability_score']:.2f}",
            flush=True,
        )

    csv_path = output_root / "qos_performing_c_dataset.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "composition_id",
            "client_names",
            "composition_size",
            "prediction_distribution",
            "latency",
            "quality_factor",
            "reliability_score",
            "weights_file",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_json(output_root / "weights_metadata.json", metadata)


if __name__ == "__main__":
    main()
