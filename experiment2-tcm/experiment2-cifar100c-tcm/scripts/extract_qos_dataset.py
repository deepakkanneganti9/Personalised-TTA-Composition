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

torch.set_num_threads(1)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TTA_ROOT = PROJECT_ROOT / "TTA techniques"
if str(TTA_ROOT) not in sys.path:
    sys.path.insert(0, str(TTA_ROOT))

from tta_techniques.tent_adapter import LocalCIFARCorruption, set_seed  # noqa: E402
from FL.train_fedavg_cifar100 import build_model_for_dataset, get_dataset_spec  # noqa: E402


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def list_artifact_dirs(artifacts_root: Path) -> List[Path]:
    artifact_dirs = []
    for path in sorted(artifacts_root.iterdir()):
        if path.is_dir() and (path / "adapted_client_model.pt").exists() and (path / "config.json").exists():
            artifact_dirs.append(path)
    if not artifact_dirs:
        raise FileNotFoundError(f"No adapted artifact folders found in {artifacts_root}")
    return artifact_dirs


def build_common_sample_count(artifact_dirs: Sequence[Path], cifar_c_root: Path) -> int:
    counts = []
    for artifact_dir in artifact_dirs:
        config = read_json(artifact_dir / "config.json")
        corruption = config["corruption"]
        dataset_name = config.get("dataset_name", "cifar100")
        split = config.get("split", "test")
        dataset = LocalCIFARCorruption(
            root=cifar_c_root,
            corruption=corruption,
            dataset_name=dataset_name,
            split=split,
            severity=int(config.get("severity", 1)),
        )
        counts.append(len(dataset))
    return int(min(counts))


def load_adapted_model(weights_path: Path, dataset_name: str):
    checkpoint = torch.load(weights_path, map_location="cpu")
    if "model_state" not in checkpoint:
        raise KeyError(f"Expected 'model_state' key in {weights_path}")
    state_dict = checkpoint["model_state"]
    model = build_model_for_dataset(dataset_name)
    bn_running_keys = [name for name in state_dict.keys() if "running_mean" in name or "running_var" in name]
    if not bn_running_keys:
        for module in model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def evaluate_predictions(
    model,
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


def compute_prediction_distribution(predictions: np.ndarray, num_classes: int) -> Dict[str, List[float]]:
    counts = np.bincount(predictions, minlength=num_classes).astype(np.int64)
    distribution = (counts / max(1, counts.sum())).tolist()
    return {
        "counts": counts.tolist(),
        "probabilities": distribution,
    }


def compute_batch_accuracies(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_batches: int = 10,
) -> List[float]:
    accuracies: List[float] = []
    for batch_indices in np.array_split(np.arange(len(labels)), num_batches):
        if len(batch_indices) == 0:
            accuracies.append(0.0)
            continue
        batch_accuracy = float((predictions[batch_indices] == labels[batch_indices]).mean())
        accuracies.append(batch_accuracy)
    return accuracies


def compute_reliability_metrics(
    model,
    dataset: Subset,
    batch_size: int,
) -> Dict[str, object]:
    split_indices = np.array_split(np.arange(len(dataset)), 10)
    partial_sizes = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88]
    partial_accuracies: List[float] = []

    for indices, partial_size in zip(split_indices, partial_sizes):
        selected = indices[: min(len(indices), partial_size)]
        if len(selected) == 0:
            partial_accuracies.append(0.0)
            continue
        subset = Subset(dataset, selected.tolist())
        predictions, labels, _ = evaluate_predictions(model, subset, batch_size=min(batch_size, len(subset)))
        partial_accuracies.append(float((predictions == labels).mean()))

    reliability_variation = float(np.std(partial_accuracies))
    reliability_score = float(np.clip(1.0 - reliability_variation, 0.0, 1.0))
    return {
        "partial_batch_sizes": partial_sizes,
        "partial_accuracies": partial_accuracies,
        "variation_std": reliability_variation,
        "reliability_score": reliability_score,
    }


def copy_weights(source_path: Path, weights_root: Path, client_id: str, corruption: str) -> Path:
    weights_root.mkdir(parents=True, exist_ok=True)
    target_name = f"{client_id}_{corruption}_adapted_client_model.pt"
    target_path = weights_root / target_name
    shutil.copy2(source_path, target_path)
    return target_path


def build_row(
    artifact_dir: Path,
    cifar_c_root: Path,
    common_sample_count: int,
    weights_root: Path,
    batch_size: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    config = read_json(artifact_dir / "config.json")
    client_id = config["client_name"]
    corruption = config["corruption"]
    dataset_name = config.get("dataset_name", "cifar100")
    split = config.get("split", "test")
    severity = int(config.get("severity", 1))
    dataset_spec = get_dataset_spec(dataset_name)

    dataset = LocalCIFARCorruption(
        root=cifar_c_root,
        corruption=corruption,
        dataset_name=dataset_name,
        split=split,
        severity=severity,
    )
    dataset = Subset(dataset, list(range(common_sample_count)))

    model = load_adapted_model(artifact_dir / "adapted_client_model.pt", dataset_name=dataset_name)
    predictions, labels, latency = evaluate_predictions(model, dataset, batch_size=batch_size)

    prediction_distribution = compute_prediction_distribution(
        predictions,
        num_classes=int(dataset_spec["num_classes"]),
    )
    batch_accuracies = compute_batch_accuracies(predictions, labels, num_batches=10)
    quality_factor = float(np.mean(batch_accuracies))
    reliability = compute_reliability_metrics(model, dataset, batch_size=batch_size)
    copied_weights_path = copy_weights(
        source_path=artifact_dir / "adapted_client_model.pt",
        weights_root=weights_root,
        client_id=client_id,
        corruption=corruption,
    )

    row = {
        "client_id": client_id,
        "corruption_category": corruption,
        "prediction_distribution": json.dumps(prediction_distribution["probabilities"]),
        "latency": latency,
        "quality_factor": quality_factor,
        "reliability_score": reliability["reliability_score"],
        "weights_file": str(copied_weights_path),
    }
    metadata = {
        "artifact_dir": str(artifact_dir),
        "client_id": client_id,
        "corruption_category": corruption,
        "dataset_name": dataset_name,
        "common_sample_count": common_sample_count,
        "prediction_distribution": prediction_distribution,
        "latency_seconds": latency,
        "quality_batch_accuracies": batch_accuracies,
        "quality_factor": quality_factor,
        "reliability": reliability,
        "weights_file": str(copied_weights_path),
    }
    return row, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", default=str(TTA_ROOT / "artifacts"))
    parser.add_argument("--cifar-c-root", default=str(PROJECT_ROOT / "Data" / "CIFAR-100-C"))
    parser.add_argument("--cifar100-c-root", dest="cifar_c_root")
    parser.add_argument("--summary-source", default=str(PROJECT_ROOT / "composability-results-chunked" / "summary.csv"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "QS"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    artifacts_root = Path(args.artifacts_root).resolve()
    cifar_c_root = Path(args.cifar_c_root).resolve()
    summary_source = Path(args.summary_source).resolve()
    output_root = Path(args.output_root).resolve()
    weights_root = output_root / "weights"
    output_root.mkdir(parents=True, exist_ok=True)

    artifact_dirs = list_artifact_dirs(artifacts_root)
    common_sample_count = build_common_sample_count(artifact_dirs, cifar_c_root)

    rows: List[Dict[str, object]] = []
    metadata: Dict[str, object] = {
        "artifacts_root": str(artifacts_root),
        "cifar_c_root": str(cifar_c_root),
        "summary_source": str(summary_source),
        "common_sample_count": common_sample_count,
        "num_models": len(artifact_dirs),
        "models": [],
    }

    for artifact_dir in artifact_dirs:
        row, model_metadata = build_row(
            artifact_dir=artifact_dir,
            cifar_c_root=cifar_c_root,
            common_sample_count=common_sample_count,
            weights_root=weights_root,
            batch_size=args.batch_size,
        )
        rows.append(row)
        metadata["models"].append(model_metadata)
        print(
            f"{row['client_id']} | {row['corruption_category']} | "
            f"quality={row['quality_factor']:.4f} | reliability={row['reliability_score']:.4f}",
            flush=True,
        )

    csv_path = output_root / "qos_dataset.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "client_id",
            "corruption_category",
            "prediction_distribution",
            "latency",
            "quality_factor",
            "reliability_score",
            "weights_file",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if summary_source.exists():
        shutil.copy2(summary_source, output_root / "summary.csv")

    write_json(output_root / "weights_metadata.json", metadata)


if __name__ == "__main__":
    main()
