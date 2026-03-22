import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FL_C.train_fedavg_mnist import MNISTCNN, load_dataset_pair


FL_RUN_DIR = PROJECT_ROOT / "FL" / "artifacts" / "mnist_fl_baseline_5clients_run"
CORRUPT_ROOT = PROJECT_ROOT / "FL_C" / "artifacts" / "federated_artifact_corrupt"
MNIST_ROOT = PROJECT_ROOT / "data"
MNIST_C_ROOT = PROJECT_ROOT / "data" / "mnist_c" / "mnist_c"
DEFAULT_OUTPUT_ROOT = CORRUPT_ROOT / "client_profiles"

BASE_CATEGORIES = ["clean", "fog", "translate", "stripe", "zigzag", "spatter"]
FUNCTIONAL_LABEL_SPACE = "digits_0_9"
INPUT_MODALITY = "mnist_image"
INPUT_SHAPE = "1x28x28"
OUTPUT_TYPE = "digit_label"


class LocalMNISTC(Dataset):
    def __init__(self, root: Path, corruption: str, split: str = "test") -> None:
        corruption_dir = root / corruption
        self.images = np.load(corruption_dir / f"{split}_images.npy")
        self.labels = np.load(corruption_dir / f"{split}_labels.npy")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        image_np = self.images[index]
        if image_np.ndim == 3 and image_np.shape[-1] == 1:
            image_np = image_np[..., 0]
        image = torch.from_numpy(image_np).float().unsqueeze(0) / 255.0
        image = (image - 0.1307) / 0.3081
        return image, int(self.labels[index])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def make_clean_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def load_clean_test_dataset() -> datasets.MNIST:
    return datasets.MNIST(root=MNIST_ROOT, train=False, download=False, transform=make_clean_transform())


def load_corrupt_test_dataset(corruption: str) -> LocalMNISTC:
    return LocalMNISTC(root=MNIST_C_ROOT, corruption=corruption, split="test")


def maybe_cap_dataset(dataset: Dataset, max_eval_samples: int, seed: int) -> Dataset:
    if max_eval_samples <= 0 or len(dataset) <= max_eval_samples:
        return dataset
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(dataset), size=max_eval_samples, replace=False))
    return Subset(dataset, chosen.tolist())


def get_available_categories() -> List[str]:
    categories = ["clean"]
    preferred = [item for item in BASE_CATEGORIES if item != "clean"]
    for category in preferred:
        if (CORRUPT_ROOT / category).exists():
            categories.append(category)

    mixed_categories = []
    for path in sorted(CORRUPT_ROOT.iterdir()):
        if not path.is_dir():
            continue
        if path.name in {"client_profiles", "clean"}:
            continue
        config_path = path / "config.json"
        clients_dir = path / "clients"
        if not config_path.exists() or not clients_dir.exists():
            continue
        config = read_json(config_path)
        category = str(config.get("corruption", path.name))
        if category not in categories and category not in mixed_categories:
            mixed_categories.append(category)

    categories.extend(mixed_categories)
    return categories


def load_model(weights_path: Path) -> MNISTCNN:
    checkpoint = torch.load(weights_path, map_location="cpu")
    model = MNISTCNN()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def evaluate_predictions(model: MNISTCNN, dataset: Dataset, batch_size: int) -> Tuple[np.ndarray, np.ndarray, float]:
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
    probabilities = (counts / max(1, counts.sum())).tolist()
    return {
        "counts": counts.tolist(),
        "probabilities": probabilities,
    }


def compute_batch_accuracies(predictions: np.ndarray, labels: np.ndarray, num_batches: int = 10) -> List[float]:
    accuracies: List[float] = []
    for batch_indices in np.array_split(np.arange(len(labels)), num_batches):
        if len(batch_indices) == 0:
            accuracies.append(0.0)
            continue
        batch_accuracy = float((predictions[batch_indices] == labels[batch_indices]).mean())
        accuracies.append(batch_accuracy)
    return accuracies


def compute_reliability_metrics(model: MNISTCNN, dataset: Dataset, batch_size: int) -> Dict[str, object]:
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

    variation_std = float(np.std(partial_accuracies))
    reliability_score = float(np.clip(1.0 - variation_std, 0.0, 1.0))
    return {
        "partial_batch_sizes": partial_sizes,
        "partial_accuracies": partial_accuracies,
        "variation_std": variation_std,
        "reliability_score": reliability_score,
    }


def parse_checkpoint_name(file_name: str) -> Tuple[str, int, str]:
    stem = Path(file_name).stem
    if "_round_" in stem:
        client_name, round_part = stem.split("_round_")
        return client_name, int(round_part), "round_snapshot"
    return stem, -1, "final_snapshot"


def build_checkpoint_entries(clean_limit: int = 0, allowed_categories: List[str] = None) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    available_categories = get_available_categories()
    allowed = set(allowed_categories or [])

    clean_split_summary = read_json(FL_RUN_DIR / "split_summary.json")
    clean_config = read_json(FL_RUN_DIR / "config.json")
    clean_weights_dir = CORRUPT_ROOT / "clean"
    clean_entries: List[Dict[str, object]] = []
    for weights_path in sorted(clean_weights_dir.glob("*.pt")):
        client_name, round_index, checkpoint_kind = parse_checkpoint_name(weights_path.name)
        checkpoint = torch.load(weights_path, map_location="cpu")
        effective_round = int(checkpoint.get("round", round_index if round_index >= 0 else clean_config["num_rounds"]))
        clean_entries.append(
            {
                "category": "clean",
                "source_type": "clean_fl",
                "source_run_dir": str(FL_RUN_DIR),
                "source_checkpoint_path": str(weights_path),
                "source_checkpoint_name": weights_path.name,
                "client_name": client_name,
                "round_index": effective_round,
                "checkpoint_kind": checkpoint_kind,
                "train_metrics": checkpoint.get("train_metrics", {}),
                "split_summary": clean_split_summary.get(client_name, {}),
                "num_rounds": clean_config["num_rounds"],
            }
        )

    if clean_limit and clean_limit > 0 and len(clean_entries) > clean_limit:
        final_clean_entries = [entry for entry in clean_entries if entry["checkpoint_kind"] == "final_snapshot"]
        non_final_clean_entries = [entry for entry in clean_entries if entry["checkpoint_kind"] != "final_snapshot"]
        selected_clean_entries = list(final_clean_entries)
        remaining_slots = max(0, clean_limit - len(selected_clean_entries))
        selected_clean_entries.extend(non_final_clean_entries[:remaining_slots])
        clean_entries = selected_clean_entries

    if not allowed or "clean" in allowed:
        entries.extend(clean_entries)

    for corruption in available_categories:
        if corruption == "clean":
            continue
        if allowed and corruption not in allowed:
            continue
        run_dir = CORRUPT_ROOT / corruption.replace("+", "_")
        if not run_dir.exists():
            continue
        split_summary = read_json(run_dir / "split_summary.json")
        run_config = read_json(run_dir / "config.json")
        for weights_path in sorted((run_dir / "clients").glob("*.pt")):
            client_name, round_index, checkpoint_kind = parse_checkpoint_name(weights_path.name)
            checkpoint = torch.load(weights_path, map_location="cpu")
            effective_round = int(checkpoint.get("round", round_index if round_index >= 0 else run_config["num_rounds"]))
            entries.append(
                {
                    "category": corruption,
                    "source_type": "corrupt_fl",
                    "source_run_dir": str(run_dir),
                    "source_checkpoint_path": str(weights_path),
                    "source_checkpoint_name": weights_path.name,
                    "client_name": client_name,
                    "round_index": effective_round,
                    "checkpoint_kind": checkpoint_kind,
                    "train_metrics": checkpoint.get("train_metrics", {}),
                    "split_summary": split_summary.get(client_name, {}),
                    "num_rounds": run_config["num_rounds"],
                }
            )

    return entries


def assign_aliases(entries: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    category_order = get_available_categories()
    counters = defaultdict(int)
    updated_entries: List[Dict[str, object]] = []
    for entry in sorted(
        entries,
        key=lambda item: (
            category_order.index(item["category"]) if item["category"] in category_order else len(category_order),
            item["client_name"],
            item["round_index"],
            item["source_checkpoint_name"],
        ),
    ):
        counters[entry["category"]] += 1
        entry = dict(entry)
        entry["service_id"] = f"{entry['category']}{counters[entry['category']]}"
        updated_entries.append(entry)
    return updated_entries


def copy_weight(source_path: Path, target_root: Path, service_id: str) -> Path:
    target_root.mkdir(parents=True, exist_ok=True)
    target_path = target_root / f"{service_id}.pt"
    shutil.copy2(source_path, target_path)
    return target_path


def build_row(
    entry: Dict[str, object],
    output_weights_root: Path,
    batch_size: int,
    max_eval_samples: int,
    seed: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    profile_start = time.perf_counter()
    category = str(entry["category"])
    weights_path = Path(str(entry["source_checkpoint_path"]))
    model = load_model(weights_path)
    if category == "clean":
        dataset = load_clean_test_dataset()
    elif "+" in category:
        _, dataset, _, _, _ = load_dataset_pair(category)
    else:
        dataset = load_corrupt_test_dataset(category)
    dataset = maybe_cap_dataset(dataset, max_eval_samples=max_eval_samples, seed=seed + int(entry["round_index"]) + len(category))
    trial_eval_start = time.perf_counter()
    predictions, labels, latency = evaluate_predictions(model, dataset, batch_size=batch_size)
    trial_eval_seconds = time.perf_counter() - trial_eval_start
    prediction_distribution = compute_prediction_distribution(predictions)
    batch_accuracies = compute_batch_accuracies(predictions, labels)
    reliability_start = time.perf_counter()
    reliability = compute_reliability_metrics(model, dataset, batch_size=batch_size)
    reliability_profile_seconds = time.perf_counter() - reliability_start
    accuracy = float((predictions == labels).mean())
    quality_factor = float(np.mean(batch_accuracies))

    copied_weights_path = copy_weight(weights_path, output_weights_root, str(entry["service_id"]))
    split_summary = dict(entry["split_summary"])
    class_counts = split_summary.get("class_counts", {})

    functional_profile = {
        "task_type": "digit_classification",
        "input_modality": INPUT_MODALITY,
        "input_shape": INPUT_SHAPE,
        "output_type": OUTPUT_TYPE,
        "label_space": FUNCTIONAL_LABEL_SPACE,
        "label_count": 10,
        "data_domain": "clean_mnist" if category == "clean" else f"mnist_c_{category}",
        "corruption_category": category,
    }

    row = {
        "service_id": entry["service_id"],
        "service_alias": entry["service_id"],
        "source_type": entry["source_type"],
        "source_run_dir": entry["source_run_dir"],
        "source_checkpoint_name": entry["source_checkpoint_name"],
        "source_checkpoint_path": entry["source_checkpoint_path"],
        "client_name": entry["client_name"],
        "round_index": entry["round_index"],
        "checkpoint_kind": entry["checkpoint_kind"],
        "category": category,
        "original_client_id": entry["client_name"],
        "functional_task_type": functional_profile["task_type"],
        "functional_input_modality": functional_profile["input_modality"],
        "functional_input_shape": functional_profile["input_shape"],
        "functional_output_type": functional_profile["output_type"],
        "functional_label_space": functional_profile["label_space"],
        "functional_label_count": functional_profile["label_count"],
        "functional_data_domain": functional_profile["data_domain"],
        "functional_corruption_category": functional_profile["corruption_category"],
        "functional_num_samples": split_summary.get("num_samples", entry["train_metrics"].get("num_samples", 0)),
        "functional_class_distribution": json.dumps(class_counts, sort_keys=True),
        "train_loss": entry["train_metrics"].get("loss"),
        "evaluation_split": "test",
        "evaluation_accuracy": accuracy,
        "latency_seconds": latency,
        "trial_evaluation_time_seconds": trial_eval_seconds,
        "reliability_profile_time_seconds": reliability_profile_seconds,
        "profile_build_time_seconds": time.perf_counter() - profile_start,
        "quality_factor": quality_factor,
        "reliability_score": reliability["reliability_score"],
        "prediction_distribution": json.dumps(prediction_distribution["probabilities"]),
        "weights_file": str(copied_weights_path),
    }
    metadata = {
        "service_id": entry["service_id"],
        "category": category,
        "source_checkpoint_name": entry["source_checkpoint_name"],
        "source_checkpoint_path": entry["source_checkpoint_path"],
        "copied_weights_path": str(copied_weights_path),
        "functional_profile": functional_profile,
        "train_metrics": entry["train_metrics"],
        "split_summary": split_summary,
        "evaluation_accuracy": accuracy,
        "latency_seconds": latency,
        "trial_evaluation_time_seconds": trial_eval_seconds,
        "reliability_profile_time_seconds": reliability_profile_seconds,
        "profile_build_time_seconds": time.perf_counter() - profile_start,
        "quality_batch_accuracies": batch_accuracies,
        "quality_factor": quality_factor,
        "prediction_distribution": prediction_distribution,
        "reliability": reliability,
    }
    return row, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-limit", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--categories", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(1)

    output_root = Path(args.output_root).resolve()
    weights_root = output_root / "weights"
    output_root.mkdir(parents=True, exist_ok=True)
    build_start = time.perf_counter()

    allowed_categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    entries = assign_aliases(build_checkpoint_entries(clean_limit=args.clean_limit, allowed_categories=allowed_categories))
    rows: List[Dict[str, object]] = []
    metadata_models: List[Dict[str, object]] = []

    for index, entry in enumerate(entries, start=1):
        row, metadata = build_row(
            entry,
            weights_root,
            batch_size=args.batch_size,
            max_eval_samples=args.max_eval_samples,
            seed=args.seed,
        )
        rows.append(row)
        metadata_models.append(metadata)
        print(
            f"[{index}/{len(entries)}] {row['service_id']} <- {row['source_checkpoint_name']} | "
            f"acc={row['evaluation_accuracy']:.4f} | quality={row['quality_factor']:.4f} | "
            f"reliability={row['reliability_score']:.4f}",
            flush=True,
        )

    csv_path = output_root / "client_service_profiles.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "service_id",
            "service_alias",
            "source_type",
            "source_run_dir",
            "source_checkpoint_name",
            "source_checkpoint_path",
            "client_name",
            "round_index",
            "checkpoint_kind",
            "category",
            "original_client_id",
            "functional_task_type",
            "functional_input_modality",
            "functional_input_shape",
            "functional_output_type",
            "functional_label_space",
            "functional_label_count",
            "functional_data_domain",
            "functional_corruption_category",
            "functional_num_samples",
            "functional_class_distribution",
            "train_loss",
            "evaluation_split",
            "evaluation_accuracy",
            "latency_seconds",
            "trial_evaluation_time_seconds",
            "reliability_profile_time_seconds",
            "profile_build_time_seconds",
            "quality_factor",
            "reliability_score",
            "prediction_distribution",
            "weights_file",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_json(
        output_root / "weights_metadata.json",
        {
            "output_root": str(output_root),
            "weights_root": str(weights_root),
            "num_services": len(rows),
            "clean_limit": args.clean_limit,
            "max_eval_samples": args.max_eval_samples,
            "allowed_categories": allowed_categories,
            "runtime_seconds": time.perf_counter() - build_start,
            "categories": BASE_CATEGORIES,
            "available_categories": get_available_categories(),
            "models": metadata_models,
        },
    )


if __name__ == "__main__":
    main()
