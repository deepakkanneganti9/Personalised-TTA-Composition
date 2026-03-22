import argparse
import json
import os
import random
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "Data"
ARTIFACT_ROOT = PROJECT_ROOT / "FL" / "artifacts"

CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR100_CLASS_NAMES = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "computer_keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]

DATASET_SPECS = {
    "cifar10": {
        "display_name": "CIFAR-10",
        "torchvision_class": datasets.CIFAR10,
        "num_classes": 10,
        "class_names": CIFAR10_CLASS_NAMES,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "corruption_dir": "CIFAR-10-C",
        "default_run_name": "cifar10_fl_run",
        "default_preferred_classes_per_client": 4,
    },
    "cifar100": {
        "display_name": "CIFAR-100",
        "torchvision_class": datasets.CIFAR100,
        "num_classes": 100,
        "class_names": CIFAR100_CLASS_NAMES,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "corruption_dir": "CIFAR-100-C",
        "default_run_name": "cifar100_fl_run",
        "default_preferred_classes_per_client": 25,
    },
}

CLASS_NAMES = CIFAR100_CLASS_NAMES

DEFAULT_CONFIG = {
    "dataset": "cifar100",
    "seed": 42,
    "num_clients": 5,
    "num_rounds": 10,
    "local_epochs": 1,
    "batch_size": 256,
    "eval_batch_size": 256,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "num_threads": min(4, os.cpu_count() or 1),
    "preferred_classes_per_client": 25,
    "preferred_share": 0.68,
    "max_train_samples_per_client": None,
    "max_eval_samples": None,
}


def normalize_dataset_name(dataset_name: Optional[str]) -> str:
    if dataset_name is None:
        return DEFAULT_CONFIG["dataset"]
    normalized = dataset_name.lower().replace("-", "").replace("_", "")
    if normalized == "cifar10":
        return "cifar10"
    if normalized == "cifar100":
        return "cifar100"
    raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected one of: cifar10, cifar100.")


def get_dataset_spec(dataset_name: Optional[str] = None) -> Dict[str, object]:
    return DATASET_SPECS[normalize_dataset_name(dataset_name)]


def dataset_display_name(dataset_name: Optional[str] = None) -> str:
    return str(get_dataset_spec(dataset_name)["display_name"])


def build_model_for_dataset(dataset_name: Optional[str] = None) -> nn.Module:
    spec = get_dataset_spec(dataset_name)
    return CIFARCNN(num_classes=int(spec["num_classes"]))


def infer_dataset_name_from_checkpoint(checkpoint: Dict[str, object]) -> str:
    dataset_name = checkpoint.get("dataset_name")
    if dataset_name:
        return normalize_dataset_name(str(dataset_name))

    dataset_label = str(checkpoint.get("dataset", "")).lower()
    if "100" in dataset_label:
        return "cifar100"
    if "10" in dataset_label:
        return "cifar10"
    return DEFAULT_CONFIG["dataset"]


def load_run_config(run_dir: Path) -> Dict[str, object]:
    return load_json(run_dir / "config.json")


def load_run_dataset_name(run_dir: Path) -> str:
    config = load_run_config(run_dir)
    return normalize_dataset_name(str(config.get("dataset", DEFAULT_CONFIG["dataset"])))


class CIFARCNN(nn.Module):
    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
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
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)


CIFAR100CNN = CIFARCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def make_transform(dataset_name: Optional[str] = None):
    spec = get_dataset_spec(dataset_name)
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(spec["mean"], spec["std"]),
        ]
    )


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def state_dict_to_cpu(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def apply_config_overrides(config, args) -> dict:
    updated = deepcopy(config)
    override_fields = [
        "dataset",
        "seed",
        "num_clients",
        "num_rounds",
        "local_epochs",
        "batch_size",
        "eval_batch_size",
        "learning_rate",
        "weight_decay",
        "num_workers",
        "num_threads",
        "preferred_classes_per_client",
        "preferred_share",
        "max_train_samples_per_client",
        "max_eval_samples",
    ]
    for field_name in override_fields:
        value = getattr(args, field_name, None)
        if value is not None:
            updated[field_name] = value
    updated["dataset"] = normalize_dataset_name(updated["dataset"])
    return updated


def apply_dataset_defaults(config: Dict[str, object]) -> Dict[str, object]:
    updated = deepcopy(config)
    dataset_name = normalize_dataset_name(str(updated.get("dataset", DEFAULT_CONFIG["dataset"])))
    spec = get_dataset_spec(dataset_name)
    if updated.get("preferred_classes_per_client") in (None, 4, 25):
        updated["preferred_classes_per_client"] = int(spec["default_preferred_classes_per_client"])
    updated["dataset"] = dataset_name
    return updated


def build_random_client_preferences(
    num_clients: int,
    preferred_classes_per_client: int,
    num_classes: int,
    seed: int,
):
    if preferred_classes_per_client > num_classes:
        raise ValueError(
            f"preferred_classes_per_client={preferred_classes_per_client} exceeds num_classes={num_classes}."
        )
    if preferred_classes_per_client * num_clients < num_classes:
        raise ValueError(
            "preferred_classes_per_client is too small to cover all classes across clients. "
            f"Need at least ceil({num_classes}/{num_clients})={int(np.ceil(num_classes / num_clients))}."
        )

    rng = np.random.default_rng(seed)
    preferences = {f"client_{client_id}": set() for client_id in range(num_clients)}

    class_ids = list(range(num_classes))
    rng.shuffle(class_ids)
    for offset, class_id in enumerate(class_ids):
        preferences[f"client_{offset % num_clients}"].add(class_id)

    all_classes = set(range(num_classes))
    for client_name, assigned in preferences.items():
        remaining_needed = preferred_classes_per_client - len(assigned)
        if remaining_needed <= 0:
            continue
        available = sorted(all_classes - assigned)
        chosen = rng.choice(available, size=remaining_needed, replace=False).tolist()
        assigned.update(int(class_id) for class_id in chosen)

    return {
        client_name: sorted(int(class_id) for class_id in assigned_classes)
        for client_name, assigned_classes in preferences.items()
    }


def build_random_mild_noniid_split(labels, config, num_classes: int):
    rng = np.random.default_rng(config["seed"])
    labels = np.asarray(labels)
    num_clients = int(config["num_clients"])
    preferences = build_random_client_preferences(
        num_clients=num_clients,
        preferred_classes_per_client=int(config["preferred_classes_per_client"]),
        num_classes=num_classes,
        seed=int(config["seed"]),
    )
    client_indices = {f"client_{client_id}": [] for client_id in range(num_clients)}

    for class_id in range(num_classes):
        indices = np.where(labels == class_id)[0]
        rng.shuffle(indices)
        indices = indices.tolist()

        preferred_clients = [
            client_name for client_name, preferred_classes in preferences.items() if class_id in preferred_classes
        ]
        secondary_clients = [client_name for client_name in client_indices if client_name not in preferred_clients]

        if not preferred_clients or not secondary_clients:
            midpoint = max(1, len(client_indices) // 2)
            ordered_clients = list(client_indices.keys())
            preferred_clients = ordered_clients[:midpoint]
            secondary_clients = ordered_clients[midpoint:]

        preferred_total = int(round(len(indices) * float(config["preferred_share"])))
        preferred_total = min(max(preferred_total, len(preferred_clients)), len(indices) - len(secondary_clients))
        secondary_total = len(indices) - preferred_total

        allocation = {client_name: 0 for client_name in client_indices}
        share, remainder = divmod(preferred_total, len(preferred_clients))
        for offset, client_name in enumerate(preferred_clients):
            allocation[client_name] = share + (1 if offset < remainder else 0)

        share, remainder = divmod(secondary_total, len(secondary_clients))
        for offset, client_name in enumerate(secondary_clients):
            allocation[client_name] = share + (1 if offset < remainder else 0)

        cursor = 0
        for client_name in client_indices:
            take = allocation[client_name]
            client_indices[client_name].extend(indices[cursor:cursor + take])
            cursor += take

    for client_name in client_indices:
        rng.shuffle(client_indices[client_name])

    return preferences, client_indices


def summarize_split(labels, client_indices, class_names: List[str]):
    summary = {}
    for client_name, indices in client_indices.items():
        counts = Counter(int(labels[index]) for index in indices)
        summary[client_name] = {
            "num_samples": len(indices),
            "class_counts": {str(class_id): counts.get(class_id, 0) for class_id in range(len(class_names))},
            "class_names": {str(class_id): class_names[class_id] for class_id in range(len(class_names))},
        }
    return summary


def get_run_dir(run_dir_arg: Optional[str], dataset_name: Optional[str] = None) -> Path:
    if run_dir_arg:
        return Path(run_dir_arg).resolve()
    spec = get_dataset_spec(dataset_name)
    return (ARTIFACT_ROOT / f"{spec['default_run_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()


def load_datasets(dataset_name: Optional[str] = None):
    spec = get_dataset_spec(dataset_name)
    transform = make_transform(dataset_name)
    train_dataset = spec["torchvision_class"](root=DATA_ROOT, train=True, download=False, transform=transform)
    test_dataset = spec["torchvision_class"](root=DATA_ROOT, train=False, download=False, transform=transform)
    return train_dataset, test_dataset


def aggregate_state_dicts(local_states, client_sizes):
    total_samples = sum(client_sizes.values())
    client_names = list(local_states.keys())
    aggregated_state = {}

    for param_name in local_states[client_names[0]].keys():
        tensors = [local_states[client_name][param_name] for client_name in client_names]
        sample_tensor = tensors[0]

        if sample_tensor.is_floating_point():
            accumulator = torch.zeros_like(sample_tensor, dtype=torch.float32)
            for client_name in client_names:
                weight = client_sizes[client_name] / total_samples
                accumulator += local_states[client_name][param_name].to(torch.float32) * weight
            aggregated_state[param_name] = accumulator.to(sample_tensor.dtype)
            continue

        accumulator = torch.zeros_like(sample_tensor, dtype=torch.float64)
        for client_name in client_names:
            weight = client_sizes[client_name] / total_samples
            accumulator += local_states[client_name][param_name].to(torch.float64) * weight
        aggregated_state[param_name] = accumulator.round().to(sample_tensor.dtype)

    return aggregated_state


def maybe_limit_indices(indices: List[int], max_samples: Optional[int]) -> List[int]:
    if max_samples is None:
        return list(indices)
    return list(indices[: max(0, int(max_samples))])


def init_run(run_dir: Path, config_overrides: Optional[dict] = None):
    config = deepcopy(DEFAULT_CONFIG)
    if config_overrides:
        config.update(config_overrides)
    config = apply_dataset_defaults(config)

    dataset_name = normalize_dataset_name(str(config["dataset"]))
    spec = get_dataset_spec(dataset_name)
    class_names = list(spec["class_names"])

    set_seed(int(config["seed"]))
    torch.set_num_threads(int(config["num_threads"]))

    train_dataset, _ = load_datasets(dataset_name)
    labels = np.array(train_dataset.targets)
    preferences, client_indices = build_random_mild_noniid_split(labels, config, num_classes=int(spec["num_classes"]))

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "rounds").mkdir(exist_ok=True)
    (run_dir / "clients").mkdir(exist_ok=True)
    (run_dir / "global").mkdir(exist_ok=True)

    save_json(run_dir / "config.json", config)
    save_json(run_dir / "client_preferences.json", preferences)
    save_json(run_dir / "client_indices.json", client_indices)
    save_json(run_dir / "split_summary.json", summarize_split(labels, client_indices, class_names))
    save_json(run_dir / "metrics.json", [])

    model = build_model_for_dataset(dataset_name)
    torch.save(
        {
            "model_state": state_dict_to_cpu(model.state_dict()),
            "round": 0,
            "dataset": spec["display_name"],
            "dataset_name": dataset_name,
        },
        run_dir / "global" / "global_model_round_0.pt",
    )
    print(str(run_dir), flush=True)


def run_round(run_dir: Path, round_index: int, device_name: str = "cpu"):
    config = load_json(run_dir / "config.json")
    config = apply_dataset_defaults(config)
    client_indices = load_json(run_dir / "client_indices.json")

    dataset_name = normalize_dataset_name(str(config["dataset"]))
    spec = get_dataset_spec(dataset_name)

    set_seed(int(config["seed"]) + round_index)
    torch.set_num_threads(int(config["num_threads"]))

    device = get_device(device_name)
    train_dataset, test_dataset = load_datasets(dataset_name)
    previous = torch.load(run_dir / "global" / f"global_model_round_{round_index - 1}.pt", map_location="cpu")
    global_model = build_model_for_dataset(dataset_name).to(device)
    global_model.load_state_dict(previous["model_state"])

    criterion = nn.CrossEntropyLoss()
    local_states = {}
    client_metrics = {}
    client_sizes = {}

    for client_name, indices in client_indices.items():
        effective_indices = maybe_limit_indices(indices, config.get("max_train_samples_per_client"))
        local_model = deepcopy(global_model)
        local_model.train()
        optimizer = torch.optim.Adam(
            local_model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        loader = DataLoader(
            Subset(train_dataset, effective_indices),
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=int(config["num_workers"]),
        )

        total_loss = 0.0
        total_batches = 0
        for _ in range(int(config["local_epochs"])):
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = local_model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_batches += 1

        local_states[client_name] = state_dict_to_cpu(local_model.state_dict())
        client_sizes[client_name] = len(effective_indices)
        client_metrics[client_name] = {
            "loss": total_loss / total_batches if total_batches else 0.0,
            "num_samples": len(effective_indices),
            "original_num_samples": len(indices),
        }

    aggregated_state = aggregate_state_dicts(local_states, client_sizes)
    global_model.load_state_dict(aggregated_state)
    global_model.eval()
    eval_dataset = test_dataset
    if config.get("max_eval_samples") is not None:
        eval_dataset = Subset(test_dataset, list(range(min(len(test_dataset), int(config["max_eval_samples"])))))
    test_loader = DataLoader(
        eval_dataset,
        batch_size=int(config["eval_batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = global_model(images)
            predictions = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = correct / total if total else 0.0
    save_json(
        run_dir / "rounds" / f"round_{round_index:02d}.json",
        {
            "round": round_index,
            "dataset": spec["display_name"],
            "global_accuracy": accuracy,
            "client_metrics": client_metrics,
            "client_sizes": client_sizes,
        },
    )
    torch.save(
        {
            "model_state": aggregated_state,
            "round": round_index,
            "accuracy": accuracy,
            "dataset": spec["display_name"],
            "dataset_name": dataset_name,
        },
        run_dir / "global" / f"global_model_round_{round_index}.pt",
    )
    for client_name, state in local_states.items():
        torch.save(
            {
                "client_name": client_name,
                "round": round_index,
                "model_state": state,
                "train_metrics": client_metrics[client_name],
                "dataset": spec["display_name"],
                "dataset_name": dataset_name,
            },
            run_dir / "clients" / f"{client_name}_round_{round_index}.pt",
        )

    metrics = load_json(run_dir / "metrics.json")
    metrics.append(
        {
            "round": round_index,
            "dataset": spec["display_name"],
            "global_accuracy": accuracy,
            "client_metrics": client_metrics,
        }
    )
    save_json(run_dir / "metrics.json", metrics)
    print(f"round={round_index} accuracy={accuracy:.4f} device={device} dataset={spec['display_name']}", flush=True)


def finalize_run(run_dir: Path):
    config = apply_dataset_defaults(load_json(run_dir / "config.json"))
    dataset_name = normalize_dataset_name(str(config["dataset"]))
    spec = get_dataset_spec(dataset_name)
    final_round = int(config["num_rounds"])
    latest_global = torch.load(run_dir / "global" / f"global_model_round_{final_round}.pt", map_location="cpu")
    torch.save(latest_global, run_dir / "global" / "global_model.pt")

    for client_id in range(int(config["num_clients"])):
        client_name = f"client_{client_id}"
        latest_client = torch.load(run_dir / "clients" / f"{client_name}_round_{final_round}.pt", map_location="cpu")
        torch.save(latest_client, run_dir / "clients" / f"{client_name}.pt")

    summary = {
        "run_dir": str(run_dir),
        "dataset": spec["display_name"],
        "dataset_name": dataset_name,
        "num_rounds": final_round,
        "final_global_accuracy": latest_global["accuracy"],
    }
    save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def run_full(run_dir: Path, device_name: str = "cpu", config_overrides: Optional[dict] = None):
    init_run(run_dir, config_overrides=config_overrides)
    config = apply_dataset_defaults(load_json(run_dir / "config.json"))
    for round_index in range(1, int(config["num_rounds"]) + 1):
        run_round(run_dir, round_index, device_name=device_name)
    finalize_run(run_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["init", "round", "finalize", "full"], required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--round-index", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-clients", type=int)
    parser.add_argument("--num-rounds", type=int)
    parser.add_argument("--local-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--preferred-classes-per-client", type=int)
    parser.add_argument("--preferred-share", type=float)
    parser.add_argument("--max-train-samples-per-client", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    args = parser.parse_args()

    config_overrides = apply_config_overrides(DEFAULT_CONFIG, args)
    run_dir = get_run_dir(args.run_dir, dataset_name=config_overrides["dataset"])

    if args.mode == "init":
        init_run(run_dir, config_overrides=config_overrides)
        return

    if args.mode == "round":
        if args.round_index is None:
            raise ValueError("--round-index is required for --mode round")
        run_round(run_dir, args.round_index, device_name=args.device)
        return

    if args.mode == "finalize":
        finalize_run(run_dir)
        return

    run_full(run_dir, device_name=args.device, config_overrides=config_overrides)


if __name__ == "__main__":
    main()
