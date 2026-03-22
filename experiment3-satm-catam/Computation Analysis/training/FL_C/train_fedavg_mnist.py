import argparse
import json
import os
import random
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.datasets import MNIST


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
MNIST_ROOT = DATA_ROOT
MNIST_C_ROOT = DATA_ROOT / "mnist_c" / "mnist_c"
ARTIFACT_ROOT = PROJECT_ROOT / "FL_C" / "artifacts" / "federated_artifact_corrupt"

DEFAULT_CORRUPTIONS = [
    "zigzag",
    "stripe",
    "impulse_noise",
    "glass_blur",
    "brightness",
    "translate",
    "spatter",
    "fog",
]

DEFAULT_CONFIG = {
    "seed": 42,
    "num_clients": 10,
    "num_rounds": 5,
    "local_epochs": 2,
    "batch_size": 128,
    "eval_batch_size": 256,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "num_threads": 1,
    "preferred_digits_per_client": 4,
    "preferred_share": 0.68,
    "corruptions": DEFAULT_CORRUPTIONS,
}


class MNISTCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)


class LocalMNISTC(Dataset):
    def __init__(self, root: Path, corruption: str, split: str = "train") -> None:
        corruption_dir = root / corruption
        images_path = corruption_dir / f"{split}_images.npy"
        labels_path = corruption_dir / f"{split}_labels.npy"
        if not images_path.exists() or not labels_path.exists():
            raise FileNotFoundError(f"Missing MNIST-C files for corruption '{corruption}' and split '{split}'")

        self.corruption = corruption
        self.split = split
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        image_np = self.images[index]
        if image_np.ndim == 3 and image_np.shape[-1] == 1:
            image_np = image_np[..., 0]
        image = torch.from_numpy(image_np).float().unsqueeze(0) / 255.0
        image = (image - 0.1307) / 0.3081
        label = int(self.labels[index])
        return image, label


class LocalMNISTClean(Dataset):
    def __init__(self, root: Path, split: str = "train") -> None:
        self.split = split
        self.dataset = MNIST(root=str(root), train=(split == "train"), download=False)
        self.labels = np.asarray(self.dataset.targets, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.dataset))

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        image_np = np.asarray(image, dtype=np.float32)
        image = torch.from_numpy(image_np).float().unsqueeze(0) / 255.0
        image = (image - 0.1307) / 0.3081
        return image, int(label)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def state_dict_to_cpu(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def build_random_client_preferences(num_clients: int, preferred_digits_per_client: int, seed: int):
    rng = np.random.default_rng(seed)
    digits = list(range(10))

    while True:
        preferences = {}
        digit_coverage = Counter()
        for client_id in range(num_clients):
            chosen = sorted(rng.choice(digits, size=preferred_digits_per_client, replace=False).tolist())
            preferences[f"client_{client_id}"] = chosen
            digit_coverage.update(chosen)

        if min(digit_coverage[digit] for digit in digits) >= 1:
            return preferences


def build_random_mild_noniid_split(labels, config):
    rng = np.random.default_rng(config["seed"])
    labels = np.asarray(labels)
    num_clients = config["num_clients"]
    preferences = build_random_client_preferences(
        num_clients=num_clients,
        preferred_digits_per_client=config["preferred_digits_per_client"],
        seed=config["seed"],
    )
    client_indices = {f"client_{client_id}": [] for client_id in range(num_clients)}

    for digit in range(10):
        indices = np.where(labels == digit)[0]
        rng.shuffle(indices)
        indices = indices.tolist()

        preferred_clients = [
            client_name for client_name, preferred_digits in preferences.items() if digit in preferred_digits
        ]
        secondary_clients = [client_name for client_name in client_indices if client_name not in preferred_clients]

        if not preferred_clients or not secondary_clients:
            midpoint = max(1, len(client_indices) // 2)
            ordered_clients = list(client_indices.keys())
            preferred_clients = ordered_clients[:midpoint]
            secondary_clients = ordered_clients[midpoint:]

        preferred_total = int(round(len(indices) * config["preferred_share"]))
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


def summarize_split(labels, client_indices):
    summary = {}
    for client_name, indices in client_indices.items():
        counts = Counter(int(labels[index]) for index in indices)
        summary[client_name] = {
            "num_samples": len(indices),
            "class_counts": {str(digit): counts.get(digit, 0) for digit in range(10)},
        }
    return summary


def sanitize_name(value: str) -> str:
    return value.strip().replace(" ", "_").replace("+", "_")


def get_run_root(run_root_arg: Optional[str]) -> Path:
    if run_root_arg:
        return Path(run_root_arg).resolve()
    return ARTIFACT_ROOT.resolve()


def get_corruption_run_dir(run_root: Path, corruption: str) -> Path:
    return run_root / sanitize_name(corruption)


def parse_dataset_spec(dataset_spec: str) -> List[str]:
    return [item.strip() for item in dataset_spec.split("+") if item.strip()]


def build_dataset(dataset_name: str, split: str):
    if dataset_name == "clean":
        dataset = LocalMNISTClean(MNIST_ROOT, split=split)
        return dataset, np.array(dataset.labels)

    dataset = LocalMNISTC(MNIST_C_ROOT, corruption=dataset_name, split=split)
    return dataset, np.array(dataset.labels)


def combine_component_datasets(
    datasets: List[Dataset],
    labels_list: List[np.ndarray],
    max_samples: Optional[int],
    seed: int,
    clean_mix_ratio: Optional[float],
) -> Tuple[Dataset, np.ndarray]:
    if len(datasets) == 1:
        return cap_dataset(datasets[0], labels_list[0], max_samples, seed)

    if len(datasets) == 2 and clean_mix_ratio is not None:
        total_available = sum(len(labels) for labels in labels_list)
        total_target = min(total_available, max_samples) if max_samples else total_available
        clean_target = int(round(total_target * clean_mix_ratio))
        corrupt_target = max(0, total_target - clean_target)
        sampled_parts = []
        sampled_labels = []
        for index, (dataset, labels) in enumerate(zip(datasets, labels_list)):
            part_target = clean_target if index == 0 else corrupt_target
            sampled_dataset, sampled_label_values = cap_dataset(dataset, labels, part_target, seed + index)
            sampled_parts.append(sampled_dataset)
            sampled_labels.append(sampled_label_values)
        combined_dataset = ConcatDataset(sampled_parts)
        return combined_dataset, np.concatenate(sampled_labels)

    combined_dataset = ConcatDataset(datasets)
    combined_labels = np.concatenate(labels_list)
    return cap_dataset(combined_dataset, combined_labels, max_samples, seed)


def load_dataset_pair(
    dataset_spec: str,
    clean_mix_ratio: Optional[float] = None,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    seed: int = 42,
):
    dataset_names = parse_dataset_spec(dataset_spec)
    train_datasets = []
    test_datasets = []
    train_labels = []
    test_labels = []

    for dataset_name in dataset_names:
        train_dataset, current_train_labels = build_dataset(dataset_name, split="train")
        test_dataset, current_test_labels = build_dataset(dataset_name, split="test")
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
        train_labels.append(current_train_labels)
        test_labels.append(current_test_labels)

    train_dataset, merged_train_labels = combine_component_datasets(
        train_datasets,
        train_labels,
        max_train_samples,
        seed,
        clean_mix_ratio,
    )
    test_dataset, merged_test_labels = combine_component_datasets(
        test_datasets,
        test_labels,
        max_test_samples,
        seed + 1,
        clean_mix_ratio,
    )
    return train_dataset, test_dataset, merged_train_labels, merged_test_labels, dataset_names


def cap_dataset(dataset, labels: np.ndarray, max_samples: Optional[int], seed: int):
    if not max_samples or len(labels) <= max_samples:
        return dataset, labels

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(labels), size=max_samples, replace=False))
    return Subset(dataset, chosen.tolist()), labels[chosen]


def evaluate_model(model: nn.Module, data_loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            logits = model(images)
            predictions = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    return correct / total if total else 0.0


def init_run(run_dir: Path, corruption: str, config: Dict) -> None:
    set_seed(config["seed"])
    torch.set_num_threads(config["num_threads"])

    train_dataset, test_dataset, train_labels, test_labels, dataset_components = load_dataset_pair(
        corruption,
        clean_mix_ratio=config.get("clean_mix_ratio"),
        max_train_samples=config.get("max_train_samples"),
        max_test_samples=config.get("max_test_samples"),
        seed=config["seed"],
    )
    preferences, client_indices = build_random_mild_noniid_split(train_labels, config)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "rounds").mkdir(exist_ok=True)
    (run_dir / "clients").mkdir(exist_ok=True)
    (run_dir / "global").mkdir(exist_ok=True)

    run_config = deepcopy(config)
    run_config["corruption"] = corruption
    run_config["artifact_dir"] = str(run_dir)

    save_json(run_dir / "config.json", run_config)
    save_json(run_dir / "client_preferences.json", preferences)
    save_json(run_dir / "client_indices.json", client_indices)
    save_json(run_dir / "split_summary.json", summarize_split(train_labels, client_indices))
    save_json(
        run_dir / "dataset_summary.json",
        {
            "corruption": corruption,
            "dataset_components": dataset_components,
            "num_train_samples": int(len(train_dataset)),
            "num_test_samples": int(len(test_dataset)),
            "train_class_counts": {str(digit): int((train_labels == digit).sum()) for digit in range(10)},
            "test_class_counts": {str(digit): int((test_labels == digit).sum()) for digit in range(10)},
        },
    )
    save_json(run_dir / "metrics.json", [])

    model = MNISTCNN()
    torch.save(
        {"model_state": state_dict_to_cpu(model.state_dict()), "round": 0, "corruption": corruption},
        run_dir / "global" / "global_model_round_0.pt",
    )


def run_round(run_dir: Path, round_index: int) -> None:
    config = load_json(run_dir / "config.json")
    client_indices = load_json(run_dir / "client_indices.json")
    corruption = config["corruption"]

    set_seed(config["seed"] + round_index)
    torch.set_num_threads(config["num_threads"])

    train_dataset, test_dataset, train_labels, test_labels, _ = load_dataset_pair(
        corruption,
        clean_mix_ratio=config.get("clean_mix_ratio"),
        max_train_samples=config.get("max_train_samples"),
        max_test_samples=config.get("max_test_samples"),
        seed=config["seed"],
    )
    previous = torch.load(run_dir / "global" / f"global_model_round_{round_index - 1}.pt", map_location="cpu")
    global_model = MNISTCNN()
    global_model.load_state_dict(previous["model_state"])

    criterion = nn.CrossEntropyLoss()
    local_states = {}
    client_metrics = {}
    client_sizes = {}

    for client_name, indices in client_indices.items():
        local_model = deepcopy(global_model)
        local_model.train()
        optimizer = torch.optim.Adam(
            local_model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        loader = DataLoader(
            Subset(train_dataset, indices),
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
        )

        total_loss = 0.0
        total_batches = 0
        for _ in range(config["local_epochs"]):
            for images, labels in loader:
                optimizer.zero_grad()
                logits = local_model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_batches += 1

        local_states[client_name] = state_dict_to_cpu(local_model.state_dict())
        client_sizes[client_name] = len(indices)
        client_metrics[client_name] = {
            "loss": total_loss / total_batches if total_batches else 0.0,
            "num_samples": len(indices),
        }

    total_samples = sum(client_sizes.values())
    aggregated_state = {}
    for param_name in next(iter(local_states.values())).keys():
        aggregated_state[param_name] = sum(
            state[param_name] * (client_sizes[client_name] / total_samples)
            for client_name, state in local_states.items()
        )

    global_model.load_state_dict(aggregated_state)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    accuracy = evaluate_model(global_model, test_loader)

    round_payload = {
        "round": round_index,
        "corruption": corruption,
        "global_accuracy": accuracy,
        "client_metrics": client_metrics,
        "client_sizes": client_sizes,
    }
    save_json(run_dir / "rounds" / f"round_{round_index:02d}.json", round_payload)
    torch.save(
        {
            "model_state": aggregated_state,
            "round": round_index,
            "accuracy": accuracy,
            "corruption": corruption,
        },
        run_dir / "global" / f"global_model_round_{round_index}.pt",
    )
    for client_name, state in local_states.items():
        torch.save(
            {
                "client_name": client_name,
                "round": round_index,
                "corruption": corruption,
                "model_state": state,
                "train_metrics": client_metrics[client_name],
            },
            run_dir / "clients" / f"{client_name}_round_{round_index}.pt",
        )

    metrics = load_json(run_dir / "metrics.json")
    metrics.append(round_payload)
    save_json(run_dir / "metrics.json", metrics)
    print(f"[{corruption}] round={round_index} accuracy={accuracy:.4f}", flush=True)


def finalize_run(run_dir: Path) -> Dict:
    config = load_json(run_dir / "config.json")
    final_round = config["num_rounds"]
    corruption = config["corruption"]
    latest_global = torch.load(run_dir / "global" / f"global_model_round_{final_round}.pt", map_location="cpu")
    torch.save(latest_global, run_dir / "global" / "global_model.pt")

    client_artifacts = []
    for client_id in range(config["num_clients"]):
        client_name = f"client_{client_id}"
        latest_client = torch.load(run_dir / "clients" / f"{client_name}_round_{final_round}.pt", map_location="cpu")
        target_path = run_dir / "clients" / f"{client_name}.pt"
        torch.save(latest_client, target_path)
        client_artifacts.append(
            {
                "client_name": client_name,
                "path": str(target_path),
                "num_samples": latest_client["train_metrics"]["num_samples"],
            }
        )

    summary = {
        "run_dir": str(run_dir),
        "corruption": corruption,
        "dataset_components": parse_dataset_spec(corruption),
        "num_clients": config["num_clients"],
        "num_rounds": final_round,
        "local_epochs": config["local_epochs"],
        "final_global_accuracy": latest_global["accuracy"],
        "global_model_path": str(run_dir / "global" / "global_model.pt"),
        "client_artifacts": client_artifacts,
    }
    save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def train_corruption(run_root: Path, corruption: str, config: Dict) -> Dict:
    start = time.perf_counter()
    run_dir = get_corruption_run_dir(run_root, corruption)
    init_run(run_dir, corruption, config)
    for round_index in range(1, config["num_rounds"] + 1):
        run_round(run_dir, round_index)
    summary = finalize_run(run_dir)
    summary["runtime_seconds"] = time.perf_counter() - start
    save_json(run_dir / "summary.json", summary)
    return summary


def train_all_corruptions(run_root: Path, config: Dict, corruptions: List[str]) -> Dict:
    start = time.perf_counter()
    run_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for corruption in corruptions:
        summaries.append(train_corruption(run_root, corruption, config))

    index_payload = {
        "artifact_root": str(run_root),
        "corruptions": corruptions,
        "num_corruptions": len(corruptions),
        "num_clients_per_corruption": config["num_clients"],
        "num_rounds": config["num_rounds"],
        "local_epochs": config["local_epochs"],
        "runtime_seconds": time.perf_counter() - start,
        "runs": summaries,
    }
    save_json(run_root / "index.json", index_payload)
    return index_payload


def parse_corruptions(raw: Optional[str]) -> List[str]:
    if not raw:
        return list(DEFAULT_CORRUPTIONS)
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train-all", "train-corruption", "init", "round", "finalize"], default="train-all")
    parser.add_argument("--run-root")
    parser.add_argument("--run-dir")
    parser.add_argument("--round-index", type=int)
    parser.add_argument("--corruption")
    parser.add_argument("--corruptions")
    parser.add_argument("--num-clients", type=int, default=DEFAULT_CONFIG["num_clients"])
    parser.add_argument("--num-rounds", type=int, default=DEFAULT_CONFIG["num_rounds"])
    parser.add_argument("--local-epochs", type=int, default=DEFAULT_CONFIG["local_epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_CONFIG["eval_batch_size"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--clean-mix-ratio", type=float, default=0.5)
    return parser.parse_args()


def build_config(args) -> Dict:
    config = deepcopy(DEFAULT_CONFIG)
    config.update(
        {
            "seed": args.seed,
            "num_clients": args.num_clients,
            "num_rounds": args.num_rounds,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "clean_mix_ratio": args.clean_mix_ratio,
            "corruptions": parse_corruptions(args.corruptions),
        }
    )
    return config


def main():
    args = parse_args()
    config = build_config(args)
    run_root = get_run_root(args.run_root)

    if args.mode == "train-all":
        payload = train_all_corruptions(run_root, config, config["corruptions"])
        print(json.dumps(payload, indent=2), flush=True)
        return

    if args.corruption is None and args.mode in {"train-corruption", "init"}:
        raise ValueError("--corruption is required for this mode")

    if args.mode == "train-corruption":
        summary = train_corruption(run_root, args.corruption, config)
        print(json.dumps(summary, indent=2), flush=True)
        return

    run_dir = Path(args.run_dir).resolve() if args.run_dir else get_corruption_run_dir(run_root, args.corruption)

    if args.mode == "init":
        init_run(run_dir, args.corruption, config)
        print(str(run_dir), flush=True)
        return

    if args.mode == "round":
        if args.round_index is None:
            raise ValueError("--round-index is required for --mode round")
        run_round(run_dir, args.round_index)
        return

    finalize_run(run_dir)


if __name__ == "__main__":
    main()
