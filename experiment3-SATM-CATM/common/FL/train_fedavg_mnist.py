import argparse
import json
import os
import random
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
ARTIFACT_ROOT = PROJECT_ROOT / "FL" / "artifacts"

DEFAULT_CONFIG = {
    "seed": 42,
    "num_clients": 5,
    "num_rounds": 10,
    "local_epochs": 1,
    "batch_size": 128,
    "eval_batch_size": 256,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "num_threads": min(4, os.cpu_count() or 1),
    "preferred_digits_per_client": 4,
    "preferred_share": 0.68,
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
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


def get_run_dir(run_dir_arg: Optional[str]) -> Path:
    if run_dir_arg:
        return Path(run_dir_arg).resolve()
    return (ARTIFACT_ROOT / f"mnist_fl_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()


def load_datasets():
    transform = make_transform()
    train_dataset = datasets.MNIST(root=DATA_ROOT, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_ROOT, train=False, download=False, transform=transform)
    return train_dataset, test_dataset


def init_run(run_dir: Path):
    config = deepcopy(DEFAULT_CONFIG)
    set_seed(config["seed"])
    torch.set_num_threads(config["num_threads"])

    train_dataset, _ = load_datasets()
    labels = np.array(train_dataset.targets)
    preferences, client_indices = build_random_mild_noniid_split(labels, config)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "rounds").mkdir(exist_ok=True)
    (run_dir / "clients").mkdir(exist_ok=True)
    (run_dir / "global").mkdir(exist_ok=True)

    save_json(run_dir / "config.json", config)
    save_json(run_dir / "client_preferences.json", preferences)
    save_json(run_dir / "client_indices.json", client_indices)
    save_json(run_dir / "split_summary.json", summarize_split(labels, client_indices))
    save_json(run_dir / "metrics.json", [])

    model = MNISTCNN()
    torch.save({"model_state": state_dict_to_cpu(model.state_dict()), "round": 0}, run_dir / "global" / "global_model_round_0.pt")
    print(str(run_dir), flush=True)


def run_round(run_dir: Path, round_index: int):
    config = load_json(run_dir / "config.json")
    client_indices = load_json(run_dir / "client_indices.json")

    set_seed(config["seed"] + round_index)
    torch.set_num_threads(config["num_threads"])

    train_dataset, test_dataset = load_datasets()
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
    global_model.eval()
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            logits = global_model(images)
            predictions = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = correct / total if total else 0.0
    save_json(
        run_dir / "rounds" / f"round_{round_index:02d}.json",
        {
            "round": round_index,
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
            },
            run_dir / "clients" / f"{client_name}_round_{round_index}.pt",
        )

    metrics = load_json(run_dir / "metrics.json")
    metrics.append(
        {
            "round": round_index,
            "global_accuracy": accuracy,
            "client_metrics": client_metrics,
        }
    )
    save_json(run_dir / "metrics.json", metrics)
    print(f"round={round_index} accuracy={accuracy:.4f}", flush=True)


def finalize_run(run_dir: Path):
    config = load_json(run_dir / "config.json")
    final_round = config["num_rounds"]
    latest_global = torch.load(run_dir / "global" / f"global_model_round_{final_round}.pt", map_location="cpu")
    torch.save(latest_global, run_dir / "global" / "global_model.pt")

    for client_id in range(config["num_clients"]):
        client_name = f"client_{client_id}"
        latest_client = torch.load(run_dir / "clients" / f"{client_name}_round_{final_round}.pt", map_location="cpu")
        torch.save(latest_client, run_dir / "clients" / f"{client_name}.pt")

    summary = {
        "run_dir": str(run_dir),
        "num_rounds": final_round,
        "final_global_accuracy": latest_global["accuracy"],
    }
    save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["init", "round", "finalize"], required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--round-index", type=int)
    args = parser.parse_args()

    run_dir = get_run_dir(args.run_dir)

    if args.mode == "init":
        init_run(run_dir)
        return

    if args.mode == "round":
        if args.round_index is None:
            raise ValueError("--round-index is required for --mode round")
        run_round(run_dir, args.round_index)
        return

    finalize_run(run_dir)


if __name__ == "__main__":
    main()
