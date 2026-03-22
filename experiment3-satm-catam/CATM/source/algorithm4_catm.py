import copy
import csv
import importlib
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


STMU_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STMU_ROOT.parent
FL_SCRIPT = PROJECT_ROOT / "FL" / "train_fedavg_mnist.py"
TTA_PACKAGE_ROOT = PROJECT_ROOT / "TTA techniques"
FL_RUN_DIR = PROJECT_ROOT / "FL" / "artifacts" / "mnist_fl_baseline_5clients_run"
MNIST_C_ROOT = PROJECT_ROOT / "data" / "mnist_c"
OUTPUT_ROOT = STMU_ROOT / "artifacts" / "composition_gala_tta"
COMPOSITION_MODEL_ROOT = STMU_ROOT / "artifacts" / "composition_baselines" / "composition_models"


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fl_module = load_module(FL_SCRIPT, "stmu_gala_fl_module")
if str(TTA_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTA_PACKAGE_ROOT))
tent_module = importlib.import_module("tta_techniques.tent_adapter")


class EvalDataset(Dataset):
    def __init__(self, datasets_list: List[Dataset]) -> None:
        self.datasets_list = datasets_list
        self.offsets = []
        total = 0
        for dataset in datasets_list:
            self.offsets.append(total)
            total += len(dataset)
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int):
        for dataset, offset in zip(self.datasets_list, self.offsets):
            if index < offset + len(dataset):
                return dataset[index - offset]
        raise IndexError(index)


def make_clean_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def maybe_subset_dataset(dataset: Dataset, max_samples: int):
    if len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, list(range(max_samples)))


def load_clean_test_dataset(max_samples: int):
    dataset = datasets.MNIST(
        root=PROJECT_ROOT / "data",
        train=False,
        download=False,
        transform=make_clean_transform(),
    )
    return maybe_subset_dataset(dataset, max_samples)


def load_corruption_test_dataset(corruption: str, max_samples: int):
    dataset = tent_module.LocalMNISTC(
        root=MNIST_C_ROOT,
        corruption=corruption,
        split="test",
        allowed_digits=None,
    )
    return maybe_subset_dataset(dataset, max_samples)


def load_checkpoint_state(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint["model_state"] if "model_state" in checkpoint else checkpoint


def build_model(state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    model = fl_module.MNISTCNN()
    model.load_state_dict(state_dict)
    return model


def evaluate_model(model: nn.Module, dataset: Dataset, batch_size: int = 128) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return tent_module.evaluate_accuracy(model, loader)


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    return -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean()


def layer_parameter_groups(model: nn.Module) -> List[List[str]]:
    groups = []
    for module_name, module in model.named_modules():
        names = []
        if hasattr(module, "weight") and isinstance(getattr(module, "weight"), nn.Parameter):
            names.append(f"{module_name}.weight")
        if hasattr(module, "bias") and isinstance(getattr(module, "bias"), nn.Parameter) and getattr(module, "bias") is not None:
            names.append(f"{module_name}.bias")
        if names:
            groups.append(names)
    return groups


def named_params_dict(model: nn.Module) -> Dict[str, nn.Parameter]:
    return dict(model.named_parameters())


def zero_all_grads(model: nn.Module) -> None:
    for param in model.parameters():
        if param.grad is not None:
            param.grad = None


def compute_group_update(
    model: nn.Module,
    x: torch.Tensor,
    param_names: Iterable[str],
    learning_rate: float,
) -> Dict[str, torch.Tensor]:
    model.eval()
    zero_all_grads(model)
    logits = model(x)
    loss = entropy_loss(logits)
    loss.backward()
    params = named_params_dict(model)
    updates = {}
    for name in param_names:
        grad = params[name].grad
        if grad is None:
            updates[name] = torch.zeros_like(params[name].data)
        else:
            updates[name] = -learning_rate * grad.detach().clone()
    return updates


def apply_updates(model: nn.Module, updates: Dict[str, torch.Tensor]) -> None:
    params = named_params_dict(model)
    with torch.no_grad():
        for name, update in updates.items():
            params[name].add_(update)


def merge_update_dicts(update_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    for update_dict in update_dicts:
        merged.update(update_dict)
    return merged


def flatten_updates(updates: Dict[str, torch.Tensor]) -> torch.Tensor:
    if not updates:
        return torch.zeros(1)
    return torch.cat([tensor.reshape(-1).float() for tensor in updates.values()])


def cosine_alignment_score(current_updates: Dict[str, torch.Tensor], displacement_updates: Dict[str, torch.Tensor]) -> float:
    current_vec = flatten_updates(current_updates)
    total_updates = {}
    for key in current_updates:
        total_updates[key] = current_updates[key] + displacement_updates.get(key, torch.zeros_like(current_updates[key]))
    total_vec = flatten_updates(total_updates)
    current_norm = current_vec.norm().item()
    total_norm = total_vec.norm().item()
    if current_norm == 0.0 or total_norm == 0.0:
        return -1.0
    return float(torch.dot(current_vec, total_vec).item() / (current_norm * total_norm))


def adapt_full_layers(
    state_dict: Dict[str, torch.Tensor],
    corruption_dataset: Dataset,
    learning_rate: float,
    batch_size: int,
) -> nn.Module:
    model = build_model(copy.deepcopy(state_dict))
    group_names = layer_parameter_groups(model)
    trainable_names = set(name for group in group_names for name in group)
    for name, param in model.named_parameters():
        param.requires_grad = name in trainable_names
    loader = DataLoader(corruption_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for x, _ in loader:
        update_dicts = [compute_group_update(model, x, group, learning_rate) for group in group_names]
        apply_updates(model, merge_update_dicts(update_dicts))
    return model


def adapt_gala_layers(
    state_dict: Dict[str, torch.Tensor],
    corruption_dataset: Dataset,
    learning_rate: float,
    batch_size: int,
    reset_window: int,
    threshold: float,
    warmup_steps: int = 1,
) -> Tuple[nn.Module, Dict[str, float]]:
    model = build_model(copy.deepcopy(state_dict))
    groups = layer_parameter_groups(model)
    trainable_names = set(name for group in groups for name in group)
    for name, param in model.named_parameters():
        param.requires_grad = name in trainable_names

    loader = DataLoader(corruption_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    displacement: List[Dict[str, torch.Tensor]] = [{name: torch.zeros_like(named_params_dict(model)[name].data) for name in group} for group in groups]
    selected_counts = [0 for _ in groups]
    skipped_steps = 0

    for step_idx, (x, _) in enumerate(loader):
        if step_idx % reset_window == 0:
            displacement = [{name: torch.zeros_like(named_params_dict(model)[name].data) for name in group} for group in groups]

        group_updates = [compute_group_update(model, x, group, learning_rate) for group in groups]

        if (step_idx % reset_window) < warmup_steps:
            merged = merge_update_dicts(group_updates)
            apply_updates(model, merged)
            for group_idx, update_dict in enumerate(group_updates):
                for name, update in update_dict.items():
                    displacement[group_idx][name] = displacement[group_idx][name] + update
                selected_counts[group_idx] += 1
            continue

        scores = [cosine_alignment_score(group_updates[idx], displacement[idx]) for idx in range(len(groups))]
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        if scores[best_idx] > threshold:
            apply_updates(model, group_updates[best_idx])
            for name, update in group_updates[best_idx].items():
                displacement[best_idx][name] = displacement[best_idx][name] + update
            selected_counts[best_idx] += 1
        else:
            skipped_steps += 1

    stats = {
        "num_groups": len(groups),
        "selected_group_0": selected_counts[0] if len(selected_counts) > 0 else 0,
        "selected_group_1": selected_counts[1] if len(selected_counts) > 1 else 0,
        "selected_group_2": selected_counts[2] if len(selected_counts) > 2 else 0,
        "selected_group_3": selected_counts[3] if len(selected_counts) > 3 else 0,
        "selected_group_4": selected_counts[4] if len(selected_counts) > 4 else 0,
        "selected_group_5": selected_counts[5] if len(selected_counts) > 5 else 0,
        "skipped_steps": skipped_steps,
    }
    return model, stats


def main():
    clean_samples = 100
    corruption_samples = 100
    batch_size = 1
    learning_rate = 1e-3
    reset_window = 20
    threshold = 0.75
    corruptions = ["brightness", "zigzag"]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    clean_dataset = load_clean_test_dataset(clean_samples)
    corruption_datasets = {name: load_corruption_test_dataset(name, corruption_samples) for name in corruptions}

    rows = []
    for model_path in sorted(COMPOSITION_MODEL_ROOT.glob("composition_*.pt")):
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
        composition_clients = checkpoint.get("clients", [])
        composition_length = len(composition_clients)

        before_clean_model = build_model(copy.deepcopy(state_dict))
        before_clean_acc = evaluate_model(before_clean_model, clean_dataset)

        for corruption in corruptions:
            corruption_dataset = corruption_datasets[corruption]
            mixed_dataset = EvalDataset([clean_dataset, corruption_dataset])

            before_model = build_model(copy.deepcopy(state_dict))
            before_corruption_acc = evaluate_model(before_model, corruption_dataset)
            before_mixed_acc = evaluate_model(before_model, mixed_dataset)

            full_model = adapt_full_layers(copy.deepcopy(state_dict), corruption_dataset, learning_rate, batch_size)
            full_clean_acc = evaluate_model(full_model, clean_dataset)
            full_corruption_acc = evaluate_model(full_model, corruption_dataset)
            full_mixed_acc = evaluate_model(full_model, mixed_dataset)

            gala_model, gala_stats = adapt_gala_layers(
                copy.deepcopy(state_dict),
                corruption_dataset,
                learning_rate,
                batch_size,
                reset_window,
                threshold,
            )
            gala_clean_acc = evaluate_model(gala_model, clean_dataset)
            gala_corruption_acc = evaluate_model(gala_model, corruption_dataset)
            gala_mixed_acc = evaluate_model(gala_model, mixed_dataset)

            rows.append(
                {
                    "composition_id": model_path.stem,
                    "composition_clients": ",".join(composition_clients),
                    "composition_length": composition_length,
                    "corruption": corruption,
                    "clean_samples": clean_samples,
                    "corruption_samples": corruption_samples,
                    "before_clean_accuracy": before_clean_acc,
                    "before_corruption_accuracy": before_corruption_acc,
                    "before_mixed_accuracy": before_mixed_acc,
                    "full_adapt_clean_accuracy": full_clean_acc,
                    "full_adapt_corruption_accuracy": full_corruption_acc,
                    "full_adapt_mixed_accuracy": full_mixed_acc,
                    "gala_clean_accuracy": gala_clean_acc,
                    "gala_corruption_accuracy": gala_corruption_acc,
                    "gala_mixed_accuracy": gala_mixed_acc,
                    "gala_gain_vs_before_mixed": gala_mixed_acc - before_mixed_acc,
                    "gala_gain_vs_full_adapt_mixed": gala_mixed_acc - full_mixed_acc,
                    "gala_gain_vs_before_corruption": gala_corruption_acc - before_corruption_acc,
                    "gala_gain_vs_full_adapt_corruption": gala_corruption_acc - full_corruption_acc,
                    **gala_stats,
                }
            )

    with (OUTPUT_ROOT / "composition_gala_tta.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    with (OUTPUT_ROOT / "composition_gala_tta.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
