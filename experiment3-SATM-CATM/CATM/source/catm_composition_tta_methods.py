import csv
import importlib
import importlib.util
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


STMU_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STMU_ROOT.parent
FL_SCRIPT = PROJECT_ROOT / "FL" / "train_fedavg_mnist.py"
TTA_PACKAGE_ROOT = PROJECT_ROOT / "TTA techniques"
MNIST_C_ROOT = PROJECT_ROOT / "data" / "mnist_c"
COMPOSITION_MODEL_ROOT = STMU_ROOT / "artifacts" / "composition_baselines" / "composition_models"
OUTPUT_ROOT = STMU_ROOT / "artifacts" / "composition_multi_tta_methods"


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fl_module = load_module(FL_SCRIPT, "stmu_multi_tta_fl_module")
if str(TTA_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTA_PACKAGE_ROOT))

tent_grad = importlib.import_module("tta_techniques.tent_grad_adapter")
tta_bn = importlib.import_module("tta_techniques.tta_bn_adapter")
tta_memo = importlib.import_module("tta_techniques.tta_memo_adapter")


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
    dataset = tent_grad.LocalMNISTC(
        root=MNIST_C_ROOT,
        corruption=corruption,
        split="test",
        allowed_digits=None,
    )
    return maybe_subset_dataset(dataset, max_samples)


def build_model(state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    model = fl_module.MNISTCNN()
    model.load_state_dict(state_dict)
    return model


def evaluate_model(model: nn.Module, dataset: Dataset, batch_size: int = 128) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return tent_grad.evaluate_accuracy(model, loader)


def adapt_with_tent(state_dict: Dict[str, torch.Tensor], corruption_dataset: Dataset) -> nn.Module:
    model = build_model(deepcopy(state_dict))
    model = tent_grad.configure_model_for_tent(model)
    params, _ = tent_grad.collect_tent_parameters(model)
    optimizer = torch.optim.Adam(params, lr=1e-3)
    loader = DataLoader(corruption_dataset, batch_size=64, shuffle=False, num_workers=0)
    for images, _ in loader:
        optimizer.zero_grad()
        logits = model(images)
        entropy = tent_grad.softmax_entropy(logits).mean()
        entropy.backward()
        optimizer.step()
    model.eval()
    return model


def adapt_with_tta_bn(state_dict: Dict[str, torch.Tensor], corruption_dataset: Dataset) -> nn.Module:
    model = build_model(deepcopy(state_dict))
    bn_before = tent_grad.snapshot_bn_parameters(model)
    model = tta_bn.set_dropout_eval(model)
    loader = DataLoader(corruption_dataset, batch_size=64, shuffle=False, num_workers=0)
    _, bn_summary, online_metrics = tta_bn.collect_target_bn_statistics(model, loader, max_batches=None)
    tta_bn.blend_bn_statistics(
        model=model,
        bn_before=bn_before,
        target_summary=bn_summary,
        pseudo_sample_size=32.0,
        num_target_samples=sum(m["batch_size"] for m in online_metrics),
    )
    model.eval()
    return model


def adapt_with_tta_memo(state_dict: Dict[str, torch.Tensor], corruption_dataset: Dataset) -> nn.Module:
    model = build_model(deepcopy(state_dict))
    model = tta_memo.set_dropout_eval_keep_gradients(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=2.5e-4)
    augmenter = tta_memo.MemoAugmenter(num_augmentations=8, severity=3)
    loader = DataLoader(corruption_dataset, batch_size=32, shuffle=False, num_workers=0)
    for images, _ in loader:
        optimizer.zero_grad()
        augmented = augmenter(images)
        batch_size_now = augmented.shape[0]
        logits = model(augmented.reshape(batch_size_now * 8, *images.shape[1:]))
        loss = tta_memo.marginal_entropy_loss(logits, batch_size=batch_size_now, num_augmentations=8)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def main():
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    clean_samples = 100
    corruption_samples = 100
    corruptions = ["brightness", "zigzag"]
    methods = {
        "tent": adapt_with_tent,
        "tta_bn": adapt_with_tta_bn,
        "tta_memo": adapt_with_tta_memo,
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    clean_dataset = load_clean_test_dataset(clean_samples)
    corruption_datasets = {name: load_corruption_test_dataset(name, corruption_samples) for name in corruptions}

    rows = []
    for model_path in sorted(COMPOSITION_MODEL_ROOT.glob("composition_*.pt")):
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
        composition_clients = checkpoint.get("clients", [])
        composition_length = len(composition_clients)

        before_clean_model = build_model(deepcopy(state_dict))
        before_clean_acc = evaluate_model(before_clean_model, clean_dataset)

        for corruption in corruptions:
            corruption_dataset = corruption_datasets[corruption]
            mixed_dataset = EvalDataset([clean_dataset, corruption_dataset])

            before_model = build_model(deepcopy(state_dict))
            before_corruption_acc = evaluate_model(before_model, corruption_dataset)
            before_mixed_acc = evaluate_model(before_model, mixed_dataset)

            for method_name, adapter_fn in methods.items():
                adapted_model = adapter_fn(deepcopy(state_dict), corruption_dataset)
                adapted_clean_acc = evaluate_model(adapted_model, clean_dataset)
                adapted_corruption_acc = evaluate_model(adapted_model, corruption_dataset)
                adapted_mixed_acc = evaluate_model(adapted_model, mixed_dataset)
                rows.append(
                    {
                        "tta_method": method_name,
                        "composition_id": model_path.stem,
                        "composition_clients": ",".join(composition_clients),
                        "composition_length": composition_length,
                        "corruption": corruption,
                        "clean_samples": clean_samples,
                        "corruption_samples": corruption_samples,
                        "before_clean_accuracy": before_clean_acc,
                        "before_corruption_accuracy": before_corruption_acc,
                        "before_mixed_accuracy": before_mixed_acc,
                        "adapted_clean_accuracy": adapted_clean_acc,
                        "adapted_corruption_accuracy": adapted_corruption_acc,
                        "adapted_mixed_accuracy": adapted_mixed_acc,
                        "gain_clean": adapted_clean_acc - before_clean_acc,
                        "gain_corruption": adapted_corruption_acc - before_corruption_acc,
                        "gain_mixed": adapted_mixed_acc - before_mixed_acc,
                    }
                )

    with (OUTPUT_ROOT / "composition_multi_tta_methods.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    with (OUTPUT_ROOT / "composition_multi_tta_methods.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
