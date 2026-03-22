import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FL.train_fedavg_cifar10 import CIFAR10CNN, CLASS_NAMES


DATA_ROOT = PROJECT_ROOT / "Data"
DEFAULT_CIFAR10_C_ROOT = DATA_ROOT / "CIFAR-10-C"


class LocalCIFAR10C(Dataset):
    def __init__(
        self,
        root: Path,
        corruption: str,
        split: str = "test",
        severity: int = 1,
        allowed_classes: Optional[List[int]] = None,
    ) -> None:
        self.root = self._resolve_root(root)
        self.corruption = corruption
        self.split = split
        self.severity = severity
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )

        if split != "test":
            raise ValueError("CIFAR-10-C only provides the corrupted test stream; use split='test'.")
        if severity < 1 or severity > 5:
            raise ValueError("severity must be in [1, 5] for CIFAR-10-C.")

        images_path = self.root / f"{corruption}.npy"
        labels_path = self.root / "labels.npy"
        if not images_path.exists() or not labels_path.exists():
            raise FileNotFoundError(f"Could not find CIFAR-10-C files for corruption '{corruption}' in {self.root}")

        images = np.load(images_path)
        labels = np.load(labels_path)

        samples_per_severity = images.shape[0] // 5
        start = (severity - 1) * samples_per_severity
        end = severity * samples_per_severity
        images = images[start:end]
        labels = labels[start:end]

        if allowed_classes is not None:
            allowed = set(int(class_id) for class_id in allowed_classes)
            keep = [index for index, label in enumerate(labels.tolist()) if int(label) in allowed]
            images = images[keep]
            labels = labels[keep]

        self.images = images
        self.labels = labels

    @staticmethod
    def _resolve_root(root: Path) -> Path:
        nested_root = root / "CIFAR-10-C"
        if nested_root.exists():
            return nested_root
        return root

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = self.images[index]
        image = self.transform(image.astype(np.uint8))
        return image, int(self.labels[index])


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)
    return -(probabilities * logits.log_softmax(dim=1)).sum(dim=1)


def bn_module_types():
    return (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def snapshot_bn_parameters(model: nn.Module) -> Dict[str, Dict[str, Optional[List[float]]]]:
    snapshot = {}
    for name, module in model.named_modules():
        if isinstance(module, bn_module_types()):
            snapshot[name] = {
                "weight": module.weight.detach().cpu().tolist() if module.affine and module.weight is not None else None,
                "bias": module.bias.detach().cpu().tolist() if module.affine and module.bias is not None else None,
                "running_mean": module.running_mean.detach().cpu().tolist() if module.running_mean is not None else None,
                "running_var": module.running_var.detach().cpu().tolist() if module.running_var is not None else None,
                "num_features": int(module.num_features),
                "track_running_stats": bool(module.track_running_stats),
            }
    return snapshot


def snapshot_model_parameters(model: nn.Module) -> Dict[str, List[float]]:
    return {
        name: parameter.detach().cpu().reshape(-1).tolist()
        for name, parameter in model.state_dict().items()
        if "running_" not in name and "num_batches_tracked" not in name
    }


def compute_model_parameter_updates(before_model: nn.Module, after_model: nn.Module) -> Dict[str, List[float]]:
    before_state = before_model.state_dict()
    after_state = after_model.state_dict()
    return {
        name: (after_state[name].detach().cpu() - before_state[name].detach().cpu()).reshape(-1).tolist()
        for name in after_state.keys()
        if "num_batches_tracked" not in name
    }


def extract_bn_affine_parameters(
    bn_snapshot: Dict[str, Dict[str, Optional[List[float]]]]
) -> Dict[str, Dict[str, Optional[List[float]]]]:
    return {
        layer_name: {
            "gamma": layer_values["weight"],
            "beta": layer_values["bias"],
        }
        for layer_name, layer_values in bn_snapshot.items()
    }


class BNStatisticsRecorder:
    def __init__(self, model: nn.Module) -> None:
        self.step_index = -1
        self.records = []
        self.handles = []

        for layer_name, module in model.named_modules():
            if isinstance(module, bn_module_types()):
                self.handles.append(module.register_forward_pre_hook(self._make_hook(layer_name)))

    def _make_hook(self, layer_name: str):
        def hook(module, inputs):
            activations = inputs[0].detach()
            if activations.ndim == 2:
                dims = (0,)
            else:
                dims = (0, 2, 3)

            batch_mean = activations.mean(dim=dims).cpu()
            batch_var = activations.var(dim=dims, unbiased=False).cpu()
            self.records.append(
                {
                    "step": self.step_index,
                    "layer": layer_name,
                    "batch_mean": batch_mean,
                    "batch_var": batch_var,
                }
            )

        return hook

    def begin_step(self) -> None:
        self.step_index += 1

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def summarize(self) -> Dict[str, Dict[str, object]]:
        summary: Dict[str, Dict[str, object]] = {}
        for record in self.records:
            layer_name = record["layer"]
            layer_summary = summary.setdefault(layer_name, {"batch_means": [], "batch_vars": []})
            layer_summary["batch_means"].append(record["batch_mean"])
            layer_summary["batch_vars"].append(record["batch_var"])

        result = {}
        for layer_name, values in summary.items():
            means_tensor = torch.stack(values["batch_means"])
            vars_tensor = torch.stack(values["batch_vars"])
            result[layer_name] = {
                "num_steps": int(means_tensor.shape[0]),
                "mean_of_batch_means": means_tensor.mean(dim=0).tolist(),
                "std_of_batch_means": means_tensor.std(dim=0, unbiased=False).tolist(),
                "mean_of_batch_vars": vars_tensor.mean(dim=0).tolist(),
                "std_of_batch_vars": vars_tensor.std(dim=0, unbiased=False).tolist(),
            }
        return result


def configure_model_for_tent(model: nn.Module) -> nn.Module:
    model.train()
    model.requires_grad_(False)

    for module in model.modules():
        if isinstance(module, bn_module_types()):
            if module.affine:
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
        elif isinstance(module, nn.Dropout):
            module.eval()

    return model


def collect_tent_parameters(model: nn.Module):
    parameters = []
    parameter_names = []
    for module_name, module in model.named_modules():
        if isinstance(module, bn_module_types()) and module.affine:
            parameters.append(module.weight)
            parameters.append(module.bias)
            parameter_names.append(f"{module_name}.weight")
            parameter_names.append(f"{module_name}.bias")
    return parameters, parameter_names


def load_client_model(fl_run_dir: Path, client_name: str) -> nn.Module:
    checkpoint_path = fl_run_dir / "clients" / f"{client_name}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model = CIFAR10CNN()
    model.load_state_dict(state_dict)
    return model


def evaluate_accuracy(model: nn.Module, dataloader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            logits = model(images)
            predictions = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    return correct / total if total else 0.0


@torch.no_grad()
def evaluate_prediction_distribution(model: nn.Module, dataloader: DataLoader) -> List[float]:
    model.eval()
    probability_sum = None
    total_examples = 0
    for images, _ in dataloader:
        probabilities = model(images).softmax(dim=1)
        if probability_sum is None:
            probability_sum = probabilities.sum(dim=0)
        else:
            probability_sum += probabilities.sum(dim=0)
        total_examples += images.size(0)

    if probability_sum is None or total_examples == 0:
        return []
    return (probability_sum / total_examples).cpu().tolist()


def adapt_client_with_tent(
    fl_run_dir: Path,
    client_name: str,
    corruption: str,
    output_dir: Path,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    max_batches: Optional[int] = None,
    cifar10_c_root: Path = DEFAULT_CIFAR10_C_ROOT,
    split: str = "test",
    severity: int = 1,
    allowed_classes: Optional[List[int]] = None,
    seed: int = 42,
) -> Dict[str, object]:
    set_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[tent] loading client model: {client_name}", flush=True)

    model = load_client_model(fl_run_dir=fl_run_dir, client_name=client_name)
    before_model = deepcopy(model)
    bn_before = snapshot_bn_parameters(model)
    print(f"[tent] loading target dataset: {corruption} severity={severity}", flush=True)

    dataset = LocalCIFAR10C(
        root=cifar10_c_root,
        corruption=corruption,
        split=split,
        severity=severity,
        allowed_classes=allowed_classes,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"[tent] baseline evaluation on {len(dataset)} samples", flush=True)
    baseline_accuracy = evaluate_accuracy(model, dataloader)

    print("[tent] configuring model for TENT", flush=True)
    model = configure_model_for_tent(model)
    parameters, parameter_names = collect_tent_parameters(model)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    recorder = BNStatisticsRecorder(model)
    print(f"[tent] adapting with {len(parameter_names)} BN affine parameters", flush=True)

    online_metrics = []
    for batch_index, (images, labels) in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        recorder.begin_step()
        optimizer.zero_grad()
        logits = model(images)
        entropy = softmax_entropy(logits).mean()
        entropy.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_accuracy = (predictions == labels).float().mean().item()
        online_metrics.append(
            {
                "batch_index": batch_index,
                "entropy": float(entropy.item()),
                "accuracy_before_update_output": float(batch_accuracy),
                "batch_size": int(labels.size(0)),
            }
        )

    recorder.close()
    print("[tent] post-adaptation evaluation", flush=True)

    adapted_accuracy = evaluate_accuracy(model, dataloader)
    adapted_prediction_distribution = evaluate_prediction_distribution(model, dataloader)
    bn_after = snapshot_bn_parameters(model)
    bn_summary = recorder.summarize()
    model_parameter_updates = compute_model_parameter_updates(before_model=before_model, after_model=model)
    adapted_model_weights = snapshot_model_parameters(model)
    adapted_bn_affine = extract_bn_affine_parameters(bn_after)
    adapted_bn_mean_var = {
        layer_name: {
            "mean": layer_summary["mean_of_batch_means"],
            "var": layer_summary["mean_of_batch_vars"],
        }
        for layer_name, layer_summary in bn_summary.items()
    }

    print(f"[tent] saving artifacts to {output_dir}", flush=True)
    torch.save(
        {
            "client_name": client_name,
            "corruption": corruption,
            "severity": severity,
            "dataset": "CIFAR-10-C",
            "model_state": model.state_dict(),
        },
        output_dir / "adapted_client_model.pt",
    )
    torch.save(recorder.records, output_dir / "bn_batch_statistics.pt")
    torch.save(adapted_model_weights, output_dir / "wdm_adapted_weights.pt")
    torch.save(model_parameter_updates, output_dir / "ucs_adapted_layer_updates.pt")
    torch.save(adapted_bn_mean_var, output_dir / "bndas_adapted_bn_mean_var.pt")
    torch.save(adapted_bn_affine, output_dir / "bnuas_adapted_bn_gamma_beta.pt")
    torch.save(adapted_prediction_distribution, output_dir / "pdam_adapted_prediction_distribution.pt")

    save_json(
        output_dir / "config.json",
        {
            "fl_run_dir": str(fl_run_dir),
            "client_name": client_name,
            "corruption": corruption,
            "dataset": "CIFAR-10",
            "target_dataset": "CIFAR-10-C",
            "split": split,
            "severity": severity,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_batches": max_batches,
            "allowed_classes": allowed_classes,
            "class_names": CLASS_NAMES,
            "seed": seed,
            "num_target_samples": len(dataset),
            "tent_parameter_names": parameter_names,
        },
    )
    save_json(output_dir / "bn_parameters_before.json", bn_before)
    save_json(output_dir / "bn_parameters_after.json", bn_after)
    save_json(output_dir / "bn_statistics_summary.json", bn_summary)
    save_json(output_dir / "online_metrics.json", online_metrics)
    save_json(output_dir / "wdm_adapted_weights.json", adapted_model_weights)
    save_json(output_dir / "ucs_adapted_layer_updates.json", model_parameter_updates)
    save_json(output_dir / "bndas_adapted_bn_mean_var.json", adapted_bn_mean_var)
    save_json(output_dir / "bnuas_adapted_bn_gamma_beta.json", adapted_bn_affine)
    save_json(output_dir / "pdam_adapted_prediction_distribution.json", adapted_prediction_distribution)

    summary = {
        "client_name": client_name,
        "corruption": corruption,
        "severity": severity,
        "dataset": "CIFAR-10",
        "target_dataset": "CIFAR-10-C",
        "num_target_samples": len(dataset),
        "baseline_accuracy": baseline_accuracy,
        "adapted_accuracy": adapted_accuracy,
        "num_adaptation_batches": len(online_metrics),
        "adaptation_method": "tent",
        "separate_artifacts": {
            "wdm_weights": "wdm_adapted_weights.json",
            "ucs_layer_updates": "ucs_adapted_layer_updates.json",
            "bndas_bn_mean_var": "bndas_adapted_bn_mean_var.json",
            "bnuas_bn_gamma_beta": "bnuas_adapted_bn_gamma_beta.json",
            "pdam_prediction_distribution": "pdam_adapted_prediction_distribution.json",
        },
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "summary.json", summary)
    print("[tent] completed", flush=True)
    return summary
