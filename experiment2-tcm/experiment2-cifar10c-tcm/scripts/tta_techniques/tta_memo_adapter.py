import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from .tent_adapter import (
    BNStatisticsRecorder,
    CLASS_NAMES,
    DEFAULT_CIFAR10_C_ROOT,
    LocalCIFAR10C,
    compute_model_parameter_updates,
    evaluate_accuracy,
    evaluate_prediction_distribution,
    extract_bn_affine_parameters,
    load_client_model,
    save_json,
    set_seed,
    snapshot_bn_parameters,
    snapshot_model_parameters,
)


CIFAR10_MEAN = torch.tensor((0.4914, 0.4822, 0.4465), dtype=torch.float32).view(1, 3, 1, 1)
CIFAR10_STD = torch.tensor((0.2470, 0.2435, 0.2616), dtype=torch.float32).view(1, 3, 1, 1)


def set_dropout_eval_keep_gradients(model: nn.Module) -> nn.Module:
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()
    return model


def denormalize_to_uint8(images: torch.Tensor) -> torch.Tensor:
    mean = CIFAR10_MEAN.to(images.device, images.dtype)
    std = CIFAR10_STD.to(images.device, images.dtype)
    images = images.detach() * std + mean
    images = images.clamp(0.0, 1.0)
    return (images * 255.0).round().to(torch.uint8).cpu()


def normalize_uint8(images_uint8: torch.Tensor) -> torch.Tensor:
    images = images_uint8.float() / 255.0
    mean = CIFAR10_MEAN.to(images.device, images.dtype)
    std = CIFAR10_STD.to(images.device, images.dtype)
    return (images - mean) / std


class MemoAugmenter:
    def __init__(self, num_augmentations: int = 8, severity: int = 3) -> None:
        self.num_augmentations = num_augmentations
        self.augment = transforms.AugMix(severity=severity)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        images_uint8 = denormalize_to_uint8(images)
        augmented_batches = []
        for _ in range(self.num_augmentations):
            augmented = torch.stack([self.augment(image) for image in images_uint8], dim=0)
            augmented_batches.append(normalize_uint8(augmented))
        return torch.stack(augmented_batches, dim=1)


def marginal_entropy_loss(logits: torch.Tensor, batch_size: int, num_augmentations: int) -> torch.Tensor:
    probabilities = logits.softmax(dim=1).reshape(batch_size, num_augmentations, -1)
    marginal = probabilities.mean(dim=1)
    return -(marginal * marginal.clamp_min(1e-12).log()).sum(dim=1).mean()


@torch.inference_mode()
def evaluate_accuracy_with_progress(
    model: nn.Module,
    dataloader: DataLoader,
    stage_name: str,
    log_every: int = 5,
) -> float:
    model.eval()
    correct = 0
    total = 0
    num_batches = len(dataloader)
    for batch_index, (images, labels) in enumerate(dataloader):
        logits = model(images)
        predictions = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
        if batch_index == 0 or (batch_index + 1) % log_every == 0 or batch_index + 1 == num_batches:
            running_accuracy = correct / total if total else 0.0
            print(
                f"[tta_memo] {stage_name}: batch {batch_index + 1}/{num_batches} "
                f"| samples={total} | running_accuracy={running_accuracy:.4f}",
                flush=True,
            )
    return correct / total if total else 0.0


def adapt_client_with_tta_memo(
    fl_run_dir: Path,
    client_name: str,
    corruption: str,
    output_dir: Path,
    batch_size: int = 64,
    learning_rate: float = 2.5e-4,
    num_augmentations: int = 8,
    augmentation_severity: int = 3,
    num_steps: int = 1,
    max_batches: Optional[int] = None,
    cifar10_c_root: Path = DEFAULT_CIFAR10_C_ROOT,
    split: str = "test",
    severity: int = 1,
    allowed_classes: Optional[List[int]] = None,
    seed: int = 42,
    log_every: int = 5,
    num_threads: int = 1,
) -> Dict[str, object]:
    set_seed(seed)
    thread_count = max(1, int(num_threads))
    torch.set_num_threads(thread_count)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(thread_count)
        except RuntimeError:
            pass

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[tta_memo] loading client model: {client_name}", flush=True)
    print(f"[tta_memo] using torch threads={thread_count}", flush=True)

    model = load_client_model(fl_run_dir=fl_run_dir, client_name=client_name)
    before_model = deepcopy(model)
    bn_before = snapshot_bn_parameters(model)

    print(f"[tta_memo] loading target dataset: {corruption} severity={severity}", flush=True)
    dataset = LocalCIFAR10C(
        root=cifar10_c_root,
        corruption=corruption,
        split=split,
        severity=severity,
        allowed_classes=allowed_classes,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"[tta_memo] baseline evaluation on {len(dataset)} samples", flush=True)
    baseline_accuracy = evaluate_accuracy_with_progress(model, dataloader, "baseline_eval", log_every=log_every)

    print("[tta_memo] configuring model for MEMO adaptation", flush=True)
    model = set_dropout_eval_keep_gradients(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    augmenter = MemoAugmenter(num_augmentations=num_augmentations, severity=augmentation_severity)
    recorder = BNStatisticsRecorder(model)

    online_metrics = []
    planned_batches = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
    for batch_index, (images, labels) in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        for step_index in range(num_steps):
            recorder.begin_step()
            optimizer.zero_grad()
            augmented = augmenter(images)
            batch_size_now = augmented.shape[0]
            logits = model(augmented.reshape(batch_size_now * num_augmentations, *images.shape[1:]))
            loss = marginal_entropy_loss(logits, batch_size=batch_size_now, num_augmentations=num_augmentations)
            loss.backward()
            optimizer.step()

            clean_logits = model(images)
            clean_predictions = clean_logits.argmax(dim=1)
            batch_accuracy = (clean_predictions == labels).float().mean().item()
            online_metrics.append(
                {
                    "batch_index": batch_index,
                    "step_index": step_index,
                    "marginal_entropy": float(loss.item()),
                    "accuracy_on_clean_batch_after_step": float(batch_accuracy),
                    "batch_size": int(labels.size(0)),
                    "num_augmentations": int(num_augmentations),
                }
            )
            if batch_index == 0 or (batch_index + 1) % log_every == 0 or batch_index + 1 == planned_batches:
                print(
                    f"[tta_memo] adaptation: batch {batch_index + 1}/{planned_batches} "
                    f"| step {step_index + 1}/{num_steps} | marginal_entropy={loss.item():.4f} "
                    f"| clean_batch_acc={batch_accuracy:.4f}",
                    flush=True,
                )

    recorder.close()
    print("[tta_memo] post-adaptation evaluation", flush=True)

    model.eval()
    adapted_accuracy = evaluate_accuracy_with_progress(model, dataloader, "adapted_eval", log_every=log_every)
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

    print(f"[tta_memo] saving artifacts to {output_dir}", flush=True)
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
            "num_augmentations": num_augmentations,
            "augmentation_severity": augmentation_severity,
            "num_steps": num_steps,
            "max_batches": max_batches,
            "allowed_classes": allowed_classes,
            "class_names": CLASS_NAMES,
            "seed": seed,
            "num_target_samples": len(dataset),
            "adaptation_method": "tta_memo",
            "tta_memo_suffix": "tta_memo",
            "log_every": log_every,
            "num_threads": thread_count,
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
        "adaptation_method": "tta_memo",
        "num_augmentations": num_augmentations,
        "augmentation_severity": augmentation_severity,
        "num_steps": num_steps,
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
    print("[tta_memo] completed", flush=True)
    return summary
