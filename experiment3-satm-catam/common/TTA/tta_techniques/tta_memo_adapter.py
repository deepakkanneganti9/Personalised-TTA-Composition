import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from .tent_grad_adapter import (
    BNStatisticsRecorder,
    DEFAULT_MNIST_C_ROOT,
    LocalMNISTC,
    evaluate_accuracy,
    evaluate_prediction_distribution,
    extract_bn_affine_parameters,
    load_client_model,
    save_json,
    set_seed,
    snapshot_bn_parameters,
    snapshot_model_parameters,
    compute_model_parameter_updates,
)


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def set_dropout_eval_keep_gradients(model: nn.Module) -> nn.Module:
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()
    return model


def denormalize_to_uint8(images: torch.Tensor) -> torch.Tensor:
    images = images.detach().cpu() * MNIST_STD + MNIST_MEAN
    images = images.clamp(0.0, 1.0)
    return (images * 255.0).round().to(torch.uint8)


def normalize_uint8(images_uint8: torch.Tensor) -> torch.Tensor:
    images = images_uint8.float() / 255.0
    return (images - MNIST_MEAN) / MNIST_STD


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
    return -(marginal * (marginal.clamp_min(1e-12).log())).sum(dim=1).mean()


def adapt_client_with_tta_memo(
    fl_run_dir: Path,
    client_name: str,
    corruption: str,
    output_dir: Path,
    batch_size: int = 32,
    learning_rate: float = 2.5e-4,
    num_augmentations: int = 8,
    augmentation_severity: int = 3,
    num_steps: int = 1,
    max_batches: Optional[int] = None,
    mnist_c_root: Path = DEFAULT_MNIST_C_ROOT,
    split: str = "test",
    allowed_digits: Optional[List[int]] = None,
    seed: int = 42,
) -> Dict[str, object]:
    set_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[tta_memo] loading client model: {client_name}", flush=True)

    model = load_client_model(fl_run_dir=fl_run_dir, client_name=client_name)
    before_model = deepcopy(model)
    bn_before = snapshot_bn_parameters(model)

    print(f"[tta_memo] loading target dataset: {corruption}", flush=True)
    dataset = LocalMNISTC(
        root=mnist_c_root,
        corruption=corruption,
        split=split,
        allowed_digits=allowed_digits,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"[tta_memo] baseline evaluation on {len(dataset)} samples", flush=True)
    baseline_accuracy = evaluate_accuracy(model, dataloader)

    print("[tta_memo] configuring model for MEMO adaptation", flush=True)
    model = set_dropout_eval_keep_gradients(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    augmenter = MemoAugmenter(num_augmentations=num_augmentations, severity=augmentation_severity)
    recorder = BNStatisticsRecorder(model)

    online_metrics = []
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

    recorder.close()
    print("[tta_memo] post-adaptation evaluation", flush=True)

    model.eval()
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

    print(f"[tta_memo] saving artifacts to {output_dir}", flush=True)
    torch.save(
        {
            "client_name": client_name,
            "corruption": corruption,
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
            "split": split,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_augmentations": num_augmentations,
            "augmentation_severity": augmentation_severity,
            "num_steps": num_steps,
            "max_batches": max_batches,
            "allowed_digits": allowed_digits,
            "seed": seed,
            "num_target_samples": len(dataset),
            "adaptation_method": "tta_memo",
            "tta_memo_suffix": "tta_memo",
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
