import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

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


def bn_module_types():
    return (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def set_dropout_eval(model: nn.Module) -> nn.Module:
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()
    return model


@torch.no_grad()
def collect_target_bn_statistics(
    model: nn.Module,
    dataloader: DataLoader,
    max_batches: Optional[int] = None,
):
    recorder = BNStatisticsRecorder(model)
    records_meta = []
    for batch_index, (images, labels) in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        recorder.begin_step()
        _ = model(images)
        records_meta.append(
            {
                "batch_index": batch_index,
                "batch_size": int(labels.size(0)),
            }
        )
    recorder.close()
    summary = recorder.summarize()
    return recorder.records, summary, records_meta


def blend_bn_statistics(
    model: nn.Module,
    bn_before: Dict[str, Dict[str, Optional[List[float]]]],
    target_summary: Dict[str, Dict[str, object]],
    pseudo_sample_size: float,
    num_target_samples: int,
) -> Dict[str, Dict[str, List[float]]]:
    blended = {}
    for layer_name, module in model.named_modules():
        if not isinstance(module, bn_module_types()):
            continue
        if layer_name not in bn_before or layer_name not in target_summary:
            continue

        source_mean = torch.tensor(bn_before[layer_name]["running_mean"], dtype=torch.float32)
        source_var = torch.tensor(bn_before[layer_name]["running_var"], dtype=torch.float32)
        target_mean = torch.tensor(target_summary[layer_name]["mean_of_batch_means"], dtype=torch.float32)
        target_var = torch.tensor(target_summary[layer_name]["mean_of_batch_vars"], dtype=torch.float32)

        total_weight = float(pseudo_sample_size + num_target_samples)
        if total_weight <= 0:
            blended_mean = target_mean
            blended_var = target_var
        else:
            blended_mean = (pseudo_sample_size / total_weight) * source_mean + (num_target_samples / total_weight) * target_mean
            blended_var = (pseudo_sample_size / total_weight) * source_var + (num_target_samples / total_weight) * target_var

        module.running_mean = blended_mean.clone()
        module.running_var = blended_var.clone()
        module.track_running_stats = True

        blended[layer_name] = {
            "mean": blended_mean.tolist(),
            "var": blended_var.tolist(),
        }
    return blended


def adapt_client_with_tta_bn(
    fl_run_dir: Path,
    client_name: str,
    corruption: str,
    output_dir: Path,
    batch_size: int = 64,
    max_batches: Optional[int] = None,
    mnist_c_root: Path = DEFAULT_MNIST_C_ROOT,
    split: str = "test",
    allowed_digits: Optional[List[int]] = None,
    seed: int = 42,
    pseudo_sample_size: float = 32.0,
) -> Dict[str, object]:
    set_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[tta_bn] loading client model: {client_name}", flush=True)

    model = load_client_model(fl_run_dir=fl_run_dir, client_name=client_name)
    before_model = deepcopy(model)
    bn_before = snapshot_bn_parameters(model)

    print(f"[tta_bn] loading target dataset: {corruption}", flush=True)
    dataset = LocalMNISTC(
        root=mnist_c_root,
        corruption=corruption,
        split=split,
        allowed_digits=allowed_digits,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    set_dropout_eval(model)
    print(f"[tta_bn] baseline evaluation on {len(dataset)} samples", flush=True)
    baseline_accuracy = evaluate_accuracy(model, dataloader)

    print("[tta_bn] collecting target BN statistics", flush=True)
    bn_batch_records, bn_statistics_summary, online_metrics = collect_target_bn_statistics(
        model=model,
        dataloader=dataloader,
        max_batches=max_batches,
    )

    print("[tta_bn] blending source and target BN statistics", flush=True)
    adapted_bn_mean_var = blend_bn_statistics(
        model=model,
        bn_before=bn_before,
        target_summary=bn_statistics_summary,
        pseudo_sample_size=pseudo_sample_size,
        num_target_samples=len(dataset) if max_batches is None else sum(m["batch_size"] for m in online_metrics),
    )

    print("[tta_bn] post-adaptation evaluation", flush=True)
    adapted_accuracy = evaluate_accuracy(model, dataloader)
    adapted_prediction_distribution = evaluate_prediction_distribution(model, dataloader)
    bn_after = snapshot_bn_parameters(model)
    model_parameter_updates = compute_model_parameter_updates(before_model=before_model, after_model=model)
    adapted_model_weights = snapshot_model_parameters(model)
    adapted_bn_affine = extract_bn_affine_parameters(bn_after)

    print(f"[tta_bn] saving artifacts to {output_dir}", flush=True)
    torch.save(
        {
            "client_name": client_name,
            "corruption": corruption,
            "model_state": model.state_dict(),
        },
        output_dir / "adapted_client_model.pt",
    )
    torch.save(bn_batch_records, output_dir / "bn_batch_statistics.pt")
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
            "max_batches": max_batches,
            "allowed_digits": allowed_digits,
            "seed": seed,
            "num_target_samples": len(dataset),
            "adaptation_method": "tta_bn",
            "tta_bn_pseudo_sample_size": pseudo_sample_size,
            "tta_bn_suffix": "tta_bn",
        },
    )
    save_json(output_dir / "bn_parameters_before.json", bn_before)
    save_json(output_dir / "bn_parameters_after.json", bn_after)
    save_json(output_dir / "bn_statistics_summary.json", bn_statistics_summary)
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
        "adaptation_method": "tta_bn",
        "tta_bn_pseudo_sample_size": pseudo_sample_size,
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
    print("[tta_bn] completed", flush=True)
    return summary
