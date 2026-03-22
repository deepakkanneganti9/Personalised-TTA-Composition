import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch


EPSILON = 1e-12
DEFAULT_TTAAS_THRESHOLD = 0.5


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_torch(path: Path):
    return torch.load(path, map_location="cpu")


def flatten_state_dict(state_dict: Dict[str, torch.Tensor], keys: Optional[List[str]] = None) -> torch.Tensor:
    selected_keys = keys if keys is not None else sorted(state_dict.keys())
    flat_tensors = [state_dict[key].detach().float().reshape(-1) for key in selected_keys]
    if not flat_tensors:
        return torch.zeros(1, dtype=torch.float32)
    return torch.cat(flat_tensors)


def flatten_serialized_mapping(mapping: Dict[str, List[float]], keys: Optional[List[str]] = None) -> torch.Tensor:
    selected_keys = keys if keys is not None else sorted(mapping.keys())
    flat_tensors = [torch.tensor(mapping[key], dtype=torch.float32).reshape(-1) for key in selected_keys]
    if not flat_tensors:
        return torch.zeros(1, dtype=torch.float32)
    return torch.cat(flat_tensors)


def cosine_similarity(left: torch.Tensor, right: torch.Tensor, epsilon: float = EPSILON) -> float:
    numerator = torch.dot(left, right)
    denominator = left.norm(p=2) * right.norm(p=2) + epsilon
    return float((numerator / denominator).item())


def cosine_deviation(left: torch.Tensor, right: torch.Tensor, epsilon: float = EPSILON) -> float:
    return 1.0 - cosine_similarity(left, right, epsilon=epsilon)


def bn_affine_keys(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    return sorted(
        key for key in state_dict.keys()
        if ("running_" not in key)
        and (key.endswith(".weight") or key.endswith(".bias"))
        and ("features.1" in key or "features.5" in key or "features.9" in key or "bn" in key.lower())
    )


def bn_running_keys(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    return sorted(key for key in state_dict.keys() if "running_mean" in key or "running_var" in key)


def extract_model_state(checkpoint_or_state: Dict) -> Dict[str, torch.Tensor]:
    if "model_state" in checkpoint_or_state:
        return checkpoint_or_state["model_state"]
    return checkpoint_or_state


def compute_parameter_delta(
    before_state: Dict[str, torch.Tensor],
    after_state: Dict[str, torch.Tensor],
    keys: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    selected_keys = keys if keys is not None else sorted(after_state.keys())
    return {
        key: after_state[key].detach().float() - before_state[key].detach().float()
        for key in selected_keys
    }


def compute_wdm(reference_state: Dict[str, torch.Tensor], adapted_state: Dict[str, torch.Tensor]) -> float:
    reference_vector = flatten_state_dict(reference_state)
    adapted_vector = flatten_state_dict(adapted_state)
    return cosine_deviation(adapted_vector, reference_vector)


def compute_wdm_from_weight_files(reference_weights: Dict[str, List[float]], adapted_weights: Dict[str, List[float]]) -> float:
    shared_keys = sorted(set(reference_weights.keys()) & set(adapted_weights.keys()))
    reference_vector = flatten_serialized_mapping(reference_weights, keys=shared_keys)
    adapted_vector = flatten_serialized_mapping(adapted_weights, keys=shared_keys)
    return cosine_deviation(adapted_vector, reference_vector)


def compute_ucs(
    adapted_delta: Dict[str, torch.Tensor],
    reference_delta: Dict[str, torch.Tensor],
    layer_weights: Optional[Dict[str, float]] = None,
    epsilon: float = EPSILON,
) -> float:
    layer_names = sorted(set(adapted_delta.keys()) & set(reference_delta.keys()))
    total = 0.0
    for layer_name in layer_names:
        weight = 1.0 if layer_weights is None else float(layer_weights.get(layer_name, 1.0))
        adapted_vector = adapted_delta[layer_name].reshape(-1)
        reference_vector = reference_delta[layer_name].reshape(-1)
        total += weight * cosine_similarity(adapted_vector, reference_vector, epsilon=epsilon)
    return total


def compute_ucs_from_update_files(
    adapted_delta: Dict[str, List[float]],
    reference_delta: Dict[str, List[float]],
    layer_weights: Optional[Dict[str, float]] = None,
    epsilon: float = EPSILON,
) -> float:
    layer_names = sorted(set(adapted_delta.keys()) & set(reference_delta.keys()))
    total = 0.0
    for layer_name in layer_names:
        weight = 1.0 if layer_weights is None else float(layer_weights.get(layer_name, 1.0))
        adapted_vector = torch.tensor(adapted_delta[layer_name], dtype=torch.float32).reshape(-1)
        reference_vector = torch.tensor(reference_delta[layer_name], dtype=torch.float32).reshape(-1)
        total += weight * cosine_similarity(adapted_vector, reference_vector, epsilon=epsilon)
    return total


def compute_bnuas(
    adapted_delta: Dict[str, torch.Tensor],
    reference_delta: Dict[str, torch.Tensor],
    bn_layer_weights: Optional[Dict[str, float]] = None,
    epsilon: float = EPSILON,
) -> float:
    layer_prefixes = sorted(
        set(key.rsplit(".", 1)[0] for key in adapted_delta.keys() if key.endswith(".weight") or key.endswith(".bias"))
        & set(key.rsplit(".", 1)[0] for key in reference_delta.keys() if key.endswith(".weight") or key.endswith(".bias"))
    )

    total = 0.0
    for layer_prefix in layer_prefixes:
        gamma_key = f"{layer_prefix}.weight"
        beta_key = f"{layer_prefix}.bias"
        if gamma_key not in adapted_delta or gamma_key not in reference_delta:
            continue
        if beta_key not in adapted_delta or beta_key not in reference_delta:
            continue

        gamma_alignment = cosine_similarity(
            adapted_delta[gamma_key].reshape(-1),
            reference_delta[gamma_key].reshape(-1),
            epsilon=epsilon,
        )
        beta_alignment = cosine_similarity(
            adapted_delta[beta_key].reshape(-1),
            reference_delta[beta_key].reshape(-1),
            epsilon=epsilon,
        )
        weight = 1.0 if bn_layer_weights is None else float(bn_layer_weights.get(layer_prefix, 1.0))
        total += weight * (gamma_alignment + beta_alignment)
    return total


def compute_bnuas_from_affine_files(
    adapted_affine: Dict[str, Dict[str, List[float]]],
    reference_affine: Dict[str, Dict[str, List[float]]],
    bn_layer_weights: Optional[Dict[str, float]] = None,
    epsilon: float = EPSILON,
) -> float:
    layer_prefixes = sorted(set(adapted_affine.keys()) & set(reference_affine.keys()))
    total = 0.0
    for layer_prefix in layer_prefixes:
        gamma_alignment = cosine_similarity(
            torch.tensor(adapted_affine[layer_prefix]["gamma"], dtype=torch.float32).reshape(-1),
            torch.tensor(reference_affine[layer_prefix]["gamma"], dtype=torch.float32).reshape(-1),
            epsilon=epsilon,
        )
        beta_alignment = cosine_similarity(
            torch.tensor(adapted_affine[layer_prefix]["beta"], dtype=torch.float32).reshape(-1),
            torch.tensor(reference_affine[layer_prefix]["beta"], dtype=torch.float32).reshape(-1),
            epsilon=epsilon,
        )
        weight = 1.0 if bn_layer_weights is None else float(bn_layer_weights.get(layer_prefix, 1.0))
        total += weight * (gamma_alignment + beta_alignment)
    return total


def compute_bndas(
    adapted_bn_stats: Dict[str, Dict[str, List[float]]],
    reference_bn_stats: Dict[str, Dict[str, List[float]]],
    tau: float,
) -> float:
    shared_layers = sorted(set(adapted_bn_stats.keys()) & set(reference_bn_stats.keys()))
    if not shared_layers:
        raise ValueError("No shared BN layers were found for BNDAS.")

    distances = []
    for layer_name in shared_layers:
        adapted_mean = torch.tensor(adapted_bn_stats[layer_name]["mean"], dtype=torch.float32)
        adapted_var = torch.tensor(adapted_bn_stats[layer_name]["var"], dtype=torch.float32)
        reference_mean = torch.tensor(reference_bn_stats[layer_name]["mean"], dtype=torch.float32)
        reference_var = torch.tensor(reference_bn_stats[layer_name]["var"], dtype=torch.float32)
        distance = (adapted_mean.sub(reference_mean).norm(p=2) + adapted_var.sub(reference_var).norm(p=2)) / 2.0
        distances.append(distance)

    d_bn = torch.stack(distances).mean()
    return float(torch.exp(-d_bn / tau).item())


def compute_pdam(
    adapted_prediction_distribution: List[float],
    reference_prediction_distribution: List[float],
    threshold: float,
) -> Dict[str, float]:
    adapted = torch.tensor(adapted_prediction_distribution, dtype=torch.float32)
    reference = torch.tensor(reference_prediction_distribution, dtype=torch.float32)
    adapted = adapted / adapted.sum()
    reference = reference / reference.sum()
    wasserstein = torch.abs(torch.cumsum(adapted, dim=0) - torch.cumsum(reference, dim=0)).sum()
    return {
        "wasserstein_distance": float(wasserstein.item()),
        "pdam": 1.0 if float(wasserstein.item()) < threshold else 0.0,
    }


def align_metric_directions(raw_metrics: Dict[str, float]) -> Dict[str, float]:
    aligned = {}
    if "wdm" in raw_metrics:
        aligned["wdm"] = 1.0 - float(raw_metrics["wdm"])
    for name in ["ucs", "bnuas", "bndas", "pdam"]:
        if name in raw_metrics:
            aligned[name] = float(raw_metrics[name])
    return aligned


def minmax_scale_records(
    records: List[Dict[str, object]],
    metric_names: List[str],
) -> List[Dict[str, object]]:
    ranges = {}
    for metric_name in metric_names:
        values = [float(record["aligned_metrics"][metric_name]) for record in records]
        ranges[metric_name] = (min(values), max(values))

    scaled_records = []
    for record in records:
        scaled_metrics = {}
        for metric_name in metric_names:
            metric_value = float(record["aligned_metrics"][metric_name])
            min_value, max_value = ranges[metric_name]
            if abs(max_value - min_value) < EPSILON:
                scaled_metrics[metric_name] = 1.0
            else:
                scaled_metrics[metric_name] = (metric_value - min_value) / (max_value - min_value)
        scaled_record = dict(record)
        scaled_record["scaled_metrics"] = scaled_metrics
        scaled_records.append(scaled_record)
    return scaled_records


def compute_ttaas(
    scaled_metrics: Dict[str, float],
    metric_weights: Optional[Dict[str, float]] = None,
) -> float:
    if not scaled_metrics:
        return 0.0
    if metric_weights is None:
        metric_weights = {metric_name: 1.0 for metric_name in scaled_metrics.keys()}
    total_weight = sum(float(metric_weights[metric_name]) for metric_name in scaled_metrics.keys())
    weighted_sum = sum(float(scaled_metrics[metric_name]) * float(metric_weights[metric_name]) for metric_name in scaled_metrics.keys())
    return weighted_sum / total_weight if total_weight else 0.0


def prediction_label(ttaas: float, threshold: float) -> bool:
    return bool(ttaas >= threshold)


def confusion_counts(records: List[Dict[str, object]], threshold: float) -> Dict[str, int]:
    tp = fp = fn = tn = 0
    for record in records:
        predicted = prediction_label(float(record["ttaas"]), threshold)
        ground_truth = bool(record["ground_truth_composable"])
        if predicted and ground_truth:
            tp += 1
        elif predicted and not ground_truth:
            fp += 1
        elif (not predicted) and ground_truth:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def classification_metrics(records: List[Dict[str, object]], threshold: float) -> Dict[str, float]:
    counts = confusion_counts(records, threshold)
    total = counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"]
    accuracy = (counts["tp"] + counts["tn"]) / total if total else 0.0
    precision_denominator = counts["tp"] + counts["fp"]
    recall_denominator = counts["tp"] + counts["fn"]
    precision = counts["tp"] / precision_denominator if precision_denominator else 0.0
    recall = counts["tp"] / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall
    f1 = (2.0 * precision * recall / f1_denominator) if f1_denominator else 0.0
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **counts,
    }


def add_threshold_columns(
    records: List[Dict[str, object]],
    thresholds: List[float],
) -> List[Dict[str, object]]:
    enriched = []
    for record in records:
        updated = dict(record)
        for threshold in thresholds:
            threshold_key = str(threshold).replace(".", "_")
            predicted = prediction_label(float(record["ttaas"]), threshold)
            updated[f"predicted_at_{threshold_key}"] = predicted
            updated[f"correct_at_{threshold_key}"] = bool(predicted == bool(record["ground_truth_composable"]))
        enriched.append(updated)
    return enriched


def finalize_ttaas_records(
    records: List[Dict[str, object]],
    threshold: float = DEFAULT_TTAAS_THRESHOLD,
    metric_weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, object]]:
    metric_names = ["wdm", "ucs", "bnuas", "bndas", "pdam"]
    aligned_records = []
    for record in records:
        aligned_record = dict(record)
        aligned_record["aligned_metrics"] = align_metric_directions(record["raw_metrics"])
        aligned_records.append(aligned_record)

    scaled_records = minmax_scale_records(aligned_records, metric_names=metric_names)
    finalized_records = []
    for record in scaled_records:
        ttaas = compute_ttaas(record["scaled_metrics"], metric_weights=metric_weights)
        finalized = dict(record)
        finalized["ttaas"] = ttaas
        finalized["threshold"] = threshold
        finalized["predicted_composable"] = bool(ttaas >= threshold)
        finalized_records.append(finalized)
    return finalized_records


def bn_stats_from_tent_summary(summary_json: Dict) -> Dict[str, Dict[str, List[float]]]:
    return {
        layer_name: {
            "mean": layer_summary["mean_of_batch_means"],
            "var": layer_summary["mean_of_batch_vars"],
        }
        for layer_name, layer_summary in summary_json.items()
    }


def bn_stats_from_bn_parameters(bn_parameters_json: Dict) -> Dict[str, Dict[str, List[float]]]:
    bn_stats = {}
    for layer_name, layer_values in bn_parameters_json.items():
        if "mean" in layer_values and "var" in layer_values:
            bn_stats[layer_name] = {
                "mean": layer_values["mean"],
                "var": layer_values["var"],
            }
            continue
        running_mean = layer_values.get("running_mean")
        running_var = layer_values.get("running_var")
        if running_mean is None or running_var is None:
            continue
        bn_stats[layer_name] = {"mean": running_mean, "var": running_var}
    return bn_stats


def compute_all_metrics_from_files(
    adapted_model_path: Optional[Path] = None,
    reference_model_path: Optional[Path] = None,
    adapted_wdm_weights_path: Optional[Path] = None,
    reference_wdm_weights_path: Optional[Path] = None,
    adapted_before_model_path: Optional[Path] = None,
    reference_before_model_path: Optional[Path] = None,
    adapted_ucs_updates_path: Optional[Path] = None,
    reference_ucs_updates_path: Optional[Path] = None,
    adapted_bn_stats_path: Optional[Path] = None,
    reference_bn_stats_path: Optional[Path] = None,
    adapted_bnuas_affine_path: Optional[Path] = None,
    reference_bnuas_affine_path: Optional[Path] = None,
    adapted_prediction_distribution_path: Optional[Path] = None,
    reference_prediction_distribution_path: Optional[Path] = None,
    tau: float = 1.0,
    pdam_threshold: float = 0.1,
) -> Dict[str, object]:
    results: Dict[str, object] = {}

    if adapted_wdm_weights_path is not None and reference_wdm_weights_path is not None:
        adapted_weights = load_json(adapted_wdm_weights_path)
        reference_weights = load_json(reference_wdm_weights_path)
        results["wdm"] = compute_wdm_from_weight_files(
            reference_weights=reference_weights,
            adapted_weights=adapted_weights,
        )
    elif adapted_model_path is not None and reference_model_path is not None:
        adapted_state = extract_model_state(load_torch(adapted_model_path))
        reference_state = extract_model_state(load_torch(reference_model_path))
        results["wdm"] = compute_wdm(reference_state=reference_state, adapted_state=adapted_state)

    if adapted_ucs_updates_path is not None and reference_ucs_updates_path is not None:
        adapted_delta = load_json(adapted_ucs_updates_path)
        reference_delta = load_json(reference_ucs_updates_path)
        results["ucs"] = compute_ucs_from_update_files(
            adapted_delta=adapted_delta,
            reference_delta=reference_delta,
        )
    elif adapted_before_model_path is not None and reference_before_model_path is not None and adapted_model_path is not None and reference_model_path is not None:
        adapted_state = extract_model_state(load_torch(adapted_model_path))
        reference_state = extract_model_state(load_torch(reference_model_path))
        adapted_before_state = extract_model_state(load_torch(adapted_before_model_path))
        reference_before_state = extract_model_state(load_torch(reference_before_model_path))
        adapted_delta = compute_parameter_delta(adapted_before_state, adapted_state)
        reference_delta = compute_parameter_delta(reference_before_state, reference_state)
        results["ucs"] = compute_ucs(adapted_delta=adapted_delta, reference_delta=reference_delta)

    if adapted_bnuas_affine_path is not None and reference_bnuas_affine_path is not None:
        adapted_affine = load_json(adapted_bnuas_affine_path)
        reference_affine = load_json(reference_bnuas_affine_path)
        results["bnuas"] = compute_bnuas_from_affine_files(
            adapted_affine=adapted_affine,
            reference_affine=reference_affine,
        )
    elif adapted_before_model_path is not None and reference_before_model_path is not None and adapted_model_path is not None and reference_model_path is not None:
        adapted_state = extract_model_state(load_torch(adapted_model_path))
        reference_state = extract_model_state(load_torch(reference_model_path))
        adapted_bn_delta = compute_parameter_delta(adapted_before_state, adapted_state, keys=bn_affine_keys(adapted_state))
        reference_bn_delta = compute_parameter_delta(reference_before_state, reference_state, keys=bn_affine_keys(reference_state))
        results["bnuas"] = compute_bnuas(adapted_delta=adapted_bn_delta, reference_delta=reference_bn_delta)

    if adapted_bn_stats_path is not None and reference_bn_stats_path is not None:
        adapted_bn_payload = load_json(adapted_bn_stats_path)
        reference_bn_payload = load_json(reference_bn_stats_path)
        if not adapted_bn_payload or not reference_bn_payload:
            raise ValueError("BN statistics payloads must be non-empty for BNDAS.")
        if "mean_of_batch_means" in next(iter(adapted_bn_payload.values())):
            adapted_bn_stats = bn_stats_from_tent_summary(adapted_bn_payload)
        else:
            adapted_bn_stats = bn_stats_from_bn_parameters(adapted_bn_payload)
        if "mean_of_batch_means" in next(iter(reference_bn_payload.values())):
            reference_bn_stats = bn_stats_from_tent_summary(reference_bn_payload)
        else:
            reference_bn_stats = bn_stats_from_bn_parameters(reference_bn_payload)
        results["bndas"] = compute_bndas(
            adapted_bn_stats=adapted_bn_stats,
            reference_bn_stats=reference_bn_stats,
            tau=tau,
        )

    if adapted_prediction_distribution_path is not None and reference_prediction_distribution_path is not None:
        adapted_prediction_distribution = load_json(adapted_prediction_distribution_path)
        reference_prediction_distribution = load_json(reference_prediction_distribution_path)
        results["pdam"] = compute_pdam(
            adapted_prediction_distribution=adapted_prediction_distribution,
            reference_prediction_distribution=reference_prediction_distribution,
            threshold=pdam_threshold,
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapted-model-path")
    parser.add_argument("--reference-model-path")
    parser.add_argument("--adapted-wdm-weights-path")
    parser.add_argument("--reference-wdm-weights-path")
    parser.add_argument("--adapted-before-model-path")
    parser.add_argument("--reference-before-model-path")
    parser.add_argument("--adapted-ucs-updates-path")
    parser.add_argument("--reference-ucs-updates-path")
    parser.add_argument("--adapted-bn-stats-path")
    parser.add_argument("--reference-bn-stats-path")
    parser.add_argument("--adapted-bnuas-affine-path")
    parser.add_argument("--reference-bnuas-affine-path")
    parser.add_argument("--adapted-prediction-distribution-path")
    parser.add_argument("--reference-prediction-distribution-path")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--pdam-threshold", type=float, default=0.1)
    args = parser.parse_args()

    results = compute_all_metrics_from_files(
        adapted_model_path=Path(args.adapted_model_path).resolve() if args.adapted_model_path else None,
        reference_model_path=Path(args.reference_model_path).resolve() if args.reference_model_path else None,
        adapted_wdm_weights_path=Path(args.adapted_wdm_weights_path).resolve() if args.adapted_wdm_weights_path else None,
        reference_wdm_weights_path=Path(args.reference_wdm_weights_path).resolve() if args.reference_wdm_weights_path else None,
        adapted_before_model_path=Path(args.adapted_before_model_path).resolve() if args.adapted_before_model_path else None,
        reference_before_model_path=Path(args.reference_before_model_path).resolve() if args.reference_before_model_path else None,
        adapted_ucs_updates_path=Path(args.adapted_ucs_updates_path).resolve() if args.adapted_ucs_updates_path else None,
        reference_ucs_updates_path=Path(args.reference_ucs_updates_path).resolve() if args.reference_ucs_updates_path else None,
        adapted_bn_stats_path=Path(args.adapted_bn_stats_path).resolve() if args.adapted_bn_stats_path else None,
        reference_bn_stats_path=Path(args.reference_bn_stats_path).resolve() if args.reference_bn_stats_path else None,
        adapted_bnuas_affine_path=Path(args.adapted_bnuas_affine_path).resolve() if args.adapted_bnuas_affine_path else None,
        reference_bnuas_affine_path=Path(args.reference_bnuas_affine_path).resolve() if args.reference_bnuas_affine_path else None,
        adapted_prediction_distribution_path=Path(args.adapted_prediction_distribution_path).resolve() if args.adapted_prediction_distribution_path else None,
        reference_prediction_distribution_path=Path(args.reference_prediction_distribution_path).resolve() if args.reference_prediction_distribution_path else None,
        tau=args.tau,
        pdam_threshold=args.pdam_threshold,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
