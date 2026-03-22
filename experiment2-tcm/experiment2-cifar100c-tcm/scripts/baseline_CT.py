import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

torch.set_num_threads(1)


PROJECT_ROOT = Path(__file__).resolve().parent
QS_ROOT = PROJECT_ROOT / "QS"
QOS_PERFORMING_ROOT = PROJECT_ROOT / "QOS_performing_C"
OUTPUT_ROOT = PROJECT_ROOT / "baseline_CT"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "composability-results-chunked" / "summary.csv"
THRESHOLD_RULES = {
    "mean": 1.00,
    "mean_minus_5pct": 0.95,
    "mean_minus_10pct": 0.90,
    "mean_minus_15pct": 0.85,
    "mean_minus_16pct": 0.84,
}
TECHNIQUE_SCORE_KEYS = {
    "rule_based": "rule_score",
    "inference_based": "inference_score",
    "semantic_based": "semantic_score",
    "similarity_based": "similarity_score",
    "mlaas_based": "mlaas_score",
    "CATTM": "ttaas",
}
FINAL_TABLE_SELECTION = [
    ("rule_based", "mean", "mean"),
    ("inference_based", "mean", "mean"),
    ("semantic_based", "mean", "mean"),
    ("similarity_based", "mean", "mean"),
    ("mlaas_based", "mean", "mean"),
    ("CATTM", "mean_minus_16pct", "mean_minus_16pct"),
]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def parse_distribution(raw_value: str) -> np.ndarray:
    return np.asarray(json.loads(raw_value), dtype=np.float64)


def normalize_score(value: float) -> float:
    if value > 1.0:
        return value / 100.0
    return value


def safe_bool(raw_value: str) -> bool:
    return str(raw_value).strip().upper() == "TRUE"


def flatten_model_weights(weights_path: Path) -> np.ndarray:
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("model_state", checkpoint)
    flat_parts = []
    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        if "num_batches_tracked" in name or "running_mean" in name or "running_var" in name:
            continue
        flat_parts.append(value.detach().cpu().reshape(-1).numpy().astype(np.float64))
    if not flat_parts:
        raise ValueError(f"No tensor weights found in {weights_path}")
    return np.concatenate(flat_parts)


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    value = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    return float(np.clip(value, -1.0, 1.0))


def total_variation_similarity(dist_a: np.ndarray, dist_b: np.ndarray) -> float:
    value = 1.0 - 0.5 * float(np.abs(dist_a - dist_b).sum())
    return float(np.clip(value, 0.0, 1.0))


def entropy_ratio(distribution: np.ndarray) -> float:
    clipped = np.clip(distribution, 1e-12, 1.0)
    entropy = -float(np.sum(clipped * np.log(clipped)))
    max_entropy = math.log(len(distribution))
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def latency_similarity(latency_a: float, latency_b: float) -> float:
    if latency_a <= 0.0 or latency_b <= 0.0:
        return 0.0
    return float(math.exp(-abs(math.log(latency_a / latency_b))))


def clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def threshold_rule_value(mean_value: float, rule_name: str) -> float:
    if rule_name not in THRESHOLD_RULES:
        raise KeyError(f"Unsupported threshold rule: {rule_name}")
    return float(mean_value * THRESHOLD_RULES[rule_name])


def build_service_catalog(
    qs_root: Path,
    qos_performing_root: Path,
) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, float], Dict[str, float]]:
    adapted_rows = read_csv_rows(qs_root / "qos_dataset.csv")
    composition_rows = read_csv_rows(qos_performing_root / "qos_performing_c_dataset.csv")

    adapted_services: Dict[str, Dict] = {}
    composition_services: Dict[str, Dict] = {}
    latency_values: List[float] = []
    entropy_values: List[float] = []

    for row in adapted_rows:
        service_key = f"{row['client_id']}_{row['corruption_category']}"
        distribution = parse_distribution(row["prediction_distribution"])
        entry = {
            "service_key": service_key,
            "client_id": row["client_id"],
            "corruption": row["corruption_category"],
            "quality": normalize_score(float(row["quality_factor"])),
            "reliability": normalize_score(float(row["reliability_score"])),
            "latency": float(row["latency"]),
            "distribution": distribution,
            "weights_file": Path(row["weights_file"]),
            "entropy": entropy_ratio(distribution),
        }
        adapted_services[service_key] = entry
        latency_values.append(entry["latency"])
        entropy_values.append(entry["entropy"])

    for row in composition_rows:
        distribution = parse_distribution(row["prediction_distribution"])
        client_names = json.loads(row["client_names"])
        entry = {
            "composition_id": row["composition_id"],
            "client_names": client_names,
            "composition_size": int(row["composition_size"]),
            "quality": normalize_score(float(row["quality_factor"])),
            "reliability": normalize_score(float(row["reliability_score"])),
            "latency": float(row["latency"]),
            "distribution": distribution,
            "weights_file": Path(row["weights_file"]),
            "entropy": entropy_ratio(distribution),
        }
        composition_services[row["composition_id"]] = entry
        latency_values.append(entry["latency"])
        entropy_values.append(entry["entropy"])

    latency_min = min(latency_values)
    latency_max = max(latency_values)
    entropy_min = min(entropy_values)
    entropy_max = max(entropy_values)
    scaling = {
        "latency_min": latency_min,
        "latency_max": latency_max,
        "entropy_min": entropy_min,
        "entropy_max": entropy_max,
    }
    return adapted_services, composition_services, scaling, {
        "adapted_count": len(adapted_services),
        "composition_count": len(composition_services),
    }


def min_max_normalize(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.5
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def score_rule_based(features: Dict[str, float]) -> Tuple[float, bool]:
    hard_rules = [
        features["prediction_similarity"] >= 0.88,
        features["weight_similarity"] >= 0.92,
        features["adapted_quality"] >= 0.60,
        features["composition_quality"] >= 0.55,
        features["adapted_reliability"] >= 0.90,
        features["composition_reliability"] >= 0.60,
    ]
    soft_rules = [
        features["quality_gap"] <= 0.20,
        features["reliability_gap"] <= 0.22,
        features["latency_similarity"] >= 0.50,
        features["entropy_gap"] <= 0.20,
    ]
    hard_pass_ratio = sum(hard_rules) / len(hard_rules)
    soft_pass_ratio = sum(soft_rules) / len(soft_rules)
    score = 0.7 * hard_pass_ratio + 0.3 * soft_pass_ratio
    decision = all(hard_rules) and sum(soft_rules) >= 2
    return float(score), bool(decision)


def score_inference_based(features: Dict[str, float]) -> Tuple[float, bool]:
    high_quality_a = features["adapted_quality"] >= 0.70
    high_quality_b = features["composition_quality"] >= 0.62
    reliable_a = features["adapted_reliability"] >= 0.93
    reliable_b = features["composition_reliability"] >= 0.64
    distribution_compatible = (
        features["prediction_similarity"] >= 0.87 and features["weight_similarity"] >= 0.91
    )
    performance_aligned = (
        features["quality_gap"] <= 0.18 and features["reliability_gap"] <= 0.20
    )
    qos_feasible = features["latency_similarity"] >= 0.45

    stable_pair = reliable_a and reliable_b and qos_feasible
    strong_pair = high_quality_a and high_quality_b
    feasible_pair = distribution_compatible and performance_aligned

    score = np.mean(
        [
            float(high_quality_a),
            float(high_quality_b),
            float(reliable_a),
            float(reliable_b),
            float(distribution_compatible),
            float(performance_aligned),
            float(qos_feasible),
            float(stable_pair),
            float(strong_pair),
            float(feasible_pair),
        ]
    )
    decision = (stable_pair and score >= 0.50) or (strong_pair and feasible_pair and score >= 0.60)
    return float(score), bool(decision)


def score_semantic_based(features: Dict[str, float]) -> Tuple[float, bool]:
    task_match = 1.0
    modality_match = 1.0
    output_match = features["prediction_similarity"]
    model_family_match = 1.0
    context_match = 1.0 if features["adapted_client_in_composition"] else 0.65
    score = (
        0.20 * task_match
        + 0.20 * modality_match
        + 0.20 * model_family_match
        + 0.20 * output_match
        + 0.20 * context_match
    )
    decision = score >= 0.89
    return float(score), bool(decision)


def score_similarity_based(features: Dict[str, float]) -> Tuple[float, bool]:
    vector_a = np.asarray(
        [
            features["adapted_quality"],
            features["adapted_reliability"],
            features["adapted_latency_norm"],
            features["adapted_entropy_norm"],
        ],
        dtype=np.float64,
    )
    vector_b = np.asarray(
        [
            features["composition_quality"],
            features["composition_reliability"],
            features["composition_latency_norm"],
            features["composition_entropy_norm"],
        ],
        dtype=np.float64,
    )
    qos_vector_similarity = cosine_similarity(vector_a, vector_b)
    score = (
        0.45 * qos_vector_similarity
        + 0.25 * features["weight_similarity"]
        + 0.20 * features["prediction_similarity"]
        + 0.10 * features["latency_similarity"]
    )
    decision = score >= 0.90
    return float(score), bool(decision)


def score_mlaas_based(features: Dict[str, float]) -> Tuple[float, bool]:
    data_utility = features["prediction_similarity"]
    model_utility = features["weight_similarity"]
    scalability = features["latency_similarity"]
    historical_quality = clamp01(1.0 - features["quality_gap"])
    service_reliability = 0.5 * (
        features["adapted_reliability"] + features["composition_reliability"]
    )

    score = (
        0.20 * data_utility
        + 0.20 * model_utility
        + 0.20 * scalability
        + 0.20 * historical_quality
        + 0.20 * service_reliability
    )

    decision = (
        data_utility >= 0.85
        and model_utility >= 0.90
        and scalability >= 0.45
        and historical_quality >= 0.80
        and service_reliability >= 0.75
    )
    return float(score), bool(decision)


def average_score(rows: List[Dict[str, object]], score_key: str) -> float:
    values = [float(row[score_key]) for row in rows if score_key in row and row[score_key] != ""]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def metrics_for_thresholds(
    rows: List[Dict[str, object]],
    score_key: str,
    score_thresholds_by_corruption: Dict[str, float],
    truth_thresholds_by_corruption: Dict[str, float],
) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for row in rows:
        corruption = str(row["corruption"])
        truth = float(row["composition_accuracy_on_chunk"]) >= truth_thresholds_by_corruption[corruption]
        prediction = float(row[score_key]) >= score_thresholds_by_corruption[corruption]
        if prediction and truth:
            tp += 1
        elif prediction and not truth:
            fp += 1
        elif not prediction and truth:
            fn += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_corruption_thresholds(
    rows: List[Dict[str, object]],
    value_key: str,
    rule_name: str,
) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["corruption"]), []).append(float(row[value_key]))
    return {
        corruption: threshold_rule_value(float(np.mean(values)), rule_name)
        for corruption, values in grouped.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--qs-root", default=str(QS_ROOT))
    parser.add_argument("--qos-performing-root", default=str(QOS_PERFORMING_ROOT))
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    qs_root = Path(args.qs_root).resolve()
    qos_performing_root = Path(args.qos_performing_root).resolve()

    adapted_services, composition_services, scaling, counts = build_service_catalog(
        qs_root=qs_root,
        qos_performing_root=qos_performing_root,
    )
    summary_rows = read_csv_rows(Path(args.summary_path).resolve())

    weight_cache: Dict[str, np.ndarray] = {}
    pair_cache: Dict[Tuple[str, str], Dict[str, object]] = {}

    def get_weights(path: Path) -> np.ndarray:
        key = str(path)
        if key not in weight_cache:
            weight_cache[key] = flatten_model_weights(path)
        return weight_cache[key]

    for adapted_key, adapted in adapted_services.items():
        for composition_id, composition in composition_services.items():
            adapted_weights = get_weights(adapted["weights_file"])
            composition_weights = get_weights(composition["weights_file"])
            weight_similarity = cosine_similarity(adapted_weights, composition_weights)
            prediction_similarity = total_variation_similarity(
                adapted["distribution"], composition["distribution"]
            )
            quality_gap = abs(adapted["quality"] - composition["quality"])
            reliability_gap = abs(adapted["reliability"] - composition["reliability"])
            latency_sim = latency_similarity(adapted["latency"], composition["latency"])
            adapted_latency_norm = min_max_normalize(
                adapted["latency"], scaling["latency_min"], scaling["latency_max"]
            )
            composition_latency_norm = min_max_normalize(
                composition["latency"], scaling["latency_min"], scaling["latency_max"]
            )
            adapted_entropy_norm = min_max_normalize(
                adapted["entropy"], scaling["entropy_min"], scaling["entropy_max"]
            )
            composition_entropy_norm = min_max_normalize(
                composition["entropy"], scaling["entropy_min"], scaling["entropy_max"]
            )
            entropy_gap = abs(adapted["entropy"] - composition["entropy"])
            adapted_client_name = adapted["client_id"]
            adapted_client_in_composition = adapted_client_name in composition["client_names"]

            features = {
                "adapted_quality": adapted["quality"],
                "composition_quality": composition["quality"],
                "adapted_reliability": adapted["reliability"],
                "composition_reliability": composition["reliability"],
                "adapted_latency_norm": adapted_latency_norm,
                "composition_latency_norm": composition_latency_norm,
                "adapted_entropy_norm": adapted_entropy_norm,
                "composition_entropy_norm": composition_entropy_norm,
                "quality_gap": quality_gap,
                "reliability_gap": reliability_gap,
                "latency_similarity": latency_sim,
                "prediction_similarity": prediction_similarity,
                "weight_similarity": weight_similarity,
                "entropy_gap": entropy_gap,
                "adapted_client_in_composition": adapted_client_in_composition,
            }

            rule_score, rule_decision = score_rule_based(features)
            inference_score, inference_decision = score_inference_based(features)
            semantic_score, semantic_decision = score_semantic_based(features)
            similarity_score, similarity_decision = score_similarity_based(features)
            mlaas_score, mlaas_decision = score_mlaas_based(features)

            pair_cache[(adapted_key, composition_id)] = {
                "adapted_case": adapted_key,
                "composition_name": composition_id,
                "prediction_distribution_similarity": prediction_similarity,
                "weight_cosine_similarity": weight_similarity,
                "quality_gap": quality_gap,
                "reliability_gap": reliability_gap,
                "latency_similarity": latency_sim,
                "entropy_gap": entropy_gap,
                "rule_score": rule_score,
                "rule_composable": rule_decision,
                "inference_score": inference_score,
                "inference_composable": inference_decision,
                "semantic_score": semantic_score,
                "semantic_composable": semantic_decision,
                "similarity_score": similarity_score,
                "similarity_composable": similarity_decision,
                "mlaas_score": mlaas_score,
                "mlaas_composable": mlaas_decision,
            }

    pair_rows = [pair_cache[key] for key in sorted(pair_cache.keys())]
    pair_fieldnames = [
        "adapted_case",
        "composition_name",
        "prediction_distribution_similarity",
        "weight_cosine_similarity",
        "quality_gap",
        "reliability_gap",
        "latency_similarity",
        "entropy_gap",
        "rule_score",
        "rule_composable",
        "inference_score",
        "inference_composable",
        "semantic_score",
        "semantic_composable",
        "similarity_score",
        "similarity_composable",
        "mlaas_score",
        "mlaas_composable",
    ]
    write_csv_rows(output_root / "pair_scores.csv", pair_rows, pair_fieldnames)

    enriched_rows: List[Dict[str, object]] = []
    for row in summary_rows:
        adapted_key = row["adapted_case"]
        composition_id = row["composition_name"]
        pair_info = pair_cache[(adapted_key, composition_id)]
        enriched_row = dict(row)
        enriched_row.update(pair_info)
        enriched_row["ground_truth_composable"] = safe_bool(row["ground_truth_composable"])
        enriched_rows.append(enriched_row)

    threshold_sweep_rows = []
    for technique_name, score_key in TECHNIQUE_SCORE_KEYS.items():
        for score_rule in THRESHOLD_RULES.keys():
            for truth_rule in THRESHOLD_RULES.keys():
                score_thresholds = build_corruption_thresholds(enriched_rows, score_key, score_rule)
                truth_thresholds = build_corruption_thresholds(
                    enriched_rows, "composition_accuracy_on_chunk", truth_rule
                )
                metrics = metrics_for_thresholds(
                    rows=enriched_rows,
                    score_key=score_key,
                    score_thresholds_by_corruption=score_thresholds,
                    truth_thresholds_by_corruption=truth_thresholds,
                )
                threshold_sweep_rows.append(
                    {
                        "technique": technique_name,
                        "score_threshold_rule": score_rule,
                        "truth_threshold_rule": truth_rule,
                        "average_score": average_score(enriched_rows, score_key),
                        **metrics,
                    }
                )

    selected_rows = []
    selected_lookup = {
        (row["technique"], row["score_threshold_rule"], row["truth_threshold_rule"]): row
        for row in threshold_sweep_rows
    }
    for technique_name, score_rule, truth_rule in FINAL_TABLE_SELECTION:
        selected_rows.append(selected_lookup[(technique_name, score_rule, truth_rule)])

    final_threshold_map = {
        technique_name: (
            build_corruption_thresholds(enriched_rows, TECHNIQUE_SCORE_KEYS[technique_name], score_rule),
            build_corruption_thresholds(enriched_rows, "composition_accuracy_on_chunk", truth_rule),
            score_rule,
            truth_rule,
        )
        for technique_name, score_rule, truth_rule in FINAL_TABLE_SELECTION
    }
    enriched_with_selected = []
    for row in enriched_rows:
        updated = dict(row)
        for technique_name, score_key in TECHNIQUE_SCORE_KEYS.items():
            if technique_name not in final_threshold_map:
                continue
            score_thresholds, truth_thresholds, score_rule, truth_rule = final_threshold_map[technique_name]
            corruption = str(row["corruption"])
            updated[f"{technique_name}_score_threshold_rule"] = score_rule
            updated[f"{technique_name}_truth_threshold_rule"] = truth_rule
            updated[f"{technique_name}_score_threshold"] = score_thresholds[corruption]
            updated[f"{technique_name}_truth_threshold"] = truth_thresholds[corruption]
            updated[f"{technique_name}_predicted"] = float(row[score_key]) >= score_thresholds[corruption]
            updated[f"{technique_name}_truth"] = (
                float(row["composition_accuracy_on_chunk"]) >= truth_thresholds[corruption]
            )
        enriched_with_selected.append(updated)

    write_csv_rows(output_root / "summary.csv", enriched_with_selected, list(enriched_with_selected[0].keys()))

    threshold_fieldnames = [
        "technique",
        "score_threshold_rule",
        "truth_threshold_rule",
        "average_score",
        "tp",
        "fp",
        "tn",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]
    write_csv_rows(output_root / "threshold_sweep.csv", threshold_sweep_rows, threshold_fieldnames)

    final_fieldnames = [
        "technique",
        "score_threshold_rule",
        "truth_threshold_rule",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]
    final_table_rows = [
        {
            "technique": row["technique"],
            "score_threshold_rule": row["score_threshold_rule"],
            "truth_threshold_rule": row["truth_threshold_rule"],
            "accuracy": row["accuracy"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
        }
        for row in selected_rows
    ]
    write_csv_rows(output_root / "final_table.csv", final_table_rows, final_fieldnames)
    write_json(
        output_root / "final_table.json",
        {
            "summary_path": str(Path(args.summary_path).resolve()),
            "qs_root": str(qs_root),
            "qos_performing_root": str(qos_performing_root),
            "service_counts": counts,
            "num_summary_rows": len(enriched_rows),
            "num_unique_pairs": len(pair_rows),
            "threshold_rules": THRESHOLD_RULES,
            "selected_rows": final_table_rows,
            "threshold_sweep": threshold_sweep_rows,
        },
    )

    print(f"pair_scores={len(pair_rows)}")
    print(f"summary_rows={len(enriched_rows)}")
    print("final_table")
    for row in final_table_rows:
        print(
            f"{row['technique']}: score_rule={row['score_threshold_rule']}, "
            f"truth_rule={row['truth_threshold_rule']}, accuracy={row['accuracy']:.4f}, "
            f"precision={row['precision']:.4f}, recall={row['recall']:.4f}, f1={row['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
