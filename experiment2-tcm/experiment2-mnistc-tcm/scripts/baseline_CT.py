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
SELECTED_FINAL_THRESHOLDS = [
    {
        "technique": "CATTM",
        "score_key": "ttaas",
        "gain_threshold": 0.05,
        "decision_threshold": 0.65,
    },
    {
        "technique": "inference_based",
        "score_key": "inference_score",
        "gain_threshold": 0.05,
        "decision_threshold": 0.65,
    },
    {
        "technique": "semantic_based",
        "score_key": "semantic_score",
        "gain_threshold": 0.05,
        "decision_threshold": 0.93,
    },
    {
        "technique": "similarity_based",
        "score_key": "similarity_score",
        "gain_threshold": 0.05,
        "decision_threshold": 0.90,
    },
    {
        "technique": "rule_based",
        "score_key": "rule_score",
        "gain_threshold": 0.05,
        "decision_threshold": 0.85,
    },
    {
        "technique": "mlaas_based",
        "score_key": "mlaas_score",
        "gain_threshold": 0.05,
        "decision_threshold": 0.80,
    },
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


def build_service_catalog() -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, float], Dict[str, float]]:
    adapted_rows = read_csv_rows(QS_ROOT / "qos_dataset.csv")
    composition_rows = read_csv_rows(QOS_PERFORMING_ROOT / "qos_performing_c_dataset.csv")

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


def precision_recall_f1(rows: List[Dict[str, object]], decision_key: str) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for row in rows:
        raw_prediction = row[decision_key]
        raw_truth = row["ground_truth_composable"]
        prediction = safe_bool(raw_prediction) if isinstance(raw_prediction, str) else bool(raw_prediction)
        truth = safe_bool(raw_truth) if isinstance(raw_truth, str) else bool(raw_truth)
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


def average_score(rows: List[Dict[str, object]], score_key: str) -> float:
    values = [float(row[score_key]) for row in rows if score_key in row and row[score_key] != ""]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def metrics_for_score_threshold(
    rows: List[Dict[str, object]],
    score_key: str,
    decision_threshold: float,
    gain_threshold: float,
) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    positives = 0
    for row in rows:
        gain = float(row["aggregated_accuracy_on_chunk"]) - float(row["composition_accuracy_on_chunk"])
        truth = gain >= gain_threshold
        prediction = float(row[score_key]) >= decision_threshold
        positives += int(truth)
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
        "positives": positives,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    adapted_services, composition_services, scaling, counts = build_service_catalog()
    summary_rows = read_csv_rows(QS_ROOT / "summary.csv")

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
    write_csv_rows(OUTPUT_ROOT / "pair_scores.csv", pair_rows, pair_fieldnames)

    enriched_rows: List[Dict[str, object]] = []
    for row in summary_rows:
        adapted_key = row["adapted_case"]
        composition_id = row["composition_name"]
        pair_info = pair_cache[(adapted_key, composition_id)]
        enriched_row = dict(row)
        enriched_row.update(pair_info)
        enriched_row["ground_truth_composable"] = safe_bool(row["ground_truth_composable"])
        enriched_rows.append(enriched_row)

    summary_fieldnames = list(enriched_rows[0].keys())
    write_csv_rows(OUTPUT_ROOT / "summary.csv", enriched_rows, summary_fieldnames)

    metrics_rows = []
    for technique_name, decision_key in [
        ("rule_based", "rule_composable"),
        ("inference_based", "inference_composable"),
        ("semantic_based", "semantic_composable"),
        ("similarity_based", "similarity_composable"),
        ("mlaas_based", "mlaas_composable"),
    ]:
        metrics = precision_recall_f1(enriched_rows, decision_key)
        score_key = decision_key.replace("_composable", "_score")
        metrics_rows.append(
            {
                "technique": technique_name,
                "threshold": "",
                "average_score": average_score(enriched_rows, score_key),
                **metrics,
            }
        )

    for threshold_label, decision_key in [
        ("0.35", "predicted_at_0_35"),
        ("0.40", "predicted_at_0_4"),
        ("0.50", "predicted_at_0_5"),
        ("0.60", "predicted_at_0_6"),
    ]:
        metrics = precision_recall_f1(enriched_rows, decision_key)
        metrics_rows.append(
            {
                "technique": "CATTM",
                "threshold": threshold_label,
                "average_score": average_score(enriched_rows, "ttaas"),
                **metrics,
            }
        )

    metric_fieldnames = [
        "technique",
        "threshold",
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
    write_csv_rows(OUTPUT_ROOT / "metrics.csv", metrics_rows, metric_fieldnames)
    write_csv_rows(OUTPUT_ROOT / "final_comparison_table.csv", metrics_rows, metric_fieldnames)
    write_json(
        OUTPUT_ROOT / "metrics.json",
        {
            "service_counts": counts,
            "num_summary_rows": len(enriched_rows),
            "num_unique_pairs": len(pair_rows),
            "metrics": metrics_rows,
        },
    )

    final_table_rows = []
    for config in SELECTED_FINAL_THRESHOLDS:
        metrics = metrics_for_score_threshold(
            rows=enriched_rows,
            score_key=config["score_key"],
            decision_threshold=config["decision_threshold"],
            gain_threshold=config["gain_threshold"],
        )
        final_table_rows.append(
            {
                "technique": config["technique"],
                "ground_truth_gain_threshold": config["gain_threshold"],
                "decision_threshold": config["decision_threshold"],
                "average_score": average_score(enriched_rows, config["score_key"]),
                **metrics,
            }
        )

    final_fieldnames = [
        "technique",
        "ground_truth_gain_threshold",
        "decision_threshold",
        "average_score",
        "positives",
        "tp",
        "fp",
        "tn",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]
    write_csv_rows(OUTPUT_ROOT / "final_table.csv", final_table_rows, final_fieldnames)
    write_json(
        OUTPUT_ROOT / "final_table.json",
        {
            "selected_thresholds": SELECTED_FINAL_THRESHOLDS,
            "rows": final_table_rows,
        },
    )

    print(f"pair_scores={len(pair_rows)}")
    print(f"summary_rows={len(enriched_rows)}")
    for row in metrics_rows:
        print(
            f"{row['technique']}"
            f"{'@' + row['threshold'] if row['threshold'] else ''}: "
            f"score={row['average_score']:.4f}, "
            f"accuracy={row['accuracy']:.4f}, "
            f"precision={row['precision']:.4f}, recall={row['recall']:.4f}, f1={row['f1']:.4f}"
        )
    print("selected_final_table")
    for row in final_table_rows:
        print(
            f"{row['technique']}: gain>={row['ground_truth_gain_threshold']:.2f}, "
            f"decision>={row['decision_threshold']:.2f}, accuracy={row['accuracy']:.4f}, "
            f"precision={row['precision']:.4f}, recall={row['recall']:.4f}, f1={row['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
