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


RELEASE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = RELEASE_ROOT / "results"
QOS_ROOT = RESULTS_ROOT / "qos"
COMPATIBILITY_ROOT = RESULTS_ROOT / "compatibility"
FINAL_ROOT = RESULTS_ROOT / "final"

THRESHOLD_RULES = {
    "mean": 1.00,
    "mean_minus_16pct": 0.84,
}

METHODS = {
    "tent": {
        "method": "TTA Grad",
        "summary_path": COMPATIBILITY_ROOT / "tent" / "summary.csv",
        "adapted_qos_path": QOS_ROOT / "qos_tent.csv",
    },
    "tta_bn": {
        "method": "TTA-BN",
        "summary_path": COMPATIBILITY_ROOT / "tta_bn" / "summary.csv",
        "adapted_qos_path": QOS_ROOT / "qos_tta_bn.csv",
    },
    "tta_memo": {
        "method": "TTA-MEMO",
        "summary_path": COMPATIBILITY_ROOT / "tta_memo" / "summary.csv",
        "adapted_qos_path": QOS_ROOT / "qos_tta_memo.csv",
    },
}

PERFORMING_QOS_PATH = QOS_ROOT / "qos_performing_c.csv"
TECHNIQUE_SCORE_KEYS = {
    "rule_based": "rule_score",
    "inference_based": "inference_score",
    "semantic_based": "semantic_score",
    "similarity_based": "similarity_score",
    "CATTM": "ttaas",
}
FINAL_TABLE_SELECTION = [
    ("rule_based", "mean", "mean"),
    ("inference_based", "mean", "mean"),
    ("semantic_based", "mean", "mean"),
    ("similarity_based", "mean", "mean"),
    ("CATTM", "mean_minus_16pct", "mean_minus_16pct"),
]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
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


def resolve_release_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return RELEASE_ROOT / path


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


def min_max_normalize(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.5
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def threshold_rule_value(mean_value: float, rule_name: str) -> float:
    return float(mean_value * THRESHOLD_RULES[rule_name])


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


def build_service_catalog(
    adapted_qos_path: Path,
    performing_qos_path: Path,
) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, float]]:
    adapted_rows = read_csv_rows(adapted_qos_path)
    composition_rows = read_csv_rows(performing_qos_path)
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
            "weights_file": resolve_release_path(row["weights_file"]),
            "entropy": entropy_ratio(distribution),
        }
        adapted_services[service_key] = entry
        latency_values.append(entry["latency"])
        entropy_values.append(entry["entropy"])

    for row in composition_rows:
        distribution = parse_distribution(row["prediction_distribution"])
        entry = {
            "composition_id": row["composition_id"],
            "client_names": json.loads(row["client_names"]),
            "quality": normalize_score(float(row["quality_factor"])),
            "reliability": normalize_score(float(row["reliability_score"])),
            "latency": float(row["latency"]),
            "distribution": distribution,
            "weights_file": resolve_release_path(row["weights_file"]),
            "entropy": entropy_ratio(distribution),
        }
        composition_services[row["composition_id"]] = entry
        latency_values.append(entry["latency"])
        entropy_values.append(entry["entropy"])

    scaling = {
        "latency_min": min(latency_values),
        "latency_max": max(latency_values),
        "entropy_min": min(entropy_values),
        "entropy_max": max(entropy_values),
    }
    return adapted_services, composition_services, scaling


def build_pair_cache(
    adapted_services: Dict[str, Dict],
    composition_services: Dict[str, Dict],
    scaling: Dict[str, float],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    weight_cache: Dict[str, np.ndarray] = {}
    pair_cache: Dict[Tuple[str, str], Dict[str, float]] = {}

    def get_weights(path: Path) -> np.ndarray:
        key = str(path)
        if key not in weight_cache:
            weight_cache[key] = flatten_model_weights(path)
        return weight_cache[key]

    for adapted_key, adapted in adapted_services.items():
        for composition_id, composition in composition_services.items():
            adapted_weights = get_weights(adapted["weights_file"])
            composition_weights = get_weights(composition["weights_file"])
            features = {
                "adapted_quality": adapted["quality"],
                "composition_quality": composition["quality"],
                "adapted_reliability": adapted["reliability"],
                "composition_reliability": composition["reliability"],
                "adapted_latency_norm": min_max_normalize(
                    adapted["latency"], scaling["latency_min"], scaling["latency_max"]
                ),
                "composition_latency_norm": min_max_normalize(
                    composition["latency"], scaling["latency_min"], scaling["latency_max"]
                ),
                "adapted_entropy_norm": min_max_normalize(
                    adapted["entropy"], scaling["entropy_min"], scaling["entropy_max"]
                ),
                "composition_entropy_norm": min_max_normalize(
                    composition["entropy"], scaling["entropy_min"], scaling["entropy_max"]
                ),
                "quality_gap": abs(adapted["quality"] - composition["quality"]),
                "reliability_gap": abs(adapted["reliability"] - composition["reliability"]),
                "latency_similarity": latency_similarity(adapted["latency"], composition["latency"]),
                "prediction_similarity": total_variation_similarity(
                    adapted["distribution"], composition["distribution"]
                ),
                "weight_similarity": cosine_similarity(adapted_weights, composition_weights),
                "entropy_gap": abs(adapted["entropy"] - composition["entropy"]),
                "adapted_client_in_composition": adapted["client_id"] in composition["client_names"],
            }
            pair_cache[(adapted_key, composition_id)] = {
                "rule_score": score_rule_based(features)[0],
                "inference_score": score_inference_based(features)[0],
                "semantic_score": score_semantic_based(features)[0],
                "similarity_score": score_similarity_based(features)[0],
            }
    return pair_cache


def build_public_table(method_key: str) -> List[Dict[str, object]]:
    config = METHODS[method_key]
    adapted_services, composition_services, scaling = build_service_catalog(
        config["adapted_qos_path"],
        PERFORMING_QOS_PATH,
    )
    pair_cache = build_pair_cache(adapted_services, composition_services, scaling)
    summary_rows = read_csv_rows(config["summary_path"])

    enriched_rows: List[Dict[str, object]] = []
    for row in summary_rows:
        enriched = dict(row)
        enriched.update(pair_cache[(row["adapted_case"], row["composition_name"])])
        enriched_rows.append(enriched)

    public_rows = []
    for technique_name, score_rule, truth_rule in FINAL_TABLE_SELECTION:
        score_key = TECHNIQUE_SCORE_KEYS[technique_name]
        score_thresholds = build_corruption_thresholds(enriched_rows, score_key, score_rule)
        truth_thresholds = build_corruption_thresholds(
            enriched_rows, "composition_accuracy_on_chunk", truth_rule
        )
        metrics = metrics_for_thresholds(
            enriched_rows,
            score_key,
            score_thresholds,
            truth_thresholds,
        )
        public_rows.append(
            {
                "method": config["method"],
                "technique": technique_name,
                "accuracy": round(metrics["accuracy"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
            }
        )
    return public_rows


def main() -> None:
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    combined_rows: List[Dict[str, object]] = []
    manifest_rows = []

    for method_key in METHODS:
        rows = build_public_table(method_key)
        combined_rows.extend(rows)
        method_output = FINAL_ROOT / f"{method_key}_final_table.csv"
        write_csv_rows(
            method_output,
            rows,
            ["method", "technique", "accuracy", "precision", "recall", "f1"],
        )
        manifest_rows.append(
            {
                "method": METHODS[method_key]["method"],
                "output_table": str(method_output.relative_to(RELEASE_ROOT)),
                "summary_path": str(METHODS[method_key]["summary_path"].relative_to(RELEASE_ROOT)),
                "adapted_qos_path": str(METHODS[method_key]["adapted_qos_path"].relative_to(RELEASE_ROOT)),
                "performing_qos_path": str(PERFORMING_QOS_PATH.relative_to(RELEASE_ROOT)),
            }
        )

    write_csv_rows(
        FINAL_ROOT / "combined_final_table.csv",
        combined_rows,
        ["method", "technique", "accuracy", "precision", "recall", "f1"],
    )
    write_json(
        FINAL_ROOT / "final_tables_manifest.json",
        {
            "threshold_rules": THRESHOLD_RULES,
            "selection": FINAL_TABLE_SELECTION,
            "tables": manifest_rows,
        },
    )


if __name__ == "__main__":
    main()
