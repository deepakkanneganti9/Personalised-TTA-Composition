import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CONFIGS = {
    "grad": {
        "summary": ROOT / "results" / "compatibility" / "grad" / "summary_with_baselines.csv",
        "thresholds": {
            "rule_based": 1.0,
            "inference_based": 1.0,
            "semantic_based": 1.0,
            "similarity_based": 1.0,
            "mlaas_based": 0.98,
            "CATTM": 0.84,
        },
    },
    "tta_bn": {
        "summary": ROOT / "results" / "compatibility" / "tta_bn" / "summary_with_baselines.csv",
        "thresholds": {
            "rule_based": 1.0,
            "inference_based": 1.0,
            "semantic_based": 1.0,
            "similarity_based": 1.0,
            "mlaas_based": 0.98,
            "CATTM": 0.84,
        },
    },
    "tta_memo": {
        "summary": ROOT / "results" / "compatibility" / "tta_memo" / "summary_with_baselines.csv",
        "thresholds": {
            "rule_based": 1.0,
            "inference_based": 1.0,
            "semantic_based": 1.0,
            "similarity_based": 1.0,
            "mlaas_based": 0.98,
            "CATTM": 0.84,
        },
    },
}

SCORE_KEYS = {
    "rule_based": "rule_score",
    "inference_based": "inference_score",
    "semantic_based": "semantic_score",
    "similarity_based": "similarity_score",
    "mlaas_based": "mlaas_score",
    "CATTM": "ttaas",
}


def read_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metrics_for_grouped_threshold(rows, score_key: str, factor: float):
    by_corruption = defaultdict(list)
    for row in rows:
        by_corruption[row["corruption"]].append(row)

    tp = fp = tn = fn = 0
    for corruption_rows in by_corruption.values():
        score_threshold = factor * (
            sum(float(row[score_key]) for row in corruption_rows) / len(corruption_rows)
        )
        truth_threshold = factor * (
            sum(float(row["composition_accuracy_on_chunk"]) for row in corruption_rows)
            / len(corruption_rows)
        )
        for row in corruption_rows:
            pred = float(row[score_key]) >= score_threshold
            truth = float(row["composition_accuracy_on_chunk"]) >= truth_threshold
            if pred and truth:
                tp += 1
            elif pred and not truth:
                fp += 1
            elif not pred and truth:
                fn += 1
            else:
                tn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_table(method: str, summary_path: Path):
    rows = read_rows(summary_path)
    thresholds = CONFIGS[method]["thresholds"]
    out = []
    for technique, factor in thresholds.items():
        metrics = metrics_for_grouped_threshold(rows, SCORE_KEYS[technique], factor)
        out.append(
            {
                "method": method,
                "technique": technique,
                **metrics,
            }
        )
    return out


def main():
    combined = []
    for method, cfg in CONFIGS.items():
        rows = build_table(method, cfg["summary"])
        combined.extend(rows)
        write_rows(
            ROOT / "results" / "final" / method / "final_table.csv",
            rows,
            ["method", "technique", "accuracy", "precision", "recall", "f1"],
        )

    write_rows(
        ROOT / "results" / "final" / "combined_final_tables.csv",
        combined,
        ["method", "technique", "accuracy", "precision", "recall", "f1"],
    )

    for row in combined:
        print(
            f"{row['method']} {row['technique']}: "
            f"acc={row['accuracy']:.4f}, prec={row['precision']:.4f}, "
            f"rec={row['recall']:.4f}, f1={row['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
