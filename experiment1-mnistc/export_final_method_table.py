from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/Users/deepakkanneganti/Documents/Experiment 1")
OUTPUT_ROOT = ROOT / "outputs_experiment_w64"
METRICS_DIR = OUTPUT_ROOT / "metrics"

FILES = {
    "POEM": METRICS_DIR / "poem_per_corruption_logwealth_sweep.csv",
    "ARS": METRICS_DIR / "asr_per_corruption_sweep.csv",
    "CATTM": METRICS_DIR / "proposed_per_corruption_sweep.csv",
    "DSS": ROOT / "baseline" / "DSS" / "outputs" / "dss_per_corruption_best.csv",
}

SELECTED_CORRUPTIONS = ["fog", "translate", "stripe", "scale", "zigzag", "spatter"]


def load_best_accuracy_by_corruption(method: str, path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if method == "DSS":
        return {row["corruption"]: float(row["accuracy"]) for row in rows}

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["corruption"]].append(row)
    best = {}
    for corruption, candidates in grouped.items():
        chosen = max(
            candidates,
            key=lambda row: (
                float(row["f1"]),
                float(row["precision"]),
                float(row["recall"]),
                float(row["accuracy"]),
            ),
        )
        best[corruption] = float(chosen["accuracy"])
    return best


def main() -> None:
    method_values = {
        method: load_best_accuracy_by_corruption(method, path)
        for method, path in FILES.items()
    }

    table_rows = []
    for method in ["POEM", "ARS", "DSS", "CATTM"]:
        row = {"Method": method}
        for corruption in SELECTED_CORRUPTIONS:
            row[corruption.capitalize()] = round(method_values[method][corruption] * 100.0, 1)
        table_rows.append(row)

    csv_path = METRICS_DIR / "final_method_accuracy_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)

    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis("off")
    table = ax.table(
        cellText=[list(row.values()) for row in table_rows],
        colLabels=list(table_rows[0].keys()),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.1, 2.0)
    png_path = METRICS_DIR / "final_method_accuracy_table.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    main()
