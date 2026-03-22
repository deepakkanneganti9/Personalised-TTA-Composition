import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs" / "plots"
TIMING_ROOT = ROOT / "outputs" / "timing"

METHOD_FILES = {
    "Semantic": ROOT / "outputs" / "semantic" / "semantic_substitution_metadata.json",
    "Context": ROOT / "outputs" / "context" / "contextual_substitution_metadata.json",
    "MLaaS": ROOT / "outputs" / "mlaas" / "mlaas_substitution_metadata.json",
    "TTA": ROOT / "outputs" / "tta" / "tta_substitution_metadata.json",
}

# Corrected offline total from the refreshed collector/compatibility summary.
SUBSTITUTION_PREP_SECONDS = 6312.460580042001


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    counts = list(range(10, 101, 10))

    method_runtime_totals = {}
    method_runtime_per_case = {}
    for method, path in METHOD_FILES.items():
        payload = load_json(path)
        num_cases = max(int(payload.get("num_substitution_rows", 1)), 1)
        total_runtime = float(payload["runtime_seconds"])
        method_runtime_totals[method] = total_runtime
        method_runtime_per_case[method] = total_runtime / num_cases

    rows = []
    for count in counts:
        for method in METHOD_FILES:
            online_seconds = method_runtime_per_case[method] * count
            prep_seconds = 0.0 if method == "TTA" else SUBSTITUTION_PREP_SECONDS
            total_seconds = prep_seconds + online_seconds
            rows.append(
                {
                    "method": method,
                    "composition_count": count,
                    "prep_seconds": prep_seconds,
                    "online_seconds": online_seconds,
                    "total_seconds": total_seconds,
                    "total_nanoseconds": total_seconds * 1_000_000_000.0,
                }
            )

    csv_path = OUTPUT_ROOT / "translate_time_scaling.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(11, 6.5))

    colors = {
        "Semantic": "#1f77b4",
        "Context": "#ff7f0e",
        "MLaaS": "#2ca02c",
        "TTA": "#d62728",
    }

    for method in METHOD_FILES:
        method_rows = [row for row in rows if row["method"] == method]
        ax.plot(
            [row["composition_count"] for row in method_rows],
            [row["total_nanoseconds"] for row in method_rows],
            marker="o",
            linewidth=2.5,
            markersize=6,
            color=colors[method],
            label=method,
        )

    ax.set_title("Projected Overall Computation Time vs Number of Compositions", fontsize=14, weight="bold")
    ax.set_xlabel("Number of Compositions", fontsize=12)
    ax.set_ylabel("Overall Computation Time (ns)", fontsize=12)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(frameon=True)
    fig.tight_layout()

    png_path = OUTPUT_ROOT / "translate_time_scaling.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary_path = OUTPUT_ROOT / "translate_time_scaling_summary.json"
    summary = {
        "counts": counts,
        "substitution_prep_seconds": SUBSTITUTION_PREP_SECONDS,
        "method_runtime_totals": method_runtime_totals,
        "method_runtime_per_case": method_runtime_per_case,
        "csv_path": str(csv_path),
        "png_path": str(png_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
