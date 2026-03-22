import csv
import json
from collections import defaultdict
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent / "results"
FILTERED_SUB_PATH = Path(__file__).resolve().parent / "SATM" / "results" / "satm_substitution_results_50_50.csv"
CATM_PATH = Path(__file__).resolve().parent / "CATM" / "results" / "catm_filtered_results_50_50.csv"


def composition_id_from_case(case: str) -> str:
    middle = case[len("perf_") :].split("_plus_", 1)[0]
    ids = [part for part in middle.split("_") if part]
    return "composition_" + "_".join(ids)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    filtered_rows = list(csv.DictReader(FILTERED_SUB_PATH.open()))
    catm_rows = list(csv.DictReader(CATM_PATH.open()))

    catm_lookup = {
        (row["composition_id"], row["corruption"]): row
        for row in catm_rows
    }

    grouped = defaultdict(list)

    for row in filtered_rows:
        tta_method = row["tta_method"]
        corruption = row["tta_corruption"]
        composition_id = composition_id_from_case(row["case"])
        catm = catm_lookup[(composition_id, corruption)]

        if tta_method == "tent":
            comp_clean = float(catm["composition_tent_clean_accuracy"])
            comp_mixed = float(catm["composition_tent_mixed_accuracy"])
        elif tta_method == "tta_bn":
            comp_clean = float(catm["composition_tta_bn_clean_accuracy"])
            comp_mixed = float(catm["composition_tta_bn_mixed_accuracy"])
        elif tta_method == "tta_memo":
            comp_clean = float(catm["composition_tta_memo_clean_accuracy"])
            comp_mixed = float(catm["composition_tta_memo_mixed_accuracy"])
        else:
            raise ValueError(f"Unsupported method: {tta_method}")

        grouped[(tta_method, corruption)].append(
            {
                "Method": tta_method,
                "Corruption": corruption,
                "Deployment": float(row["deployment_accuracy"]),
                "Data Shift": float(row["data_shift_accuracy"]),
                "Initial Mixed": float(row["initial_mixed_accuracy"]),
                "Composition-Level TTA Clean": comp_clean,
                "Composition-Level TTA Mixed": comp_mixed,
                "SATM": float(row["protected_v2_mixed_accuracy"]),
                "CATM Clean": float(catm["composition_gala_clean_accuracy"]),
                "CATM Mixed": float(catm["composition_gala_mixed_accuracy"]),
                "Random Sub 20%": float(row["substitution_20_random_accuracy"]),
                "Random Sub 50%": float(row["substitution_50_random_accuracy"]),
            }
        )

    summary_rows = []
    for (method, corruption), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        summary_rows.append(
            {
                "Method": method,
                "Corruption": corruption,
                "Cases": len(rows),
                "Deployment": mean([r["Deployment"] for r in rows]),
                "Data Shift": mean([r["Data Shift"] for r in rows]),
                "Initial Mixed": mean([r["Initial Mixed"] for r in rows]),
                "Composition-Level TTA Clean": mean([r["Composition-Level TTA Clean"] for r in rows]),
                "Composition-Level TTA Mixed": mean([r["Composition-Level TTA Mixed"] for r in rows]),
                "SATM": mean([r["SATM"] for r in rows]),
                "CATM Clean": mean([r["CATM Clean"] for r in rows]),
                "CATM Mixed": mean([r["CATM Mixed"] for r in rows]),
                "Random Sub 20%": mean([r["Random Sub 20%"] for r in rows]),
                "Random Sub 50%": mean([r["Random Sub 50%"] for r in rows]),
            }
        )

    csv_path = RESULTS_DIR / "final_table_by_corruption_50_50.csv"
    json_path = RESULTS_DIR / "final_table_by_corruption_50_50.json"
    md_path = RESULTS_DIR / "final_table_by_corruption_50_50.md"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)

    header = list(summary_rows[0].keys())
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in summary_rows:
        formatted = []
        for key in header:
            value = row[key]
            if isinstance(value, float):
                formatted.append(f"{value:.4f}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
