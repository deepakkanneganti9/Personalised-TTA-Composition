import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


RELEASE_ROOT = Path(__file__).resolve().parents[1]
INTERNAL_ROOT = RELEASE_ROOT / "results" / "final" / "internal"
PUBLIC_ROOT = RELEASE_ROOT / "results" / "final"
BASELINE_SCRIPT = RELEASE_ROOT / "scripts" / "baseline_CT.py"

TECHNIQUE_ORDER = [
    "rule_based",
    "inference_based",
    "semantic_based",
    "similarity_based",
    "mlaas_based",
    "CATTM",
]

METHODS = [
    {
        "name": "TTA-Grad",
        "summary_path": RELEASE_ROOT / "results" / "compatibility" / "TTA-Grad" / "summary.csv",
        "qs_root": RELEASE_ROOT / "results" / "qos" / "tta_grad",
        "qos_performing_root": RELEASE_ROOT / "results" / "qos" / "composition",
        "internal_output": INTERNAL_ROOT / "TTA-Grad",
    },
    {
        "name": "TTA-BN",
        "summary_path": RELEASE_ROOT / "results" / "compatibility" / "TTA-BN" / "summary.csv",
        "qs_root": RELEASE_ROOT / "results" / "qos" / "tta_bn",
        "qos_performing_root": RELEASE_ROOT / "results" / "qos" / "composition",
        "internal_output": INTERNAL_ROOT / "TTA-BN",
    },
    {
        "name": "TTA-MEMO",
        "summary_path": RELEASE_ROOT / "results" / "compatibility" / "TTA-MEMO" / "summary.csv",
        "qs_root": RELEASE_ROOT / "results" / "qos" / "tta_memo",
        "qos_performing_root": RELEASE_ROOT / "results" / "qos" / "composition",
        "internal_output": INTERNAL_ROOT / "TTA-MEMO",
    },
]

# Keep the paper-facing outputs threshold-free while keeping the chosen logic in code.
SELECTED_RULES = {
    "rule_based": "mean_minus_5pct",
    "inference_based": "mean_minus_5pct",
    "semantic_based": "mean_minus_5pct",
    "similarity_based": "mean_minus_5pct",
    "mlaas_based": "mean_minus_5pct",
    "CATTM": "mean_minus_16pct",
}


def run_internal_baseline(method: dict) -> None:
    method["internal_output"].mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(BASELINE_SCRIPT),
        "--summary-path",
        str(method["summary_path"]),
        "--qs-root",
        str(method["qs_root"]),
        "--qos-performing-root",
        str(method["qos_performing_root"]),
        "--output-root",
        str(method["internal_output"]),
    ]
    subprocess.run(cmd, check=True, cwd=str(RELEASE_ROOT))


def select_public_rows(method: dict) -> list[dict]:
    threshold_sweep_path = method["internal_output"] / "threshold_sweep.csv"
    df = pd.read_csv(threshold_sweep_path)

    rows = []
    for technique in TECHNIQUE_ORDER:
        rule = SELECTED_RULES[technique]
        match = df[
            (df["technique"] == technique)
            & (df["score_threshold_rule"] == rule)
            & (df["truth_threshold_rule"] == rule)
        ]
        if match.empty:
            raise RuntimeError(
                f"Missing threshold sweep row for {method['name']} / {technique} / {rule}"
            )
        row = match.iloc[0]
        rows.append(
            {
                "method": method["name"],
                "technique": technique,
                "accuracy": round(float(row["accuracy"]), 4),
                "precision": round(float(row["precision"]), 4),
                "recall": round(float(row["recall"]), 4),
                "f1": round(float(row["f1"]), 4),
            }
        )
    return rows


def main() -> None:
    INTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    PUBLIC_ROOT.mkdir(parents=True, exist_ok=True)

    combined_rows = []
    for method in METHODS:
        run_internal_baseline(method)
        public_rows = select_public_rows(method)
        combined_rows.extend(public_rows)

        method_df = pd.DataFrame(public_rows)
        method_df.to_csv(PUBLIC_ROOT / f"{method['name']}.csv", index=False)
        with (PUBLIC_ROOT / f"{method['name']}.json").open("w", encoding="utf-8") as handle:
            json.dump({"rows": public_rows}, handle, indent=2)

    combined_df = pd.DataFrame(combined_rows)
    combined_df.to_csv(PUBLIC_ROOT / "combined_final_table.csv", index=False)
    with (PUBLIC_ROOT / "combined_final_table.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "rows": combined_rows,
                "selected_rules": SELECTED_RULES,
            },
            handle,
            indent=2,
        )

    print(combined_df.to_string(index=False))


if __name__ == "__main__":
    main()
