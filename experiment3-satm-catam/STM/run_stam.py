import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
INPUT = RESULTS_DIR / "reviewer_composition_multi_tta_v2.csv"
OUTPUT_JSON = RESULTS_DIR / "stam_packet_summary.json"


if __name__ == "__main__":
    rows = list(csv.DictReader(INPUT.open()))
    summary = {
        "rows": len(rows),
        "methods": sorted({row["tta_method"] for row in rows}),
        "lengths": sorted({int(row["performing_length"]) for row in rows}),
        "clean_samples": sorted({int(row["clean_samples"]) for row in rows}),
        "corruption_samples": sorted({int(row["corruption_samples"]) for row in rows}),
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
