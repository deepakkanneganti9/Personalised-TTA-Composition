import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
INPUT = RESULTS_DIR / "catm_filtered_results_50_50.csv"
OUTPUT_JSON = RESULTS_DIR / "catm_packet_summary.json"


if __name__ == "__main__":
    rows = list(csv.DictReader(INPUT.open()))
    summary = {
        "rows": len(rows),
        "composition_lengths": sorted({int(row["composition_length"]) for row in rows}),
        "corruptions": sorted({row["corruption"] for row in rows}),
        "clean_samples": sorted({int(row["clean_samples"]) for row in rows}),
        "corruption_samples": sorted({int(row["corruption_samples"]) for row in rows}),
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
