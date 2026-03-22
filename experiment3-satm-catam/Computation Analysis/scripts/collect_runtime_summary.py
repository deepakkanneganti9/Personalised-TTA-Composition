import csv
import json
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRADITIONAL_ROOT = Path(__file__).resolve().parent

TRAINING_SUMMARY_PATH = PROJECT_ROOT / "FL_C" / "artifacts" / "federated_artifact_corrupt" / "clean_translate" / "summary.json"
CLIENT_PROFILE_CSV = TRADITIONAL_ROOT / "client_profiles" / "client_service_profiles.csv"
CLIENT_PROFILE_METADATA = TRADITIONAL_ROOT / "client_profiles" / "weights_metadata.json"
COMPOSITION_PROFILE_CSV = TRADITIONAL_ROOT / "profile" / "composition_qos_profiles.csv"
COMPOSITION_PROFILE_METADATA = TRADITIONAL_ROOT / "profile" / "weights_metadata.json"
COMPATIBILITY_SUMMARY = PROJECT_ROOT / "Compatibility results chunked" / "summary.json"
OUTPUT_DIR = TRADITIONAL_ROOT / "outputs" / "timing"


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sum_float(rows: List[Dict[str, str]], field: str) -> float:
    total = 0.0
    for row in rows:
        value = row.get(field)
        if value in (None, "", "n/a"):
            continue
        total += float(value)
    return total


def avg_float(rows: List[Dict[str, str]], field: str) -> float:
    values = []
    for row in rows:
        value = row.get(field)
        if value in (None, "", "n/a"):
            continue
        values.append(float(value))
    return sum(values) / len(values) if values else 0.0


def build_stage_payload() -> Dict:
    training = load_json(TRAINING_SUMMARY_PATH)
    client_meta = load_json(CLIENT_PROFILE_METADATA)
    client_rows = read_csv_rows(CLIENT_PROFILE_CSV)
    composition_meta = load_json(COMPOSITION_PROFILE_METADATA)
    composition_rows = read_csv_rows(COMPOSITION_PROFILE_CSV)
    compatibility = load_json(COMPATIBILITY_SUMMARY) if COMPATIBILITY_SUMMARY.exists() else {}

    client_stage = {
        "num_profiles": len(client_rows),
        "runtime_seconds": float(client_meta.get("runtime_seconds", 0.0)),
        "trial_evaluation_time_seconds_total": sum_float(client_rows, "trial_evaluation_time_seconds"),
        "reliability_profile_time_seconds_total": sum_float(client_rows, "reliability_profile_time_seconds"),
        "profile_build_time_seconds_total": sum_float(client_rows, "profile_build_time_seconds"),
        "trial_evaluation_time_seconds_avg": avg_float(client_rows, "trial_evaluation_time_seconds"),
        "reliability_profile_time_seconds_avg": avg_float(client_rows, "reliability_profile_time_seconds"),
        "profile_build_time_seconds_avg": avg_float(client_rows, "profile_build_time_seconds"),
    }

    composition_stage = {
        "num_profiles": len(composition_rows),
        "runtime_seconds": float(composition_meta.get("runtime_seconds", 0.0)),
        "trial_evaluation_time_seconds_total": sum_float(composition_rows, "trial_evaluation_time_seconds"),
        "reliability_profile_time_seconds_total": sum_float(composition_rows, "reliability_profile_time_seconds"),
        "profile_build_time_seconds_total": sum_float(composition_rows, "profile_build_time_seconds"),
        "trial_evaluation_time_seconds_avg": avg_float(composition_rows, "trial_evaluation_time_seconds"),
        "reliability_profile_time_seconds_avg": avg_float(composition_rows, "reliability_profile_time_seconds"),
        "profile_build_time_seconds_avg": avg_float(composition_rows, "profile_build_time_seconds"),
    }

    compatibility_stage = {
        "runtime_seconds": float(compatibility.get("runtime_seconds", 0.0)),
        "num_rows": int(compatibility.get("num_rows", 0)),
        "num_adapted_cases": int(compatibility.get("num_adapted_cases", 0)),
        "num_compositions": int(compatibility.get("num_compositions", 0)),
        "chunk_size": int(compatibility.get("chunk_size", 0)) if compatibility else 0,
    }

    training_stage = {
        "runtime_seconds": float(training.get("runtime_seconds", 0.0)),
        "num_clients": int(training.get("num_clients", 0)),
        "num_rounds": int(training.get("num_rounds", 0)),
        "local_epochs": int(training.get("local_epochs", 0)),
        "final_global_accuracy": float(training.get("final_global_accuracy", 0.0)),
        "corruption": training.get("corruption", ""),
    }

    total_offline_seconds = (
        training_stage["runtime_seconds"]
        + client_stage["runtime_seconds"]
        + composition_stage["runtime_seconds"]
        + compatibility_stage["runtime_seconds"]
    )

    return {
        "training": training_stage,
        "client_profile_generation": client_stage,
        "composition_profile_generation": composition_stage,
        "compatibility_profile_generation": compatibility_stage,
        "offline_total_runtime_seconds": total_offline_seconds,
    }


def write_outputs(payload: Dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "offline_runtime_summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = OUTPUT_DIR / "offline_runtime_summary.csv"
    rows = [
        {
            "stage": "training",
            "runtime_seconds": payload["training"]["runtime_seconds"],
            "num_items": payload["training"]["num_clients"],
            "details": f"clients={payload['training']['num_clients']};rounds={payload['training']['num_rounds']};epochs={payload['training']['local_epochs']};corruption={payload['training']['corruption']}",
        },
        {
            "stage": "client_profile_generation",
            "runtime_seconds": payload["client_profile_generation"]["runtime_seconds"],
            "num_items": payload["client_profile_generation"]["num_profiles"],
            "details": f"trial_total={payload['client_profile_generation']['trial_evaluation_time_seconds_total']:.4f};reliability_total={payload['client_profile_generation']['reliability_profile_time_seconds_total']:.4f};build_total={payload['client_profile_generation']['profile_build_time_seconds_total']:.4f}",
        },
        {
            "stage": "composition_profile_generation",
            "runtime_seconds": payload["composition_profile_generation"]["runtime_seconds"],
            "num_items": payload["composition_profile_generation"]["num_profiles"],
            "details": f"trial_total={payload['composition_profile_generation']['trial_evaluation_time_seconds_total']:.4f};reliability_total={payload['composition_profile_generation']['reliability_profile_time_seconds_total']:.4f};build_total={payload['composition_profile_generation']['profile_build_time_seconds_total']:.4f}",
        },
        {
            "stage": "compatibility_profile_generation",
            "runtime_seconds": payload["compatibility_profile_generation"]["runtime_seconds"],
            "num_items": payload["compatibility_profile_generation"]["num_rows"],
            "details": f"adapted_cases={payload['compatibility_profile_generation']['num_adapted_cases']};compositions={payload['compatibility_profile_generation']['num_compositions']};chunk_size={payload['compatibility_profile_generation']['chunk_size']}",
        },
        {
            "stage": "offline_total",
            "runtime_seconds": payload["offline_total_runtime_seconds"],
            "num_items": "",
            "details": "training+client_profile+composition_profile+compatibility",
        },
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "runtime_seconds", "num_items", "details"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    payload = build_stage_payload()
    write_outputs(payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
