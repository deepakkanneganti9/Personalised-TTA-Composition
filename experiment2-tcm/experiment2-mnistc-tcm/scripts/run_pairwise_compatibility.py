import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch.utils.data import DataLoader

from compatibility_metrics import (
    DEFAULT_TTAAS_THRESHOLD,
    add_threshold_columns,
    classification_metrics,
    compute_all_metrics_from_files,
    compute_pdam,
    finalize_ttaas_records,
)


PROJECT_ROOT = Path(__file__).resolve().parent
ADAPTED_ROOT = PROJECT_ROOT / "TTA techniques" / "artifacts"
COMPOSITION_ROOT = PROJECT_ROOT / "Performing composition" / "artifacts_expanded"
OUTPUT_ROOT = PROJECT_ROOT / "Compatibility results chunked"
FL_RUN_DIR = PROJECT_ROOT / "FL" / "artifacts" / "mnist_fl_baseline_5clients_run"
TTA_DIR = PROJECT_ROOT / "TTA techniques"
if str(TTA_DIR) not in sys.path:
    sys.path.insert(0, str(TTA_DIR))

from tta_techniques.tent_grad_adapter import LocalMNISTC, MNISTCNN  # noqa: E402


DEFAULT_THRESHOLDS = [0.35, 0.4, 0.5, 0.6]
DEFAULT_PDAM_THRESHOLD = 0.1
DEFAULT_CHUNK_SIZE = 500


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_model_state(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint["model_state"]


def build_model(state_dict):
    model = MNISTCNN()
    model.load_state_dict(state_dict, strict=False)
    return model


def aggregate_states(adapted_state, composition_state, adapted_weight: float, composition_weight: float):
    total = adapted_weight + composition_weight
    aggregated = {}
    for key in adapted_state.keys():
        aggregated[key] = (
            adapted_state[key].detach().cpu() * (adapted_weight / total)
            + composition_state[key].detach().cpu() * (composition_weight / total)
        )
    return aggregated


def get_client_sample_count(client_name: str) -> int:
    checkpoint = torch.load(FL_RUN_DIR / "clients" / f"{client_name}.pt", map_location="cpu")
    if "num_samples" in checkpoint:
        return int(checkpoint["num_samples"])
    return int(checkpoint["train_metrics"]["num_samples"])


def adapted_case_sort_key(path: Path) -> Tuple[int, str]:
    name = path.name
    client_number = int(name.split("_")[1])
    return (client_number, name)


def chunk_ranges(total_examples: int, chunk_size: int) -> List[Tuple[int, int, int]]:
    ranges = []
    chunk_id = 0
    for start in range(0, total_examples, chunk_size):
        end = min(start + chunk_size, total_examples)
        ranges.append((chunk_id, start, end))
        chunk_id += 1
    return ranges


def load_adapted_cases(adapted_root: Path) -> List[Path]:
    return sorted(
        [path for path in adapted_root.iterdir() if path.is_dir() and path.name.startswith("client_")],
        key=adapted_case_sort_key,
    )


def load_artifact_config(artifact_dir: Path) -> Dict[str, object]:
    config_path = artifact_dir / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text())


def load_compositions(composition_root: Path) -> List[Dict[str, object]]:
    compositions = []
    for path in sorted(composition_root.iterdir()):
        if not path.is_dir():
            continue
        summary_path = path / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        compositions.append(
            {
                "name": path.name,
                "dir": path,
                "client_names": summary["client_names"],
            }
        )
    return compositions


@torch.no_grad()
def evaluate_model_outputs(model: torch.nn.Module, dataloader: DataLoader):
    model.eval()
    probabilities = []
    correctness = []
    for images, labels in dataloader:
        logits = model(images)
        batch_probabilities = logits.softmax(dim=1).cpu()
        batch_predictions = logits.argmax(dim=1).cpu()
        probabilities.append(batch_probabilities)
        correctness.append((batch_predictions == labels.cpu()).float())
    return {
        "probabilities": torch.cat(probabilities, dim=0),
        "correctness": torch.cat(correctness, dim=0),
    }


def chunk_metrics_from_outputs(outputs: Dict[str, torch.Tensor], start: int, end: int) -> Dict[str, object]:
    probability_slice = outputs["probabilities"][start:end]
    correctness_slice = outputs["correctness"][start:end]
    return {
        "accuracy": float(correctness_slice.mean().item()),
        "prediction_distribution": probability_slice.mean(dim=0).tolist(),
    }


def threshold_field_name(prefix: str, threshold: float) -> str:
    return f"{prefix}_{str(threshold).replace('.', '_')}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapted-root", type=Path, default=ADAPTED_ROOT)
    parser.add_argument("--composition-root", type=Path, default=COMPOSITION_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--mnist-c-root", type=Path, default=PROJECT_ROOT / "data" / "mnist_c")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--pdam-threshold", type=float, default=DEFAULT_PDAM_THRESHOLD)
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    adapted_root = args.adapted_root.resolve()
    composition_root = args.composition_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    adapted_dirs = load_adapted_cases(adapted_root)
    compositions = load_compositions(composition_root)
    sample_counts = {f"client_{index}": get_client_sample_count(f"client_{index}") for index in range(5)}

    adapted_static = {}
    for adapted_dir in adapted_dirs:
        adapted_case = adapted_dir.name
        artifact_config = load_artifact_config(adapted_dir)
        adapted_client = artifact_config.get("client_name", "_".join(adapted_case.split("_")[:2]))
        corruption = artifact_config.get("corruption", adapted_case.replace(f"{adapted_client}_", "", 1))
        static_metrics = compute_all_metrics_from_files(
            adapted_wdm_weights_path=adapted_dir / "wdm_adapted_weights.json",
            adapted_ucs_updates_path=adapted_dir / "ucs_adapted_layer_updates.json",
            adapted_bn_stats_path=adapted_dir / "bndas_adapted_bn_mean_var.json",
            adapted_bnuas_affine_path=adapted_dir / "bnuas_adapted_bn_gamma_beta.json",
        )
        adapted_static[adapted_case] = {
            "dir": adapted_dir,
            "client_name": adapted_client,
            "corruption": corruption,
            "state": load_model_state(adapted_dir / "adapted_client_model.pt"),
            "model": build_model(load_model_state(adapted_dir / "adapted_client_model.pt")),
            "metrics": static_metrics,
            "sample_count": sample_counts[adapted_client],
        }

    composition_static = {}
    for composition in compositions:
        composition_name = composition["name"]
        composition_dir = composition["dir"]
        state = load_model_state(composition_dir / "composition_model.pt")
        static_metrics = compute_all_metrics_from_files(
            reference_wdm_weights_path=composition_dir / "wdm_composition_weights.json",
            reference_ucs_updates_path=composition_dir / "ucs_composition_layer_updates.json",
            reference_bn_stats_path=composition_dir / "bndas_composition_bn_mean_var.json",
            reference_bnuas_affine_path=composition_dir / "bnuas_composition_bn_gamma_beta.json",
        )
        composition_static[composition_name] = {
            "dir": composition_dir,
            "client_names": composition["client_names"],
            "state": state,
            "model": build_model(state),
            "metrics": static_metrics,
            "sample_count": sum(sample_counts[client_name] for client_name in composition["client_names"]),
        }

    pair_static_metrics = {}
    for adapted_case, adapted_info in adapted_static.items():
        for composition_name, composition_info in composition_static.items():
            pair_static_metrics[(adapted_case, composition_name)] = compute_all_metrics_from_files(
                adapted_wdm_weights_path=adapted_info["dir"] / "wdm_adapted_weights.json",
                reference_wdm_weights_path=composition_info["dir"] / "wdm_composition_weights.json",
                adapted_ucs_updates_path=adapted_info["dir"] / "ucs_adapted_layer_updates.json",
                reference_ucs_updates_path=composition_info["dir"] / "ucs_composition_layer_updates.json",
                adapted_bn_stats_path=adapted_info["dir"] / "bndas_adapted_bn_mean_var.json",
                reference_bn_stats_path=composition_info["dir"] / "bndas_composition_bn_mean_var.json",
                adapted_bnuas_affine_path=adapted_info["dir"] / "bnuas_adapted_bn_gamma_beta.json",
                reference_bnuas_affine_path=composition_info["dir"] / "bnuas_composition_bn_gamma_beta.json",
            )

    rows = []
    cached_datasets = {}
    cached_composition_outputs = {}
    cached_adapted_outputs = {}
    cached_aggregated_outputs = {}

    for adapted_case, adapted_info in adapted_static.items():
        corruption = adapted_info["corruption"]
        if corruption not in cached_datasets:
            cached_datasets[corruption] = LocalMNISTC(
                root=args.mnist_c_root.resolve(),
                corruption=corruption,
                split="test",
            )
        dataset = cached_datasets[corruption]
        full_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        if adapted_case not in cached_adapted_outputs:
            cached_adapted_outputs[adapted_case] = evaluate_model_outputs(adapted_info["model"], full_dataloader)
        adapted_outputs = cached_adapted_outputs[adapted_case]

        for chunk_id, start, end in chunk_ranges(len(dataset), args.chunk_size):
            adapted_chunk_eval = chunk_metrics_from_outputs(adapted_outputs, start, end)
            for composition_name, composition_info in composition_static.items():
                composition_eval_key = (composition_name, corruption)
                if composition_eval_key not in cached_composition_outputs:
                    cached_composition_outputs[composition_eval_key] = evaluate_model_outputs(
                        composition_info["model"], full_dataloader
                    )
                composition_outputs = cached_composition_outputs[composition_eval_key]
                composition_chunk_eval = chunk_metrics_from_outputs(composition_outputs, start, end)

                pdam_result = compute_pdam(
                    adapted_prediction_distribution=adapted_chunk_eval["prediction_distribution"],
                    reference_prediction_distribution=composition_chunk_eval["prediction_distribution"],
                    threshold=args.pdam_threshold,
                )

                aggregated_eval_key = (adapted_case, composition_name)
                if aggregated_eval_key not in cached_aggregated_outputs:
                    aggregated_state = aggregate_states(
                        adapted_state=adapted_info["state"],
                        composition_state=composition_info["state"],
                        adapted_weight=float(adapted_info["sample_count"]),
                        composition_weight=float(composition_info["sample_count"]),
                    )
                    cached_aggregated_outputs[aggregated_eval_key] = evaluate_model_outputs(
                        build_model(aggregated_state), full_dataloader
                    )
                aggregated_outputs = cached_aggregated_outputs[aggregated_eval_key]
                aggregated_accuracy = chunk_metrics_from_outputs(aggregated_outputs, start, end)["accuracy"]

                raw_metrics = {
                    "wdm": pair_static_metrics[(adapted_case, composition_name)]["wdm"],
                    "ucs": pair_static_metrics[(adapted_case, composition_name)]["ucs"],
                    "bnuas": pair_static_metrics[(adapted_case, composition_name)]["bnuas"],
                    "bndas": pair_static_metrics[(adapted_case, composition_name)]["bndas"],
                    "pdam": pdam_result["pdam"],
                }

                rows.append(
                    {
                        "adapted_case": adapted_case,
                        "adapted_client": adapted_info["client_name"],
                        "corruption": corruption,
                        "chunk_id": chunk_id,
                        "chunk_start": start,
                        "chunk_end": end,
                        "chunk_size": end - start,
                        "composition_name": composition_name,
                        "performing_clients": ",".join(composition_info["client_names"]),
                        "performing_size": len(composition_info["client_names"]),
                        "raw_metrics": raw_metrics,
                        "wasserstein_distance": pdam_result["wasserstein_distance"],
                        "composition_accuracy_on_chunk": composition_chunk_eval["accuracy"],
                        "aggregated_accuracy_on_chunk": aggregated_accuracy,
                        "ground_truth_composable": bool(aggregated_accuracy > composition_chunk_eval["accuracy"]),
                    }
                )

    finalized_rows = finalize_ttaas_records(rows, threshold=DEFAULT_TTAAS_THRESHOLD)
    finalized_rows = add_threshold_columns(finalized_rows, thresholds=args.thresholds)

    threshold_summaries = []
    for threshold in args.thresholds:
        metrics = classification_metrics(finalized_rows, threshold)
        metrics["num_samples"] = len(finalized_rows)
        threshold_summaries.append(metrics)

    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    summary_json_name = f"summary{suffix}.json"
    summary_csv_name = f"summary{suffix}.csv"
    threshold_metrics_name = f"threshold_metrics{suffix}.csv"

    save_json(
        output_root / summary_json_name,
        {
            "num_rows": len(finalized_rows),
            "chunk_size": args.chunk_size,
            "num_chunks_per_corruption": len(chunk_ranges(10000, args.chunk_size)),
            "num_adapted_cases": len(adapted_dirs),
            "num_compositions": len(compositions),
            "thresholds": args.thresholds,
            "threshold_metrics": threshold_summaries,
        },
    )

    detail_fieldnames = [
        "adapted_case",
        "adapted_client",
        "corruption",
        "chunk_id",
        "chunk_start",
        "chunk_end",
        "chunk_size",
        "composition_name",
        "performing_clients",
        "performing_size",
        "raw_wdm",
        "raw_ucs",
        "raw_bnuas",
        "raw_bndas",
        "raw_pdam",
        "wasserstein_distance",
        "aligned_wdm",
        "aligned_ucs",
        "aligned_bnuas",
        "aligned_bndas",
        "aligned_pdam",
        "scaled_wdm",
        "scaled_ucs",
        "scaled_bnuas",
        "scaled_bndas",
        "scaled_pdam",
        "ttaas",
        "composition_accuracy_on_chunk",
        "aggregated_accuracy_on_chunk",
        "ground_truth_composable",
    ]
    for threshold in args.thresholds:
        detail_fieldnames.append(threshold_field_name("predicted_at", threshold))
        detail_fieldnames.append(threshold_field_name("correct_at", threshold))

    with (output_root / summary_csv_name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fieldnames)
        writer.writeheader()
        for row in finalized_rows:
            csv_row = {
                "adapted_case": row["adapted_case"],
                "adapted_client": row["adapted_client"],
                "corruption": row["corruption"],
                "chunk_id": row["chunk_id"],
                "chunk_start": row["chunk_start"],
                "chunk_end": row["chunk_end"],
                "chunk_size": row["chunk_size"],
                "composition_name": row["composition_name"],
                "performing_clients": row["performing_clients"],
                "performing_size": row["performing_size"],
                "raw_wdm": row["raw_metrics"]["wdm"],
                "raw_ucs": row["raw_metrics"]["ucs"],
                "raw_bnuas": row["raw_metrics"]["bnuas"],
                "raw_bndas": row["raw_metrics"]["bndas"],
                "raw_pdam": row["raw_metrics"]["pdam"],
                "wasserstein_distance": row["wasserstein_distance"],
                "aligned_wdm": row["aligned_metrics"]["wdm"],
                "aligned_ucs": row["aligned_metrics"]["ucs"],
                "aligned_bnuas": row["aligned_metrics"]["bnuas"],
                "aligned_bndas": row["aligned_metrics"]["bndas"],
                "aligned_pdam": row["aligned_metrics"]["pdam"],
                "scaled_wdm": row["scaled_metrics"]["wdm"],
                "scaled_ucs": row["scaled_metrics"]["ucs"],
                "scaled_bnuas": row["scaled_metrics"]["bnuas"],
                "scaled_bndas": row["scaled_metrics"]["bndas"],
                "scaled_pdam": row["scaled_metrics"]["pdam"],
                "ttaas": row["ttaas"],
                "composition_accuracy_on_chunk": row["composition_accuracy_on_chunk"],
                "aggregated_accuracy_on_chunk": row["aggregated_accuracy_on_chunk"],
                "ground_truth_composable": row["ground_truth_composable"],
            }
            for threshold in args.thresholds:
                csv_row[threshold_field_name("predicted_at", threshold)] = row[
                    threshold_field_name("predicted_at", threshold)
                ]
                csv_row[threshold_field_name("correct_at", threshold)] = row[
                    threshold_field_name("correct_at", threshold)
                ]
            writer.writerow(csv_row)

    with (output_root / threshold_metrics_name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["threshold", "accuracy", "precision", "recall", "f1", "tp", "fp", "fn", "tn", "num_samples"],
        )
        writer.writeheader()
        for row in threshold_summaries:
            writer.writerow(row)

    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "num_rows": len(finalized_rows),
                "num_adapted_cases": len(adapted_dirs),
                "num_compositions": len(compositions),
                "chunk_size": args.chunk_size,
                "thresholds": args.thresholds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
