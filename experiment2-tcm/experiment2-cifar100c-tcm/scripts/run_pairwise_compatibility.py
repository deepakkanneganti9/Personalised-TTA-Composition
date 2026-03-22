import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch.utils.data import DataLoader, Subset

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
OUTPUT_ROOT = PROJECT_ROOT / "composability-results-chunked"
FL_RUN_DIR = PROJECT_ROOT / "FL" / "artifacts" / "cifar100_fl_baseline_5clients_run_clean"
TTA_DIR = PROJECT_ROOT / "TTA techniques"
if str(TTA_DIR) not in sys.path:
    sys.path.insert(0, str(TTA_DIR))

from tta_techniques.tent_adapter import LocalCIFARCorruption  # noqa: E402
from FL.train_fedavg_cifar100 import build_model_for_dataset, load_run_dataset_name  # noqa: E402


DEFAULT_THRESHOLDS = [0.35, 0.4, 0.5, 0.6]
DEFAULT_PDAM_THRESHOLD = 0.1
DEFAULT_CHUNK_SIZE = 500
DEFAULT_BATCH_SIZE = 500
DEFAULT_SEVERITY = 1
DEFAULT_NUM_THREADS = 1


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_progress(path: Path, payload: Dict[str, object]) -> None:
    save_json(path, payload)


def load_model_state(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint["model_state"]


def build_model(state_dict):
    dataset_name = load_run_dataset_name(FL_RUN_DIR) if FL_RUN_DIR.exists() else "cifar100"
    model = build_model_for_dataset(dataset_name)
    model.load_state_dict(state_dict, strict=False)
    return model


def aggregate_states(adapted_state, composition_state, adapted_weight: float, composition_weight: float):
    total = adapted_weight + composition_weight
    aggregated = {}
    for key in adapted_state.keys():
        sample_tensor = adapted_state[key]
        if sample_tensor.is_floating_point():
            aggregated[key] = (
                adapted_state[key].detach().cpu().to(torch.float32) * (adapted_weight / total)
                + composition_state[key].detach().cpu().to(torch.float32) * (composition_weight / total)
            ).to(sample_tensor.dtype)
        else:
            aggregated[key] = (
                adapted_state[key].detach().cpu().to(torch.float64) * (adapted_weight / total)
                + composition_state[key].detach().cpu().to(torch.float64) * (composition_weight / total)
            ).round().to(sample_tensor.dtype)
    return aggregated


def get_client_sample_count(client_name: str) -> int:
    checkpoint = torch.load(FL_RUN_DIR / "clients" / f"{client_name}.pt", map_location="cpu")
    if "num_samples" in checkpoint:
        return int(checkpoint["num_samples"])
    return int(checkpoint["train_metrics"]["num_samples"])


def discover_client_names(fl_run_dir: Path) -> List[str]:
    clients_dir = fl_run_dir / "clients"
    if not clients_dir.exists():
        return []
    client_names = []
    for path in sorted(clients_dir.glob("client_*.pt")):
        client_names.append(path.stem)
    return client_names


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
    parser.add_argument("--cifar-c-root", type=Path, dest="cifar_c_root", default=PROJECT_ROOT / "Data" / "CIFAR-100-C")
    parser.add_argument("--cifar100-c-root", type=Path, dest="cifar_c_root")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--severity", type=int, default=DEFAULT_SEVERITY)
    parser.add_argument("--pdam-threshold", type=float, default=DEFAULT_PDAM_THRESHOLD)
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument("--output-tag", type=str)
    args = parser.parse_args()

    thread_count = max(1, int(args.num_threads))
    torch.set_num_threads(thread_count)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(thread_count)
        except RuntimeError:
            pass

    adapted_root = args.adapted_root.resolve()
    composition_root = args.composition_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    raw_rows_path = output_root / "raw_rows.csv"
    progress_path = output_root / "progress.json"
    summary_csv_path = output_root / "summary.csv"
    summary_json_path = output_root / "summary.json"
    threshold_metrics_csv_path = output_root / "threshold_metrics.csv"
    tagged_summary_csv_path = output_root / f"summary_{args.output_tag}.csv" if args.output_tag else None
    tagged_summary_json_path = output_root / f"summary_{args.output_tag}.json" if args.output_tag else None
    tagged_threshold_metrics_csv_path = (
        output_root / f"threshold_metrics_{args.output_tag}.csv" if args.output_tag else None
    )
    write_progress(
        progress_path,
        {
            "status": "starting",
            "adapted_root": str(adapted_root),
            "composition_root": str(composition_root),
            "output_root": str(output_root),
            "num_threads": thread_count,
        },
    )

    adapted_dirs = load_adapted_cases(adapted_root)
    compositions = load_compositions(composition_root)
    client_names = discover_client_names(FL_RUN_DIR)
    sample_counts = {client_name: get_client_sample_count(client_name) for client_name in client_names}
    write_progress(
        progress_path,
        {
            "status": "precomputing",
            "adapted_root": str(adapted_root),
            "composition_root": str(composition_root),
            "output_root": str(output_root),
            "num_threads": thread_count,
            "num_adapted_cases": len(adapted_dirs),
            "num_compositions": len(compositions),
            "raw_rows_path": str(raw_rows_path),
        },
    )

    adapted_static = {}
    for adapted_dir in adapted_dirs:
        adapted_case = adapted_dir.name
        adapted_client = "_".join(adapted_case.split("_")[:2])
        corruption = adapted_case.replace(f"{adapted_client}_", "", 1)
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
    raw_fieldnames = [
        "adapted_case",
        "adapted_client",
        "corruption",
        "severity",
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
        "composition_accuracy_on_chunk",
        "aggregated_accuracy_on_chunk",
        "ground_truth_composable",
    ]
    cached_datasets = {}
    cached_composition_outputs = {}
    cached_adapted_outputs = {}
    cached_aggregated_outputs = {}
    stop_requested = False
    with raw_rows_path.open("w", newline="", encoding="utf-8") as raw_handle:
        raw_writer = csv.DictWriter(raw_handle, fieldnames=raw_fieldnames)
        raw_writer.writeheader()
        raw_handle.flush()

        for adapted_case, adapted_info in adapted_static.items():
            if stop_requested:
                break
            corruption = adapted_info["corruption"]
            if corruption not in cached_datasets:
                cached_datasets[corruption] = LocalCIFARCorruption(
                    root=args.cifar_c_root.resolve(),
                    corruption=corruption,
                    dataset_name=load_run_dataset_name(FL_RUN_DIR),
                    split="test",
                    severity=args.severity,
                )
            dataset = cached_datasets[corruption]
            full_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

            if adapted_case not in cached_adapted_outputs:
                cached_adapted_outputs[adapted_case] = evaluate_model_outputs(adapted_info["model"], full_dataloader)
            adapted_outputs = cached_adapted_outputs[adapted_case]

            for chunk_id, start, end in chunk_ranges(len(dataset), args.chunk_size):
                if stop_requested:
                    break
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

                    row = {
                        "adapted_case": adapted_case,
                        "adapted_client": adapted_info["client_name"],
                        "corruption": corruption,
                        "severity": args.severity,
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
                    rows.append(row)
                    raw_writer.writerow(
                        {
                            "adapted_case": row["adapted_case"],
                            "adapted_client": row["adapted_client"],
                            "corruption": row["corruption"],
                            "severity": row["severity"],
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
                            "composition_accuracy_on_chunk": row["composition_accuracy_on_chunk"],
                            "aggregated_accuracy_on_chunk": row["aggregated_accuracy_on_chunk"],
                            "ground_truth_composable": row["ground_truth_composable"],
                        }
                    )
                    raw_handle.flush()
                    write_progress(
                        progress_path,
                        {
                            "status": "running",
                            "rows_completed": len(rows),
                            "max_rows": args.max_rows,
                            "last_completed": {
                                "adapted_case": row["adapted_case"],
                                "composition_name": row["composition_name"],
                                "chunk_id": row["chunk_id"],
                            },
                            "raw_rows_path": str(raw_rows_path),
                        },
                    )
                    if args.max_rows is not None and len(rows) >= args.max_rows:
                        stop_requested = True
                        break

    finalized_rows = finalize_ttaas_records(rows, threshold=DEFAULT_TTAAS_THRESHOLD)
    finalized_rows = add_threshold_columns(finalized_rows, thresholds=args.thresholds)

    threshold_summaries = []
    for threshold in args.thresholds:
        metrics = classification_metrics(finalized_rows, threshold)
        metrics["num_samples"] = len(finalized_rows)
        threshold_summaries.append(metrics)

    save_json(
        summary_json_path,
        {
            "status": "completed",
            "num_rows": len(finalized_rows),
            "chunk_size": args.chunk_size,
            "num_chunks_per_corruption": len(chunk_ranges(10000, args.chunk_size)),
            "num_adapted_cases": len(adapted_dirs),
            "num_compositions": len(compositions),
            "severity": args.severity,
            "thresholds": args.thresholds,
            "threshold_metrics": threshold_summaries,
        },
    )
    write_progress(
        progress_path,
        {
            "status": "completed",
            "rows_completed": len(finalized_rows),
            "max_rows": args.max_rows,
            "summary_csv": str(summary_csv_path),
            "summary_json": str(summary_json_path),
            "threshold_metrics_csv": str(threshold_metrics_csv_path),
            "summary_csv_tagged": str(tagged_summary_csv_path) if tagged_summary_csv_path else None,
            "summary_json_tagged": str(tagged_summary_json_path) if tagged_summary_json_path else None,
            "threshold_metrics_csv_tagged": (
                str(tagged_threshold_metrics_csv_path) if tagged_threshold_metrics_csv_path else None
            ),
        },
    )

    detail_fieldnames = [
        "adapted_case",
        "adapted_client",
        "corruption",
        "severity",
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

    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fieldnames)
        writer.writeheader()
        for row in finalized_rows:
            csv_row = {
                "adapted_case": row["adapted_case"],
                "adapted_client": row["adapted_client"],
                "corruption": row["corruption"],
                "severity": row["severity"],
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

    with threshold_metrics_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["threshold", "accuracy", "precision", "recall", "f1", "tp", "fp", "fn", "tn", "num_samples"],
        )
        writer.writeheader()
        for row in threshold_summaries:
            writer.writerow(row)

    if tagged_summary_csv_path is not None:
        tagged_summary_csv_path.write_text(summary_csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    if tagged_summary_json_path is not None:
        tagged_summary_json_path.write_text(summary_json_path.read_text(encoding="utf-8"), encoding="utf-8")
    if tagged_threshold_metrics_csv_path is not None:
        tagged_threshold_metrics_csv_path.write_text(
            threshold_metrics_csv_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "num_rows": len(finalized_rows),
                "num_adapted_cases": len(adapted_dirs),
                "num_compositions": len(compositions),
                "chunk_size": args.chunk_size,
                "severity": args.severity,
                "thresholds": args.thresholds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
