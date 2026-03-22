import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FL_C.train_fedavg_mnist import MNISTCNN
from FL_C.generate_client_service_profiles import LocalMNISTC
OUTPUT_ROOT = ROOT / "outputs" / "fair_mixed"
CLIENT_PROFILES_CSV = ROOT / "client_profiles" / "client_service_profiles.csv"
COMPOSITION_PROFILES_CSV = ROOT / "profile" / "composition_qos_profiles.csv"
MNIST_C_ROOT = PROJECT_ROOT / "data" / "mnist_c" / "mnist_c"
DEFAULT_CORRUPTIONS = ["fog", "translate", "stripe", "zigzag", "spatter"]
METHOD_FILES = {
    "semantic": ROOT / "outputs" / "semantic" / "semantic_substitution_results.csv",
    "context": ROOT / "outputs" / "context" / "contextual_substitution_results.csv",
    "mlaas": ROOT / "outputs" / "mlaas" / "mlaas_substitution_results.csv",
    "tta": ROOT / "outputs" / "tta" / "tta_substitution_results.csv",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def parse_maybe_json_list(value: str) -> List[str]:
    if not value:
        return []
    stripped = value.strip()
    if stripped.startswith("["):
        return list(json.loads(stripped))
    return [stripped]


def clean_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


def load_clean_test_dataset() -> Dataset:
    return datasets.MNIST(root=PROJECT_ROOT / "data", train=False, download=False, transform=clean_transform())


def sample_subset(dataset: Dataset, num_samples: int, seed: int) -> Dataset:
    if num_samples <= 0 or len(dataset) <= num_samples:
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=num_samples, replace=False))
    return Subset(dataset, indices.tolist())


def build_mixed_dataset(corruption: str, clean_fraction: float, total_samples: int, seed: int) -> Dataset:
    clean_count = int(round(total_samples * clean_fraction))
    corrupt_count = total_samples - clean_count
    parts: List[Dataset] = []
    if clean_count > 0:
        parts.append(sample_subset(load_clean_test_dataset(), clean_count, seed))
    if corrupt_count > 0:
        corrupt_dataset = LocalMNISTC(MNIST_C_ROOT, corruption=corruption, split="test")
        parts.append(sample_subset(corrupt_dataset, corrupt_count, seed + 1000))
    return ConcatDataset(parts)


def load_model_from_checkpoint(path: Path) -> MNISTCNN:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model = MNISTCNN()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_state(path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint["model_state"] if "model_state" in checkpoint else checkpoint


def evaluate_model(model: MNISTCNN, dataset: Dataset, batch_size: int) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images).argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    return float(correct / total if total else 0.0)


def fedavg_aggregate(paths: List[Path], sample_counts: List[int]) -> MNISTCNN:
    states = [load_state(path) for path in paths]
    total = float(sum(sample_counts))
    aggregated = {}
    for key in states[0].keys():
        aggregated[key] = sum(state[key] * (count / total) for state, count in zip(states, sample_counts))
    model = MNISTCNN()
    model.load_state_dict(aggregated, strict=False)
    model.eval()
    return model


def protected_aggregate(client_states: Dict[str, Dict[str, torch.Tensor]], weights: Dict[str, float]) -> MNISTCNN:
    param_names = sorted({param_name for state in client_states.values() for param_name in state.keys()})
    aggregated = {}
    for param_name in param_names:
        available = [client_name for client_name, state in client_states.items() if param_name in state]
        if not available:
            continue
        if param_name.endswith("num_batches_tracked"):
            aggregated[param_name] = client_states[available[0]][param_name].clone()
            continue
        total_weight = sum(weights[client_name] for client_name in available)
        aggregated[param_name] = sum(
            client_states[client_name][param_name] * (weights[client_name] / total_weight)
            for client_name in available
        )
    model = MNISTCNN()
    model.load_state_dict(aggregated, strict=False)
    model.eval()
    return model


def aggregate_states_with_weights(states: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    aggregated = {}
    for key in states[0].keys():
        if key.endswith("num_batches_tracked"):
            aggregated[key] = states[0][key].clone()
            continue
        acc = None
        for state, weight in zip(states, weights):
            term = state[key] * weight
            acc = term if acc is None else acc + term
        aggregated[key] = acc
    return aggregated


def state_key_groups(state: Dict[str, torch.Tensor]):
    bn_keys = []
    weight_keys = []
    for key in state.keys():
        if key.endswith("num_batches_tracked"):
            continue
        if key.endswith("running_mean") or key.endswith("running_var") or key in {
            "features.1.weight",
            "features.1.bias",
            "features.5.weight",
            "features.5.bias",
        }:
            bn_keys.append(key)
        else:
            weight_keys.append(key)
    return weight_keys, bn_keys


def flatten_subset(state: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
    parts = [state[key].detach().cpu().reshape(-1).float() for key in keys]
    if not parts:
        return torch.zeros(1)
    return torch.cat(parts)


def pairwise_preference_weights(
    performing_state: Dict[str, torch.Tensor],
    tta_state: Dict[str, torch.Tensor],
    keys: List[str],
    beta: float,
):
    perf_vec = flatten_subset(performing_state, keys)
    tta_vec = flatten_subset(tta_state, keys)
    distance = torch.norm(perf_vec - tta_vec, p=2).item()
    tta_score = float(torch.exp(torch.tensor(-beta * distance)).item())
    perf_weight = 1.0 / (1.0 + tta_score)
    tta_weight = tta_score / (1.0 + tta_score)
    return perf_weight, tta_weight, distance


def family_specific_hybrid_aggregation(
    performing_state: Dict[str, torch.Tensor],
    tta_state: Dict[str, torch.Tensor],
    weight_pref,
    bn_pref,
) -> MNISTCNN:
    weight_keys, bn_keys = state_key_groups(performing_state)
    out = {}
    weight_perf, weight_tta = weight_pref
    bn_perf, bn_tta = bn_pref
    for name in performing_state.keys():
        if name.endswith("num_batches_tracked"):
            out[name] = performing_state[name].clone()
        elif name in weight_keys:
            out[name] = weight_perf * performing_state[name] + weight_tta * tta_state[name]
        elif name in bn_keys:
            out[name] = bn_perf * performing_state[name] + bn_tta * tta_state[name]
        else:
            out[name] = performing_state[name].clone()
    model = MNISTCNN()
    model.load_state_dict(out, strict=False)
    model.eval()
    return model


def summarize(rows: List[Dict[str, object]], threshold: float) -> Dict[str, float]:
    return {
        "Healthy": mean(float(row["healthy_accuracy"]) for row in rows),
        "Degraded": mean(float(row["degraded_accuracy"]) for row in rows),
        "After Substitution": mean(float(row["after_adaptation_accuracy"]) for row in rows),
        "Recovery Ratio": mean(float(row["recovery_ratio"]) for row in rows),
        "Requirement Satisfaction": sum(float(row["after_adaptation_accuracy"]) >= threshold for row in rows) / len(rows),
        "Substitution Success": sum(float(row["after_adaptation_accuracy"]) > float(row["degraded_accuracy"]) for row in rows) / len(rows),
        "False Accept: No Improvement": sum(float(row["after_adaptation_accuracy"]) <= float(row["degraded_accuracy"]) for row in rows) / len(rows),
        "False Accept: Below Requirement": sum(float(row["after_adaptation_accuracy"]) < threshold for row in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-samples", type=int, default=1000)
    parser.add_argument("--clean-ratios", nargs="+", type=float, default=[0.8, 0.5])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corruptions", nargs="+", default=DEFAULT_CORRUPTIONS)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(1)

    client_rows = read_csv(CLIENT_PROFILES_CSV)
    composition_rows = read_csv(COMPOSITION_PROFILES_CSV)
    client_by_service = {row["service_id"]: row for row in client_rows}
    clean_final_by_client = {
        row["original_client_id"]: row
        for row in client_rows
        if row["category"] == "clean" and row["checkpoint_kind"] == "final_snapshot"
    }
    composition_by_name = {row["composition_name"]: row for row in composition_rows}

    all_ratio_summaries = {}
    for clean_ratio in args.clean_ratios:
        ratio_label = f"{int(round(clean_ratio * 100))}clean_{int(round((1.0 - clean_ratio) * 100))}corrupt"
        mixed_cache = {
            corruption: build_mixed_dataset(corruption, clean_ratio, args.total_samples, args.seed + idx)
            for idx, corruption in enumerate(args.corruptions)
        }

        ratio_rows = []
        overall_rows = []
        for method, csv_path in METHOD_FILES.items():
            method_rows = read_csv(csv_path)
            method_eval_rows = []
            for row in method_rows:
                corruption = row["corruption_category"]
                dataset = mixed_cache[corruption]
                composition_row = composition_by_name[row["composition_name"]]
                composition_model = load_model_from_checkpoint(Path(composition_row["weights_file"]))
                degraded_accuracy = evaluate_model(composition_model, dataset, args.batch_size)

                if method in {"semantic", "context", "mlaas"}:
                    remaining_clients = json.loads(row["remaining_clients"])
                    substitute_services = [client_by_service[service_id] for service_id in parse_maybe_json_list(row["recommended_substitute_service"])]
                    paths = [Path(clean_final_by_client[client_name]["weights_file"]) for client_name in remaining_clients]
                    counts = [int(clean_final_by_client[client_name]["functional_num_samples"]) for client_name in remaining_clients]
                    for substitute_service in substitute_services:
                        paths.append(Path(substitute_service["weights_file"]))
                        counts.append(int(substitute_service["functional_num_samples"]))
                    recomposed_model = fedavg_aggregate(paths, counts)
                else:
                    remaining_clients = json.loads(row["remaining_clients"])
                    failing_clients = parse_maybe_json_list(row["failing_client"])
                    failing_client = failing_clients[0]
                    adapted_path = ROOT / "tta_adapted_clients" / f"{failing_client}_{corruption}" / "adapted_client_model.pt"
                    tta_state = load_state(adapted_path)
                    if remaining_clients:
                        performing_states = [
                            load_state(Path(clean_final_by_client[client_name]["weights_file"]))
                            for client_name in remaining_clients
                        ]
                        performing_state = aggregate_states_with_weights(
                            performing_states,
                            [1.0 / len(performing_states)] * len(performing_states),
                        )
                        weight_keys, bn_keys = state_key_groups(performing_state)
                        model_pref = pairwise_preference_weights(performing_state, tta_state, weight_keys, beta=2.0)
                        bn_pref = pairwise_preference_weights(performing_state, tta_state, bn_keys, beta=2.0)
                        recomposed_model = family_specific_hybrid_aggregation(
                            performing_state,
                            tta_state,
                            weight_pref=(model_pref[0], model_pref[1]),
                            bn_pref=(bn_pref[0], bn_pref[1]),
                        )
                    else:
                        model = MNISTCNN()
                        model.load_state_dict(tta_state, strict=False)
                        model.eval()
                        recomposed_model = model

                after_accuracy = evaluate_model(recomposed_model, dataset, args.batch_size)
                healthy_accuracy = float(row["before_accuracy"])
                recovery_ratio = 0.0
                if healthy_accuracy > degraded_accuracy:
                    recovery_ratio = max(0.0, after_accuracy - degraded_accuracy) / (healthy_accuracy - degraded_accuracy)

                result_row = {
                    "method": method,
                    "ratio_label": ratio_label,
                    "clean_ratio": clean_ratio,
                    "corruption_ratio": 1.0 - clean_ratio,
                    "composition_name": row["composition_name"],
                    "corruption_category": corruption,
                    "healthy_accuracy": healthy_accuracy,
                    "degraded_accuracy": degraded_accuracy,
                    "after_adaptation_accuracy": after_accuracy,
                    "recovery_ratio": recovery_ratio,
                }
                ratio_rows.append(result_row)
                method_eval_rows.append(result_row)

            summary = summarize(method_eval_rows, args.threshold)
            summary["Method"] = method.capitalize() if method != "mlaas" else "MLaaS"
            summary["ratio_label"] = ratio_label
            overall_rows.append(summary)

        write_csv(OUTPUT_ROOT / f"{ratio_label}_case_results.csv", ratio_rows)
        write_csv(OUTPUT_ROOT / f"{ratio_label}_overall_summary.csv", overall_rows)
        all_ratio_summaries[ratio_label] = overall_rows

    write_json(
        OUTPUT_ROOT / "mixed_fairness_metadata.json",
        {
            "total_samples": args.total_samples,
            "clean_ratios": args.clean_ratios,
            "threshold": args.threshold,
            "corruptions": args.corruptions,
            "summaries": all_ratio_summaries,
        },
    )


if __name__ == "__main__":
    main()
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
