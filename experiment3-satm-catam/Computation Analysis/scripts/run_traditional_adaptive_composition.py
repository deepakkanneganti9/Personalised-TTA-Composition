import argparse
import csv
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FL_C.train_fedavg_mnist import MNISTCNN
from FL_C.generate_client_service_profiles import LocalMNISTC


ROOT = Path(__file__).resolve().parent
CLIENT_PROFILES_CSV = ROOT / "client_profiles" / "client_service_profiles.csv"
COMPOSITION_PROFILES_CSV = ROOT / "profile" / "composition_qos_profiles.csv"
OUTPUTS_ROOT = ROOT / "outputs"
MNIST_C_ROOT = PROJECT_ROOT / "data" / "mnist_c" / "mnist_c"
DEFAULT_CORRUPTIONS = ["fog", "translate", "stripe", "zigzag", "spatter"]
FL_RUN_DIR = PROJECT_ROOT / "FL" / "artifacts" / "mnist_fl_baseline_5clients_run"
TTA_ARTIFACTS_ROOT = ROOT / "tta_adapted_clients"
TTA_COMPAT_ROOT = ROOT / "outputs" / "tta_compatibility"
LEGACY_TTA_ARTIFACTS_ROOT = PROJECT_ROOT / "TTA techniques" / "artifacts"


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tent_grad_module = load_module(
    PROJECT_ROOT / "TTA techniques" / "tta_techniques" / "tent_grad_adapter.py",
    "tent_grad_adapter_eval",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def maybe_cap_dataset(dataset: Dataset, max_eval_samples: int, seed: int) -> Dataset:
    if max_eval_samples <= 0 or len(dataset) <= max_eval_samples:
        return dataset
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(dataset), size=max_eval_samples, replace=False))
    return Subset(dataset, chosen.tolist())


def load_model_from_checkpoint(path: Path) -> MNISTCNN:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model = MNISTCNN()
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_model(model: MNISTCNN, dataset: Dataset, batch_size: int) -> Tuple[float, float]:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct = 0
    total = 0
    start = time.perf_counter()
    with torch.no_grad():
        for images, labels in dataloader:
            logits = model(images)
            predictions = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    latency = time.perf_counter() - start
    return float(correct / total if total else 0.0), float(latency)


def aggregate_client_checkpoints(paths: List[Path], sample_counts: List[int]) -> MNISTCNN:
    total_samples = float(sum(sample_counts))
    state_dicts = []
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu")
        state_dicts.append(checkpoint["model_state"])

    aggregated_state = {}
    for key in state_dicts[0].keys():
        aggregated_state[key] = sum(
            state_dict[key] * (sample_count / total_samples)
            for state_dict, sample_count in zip(state_dicts, sample_counts)
        )

    model = MNISTCNN()
    model.load_state_dict(aggregated_state)
    model.eval()
    return model


def select_compositions(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    size_leq_three = [row for row in rows if int(row["combination_size"]) <= 3]
    size_four = [row for row in rows if int(row["combination_size"]) == 4]
    selected = list(size_leq_three)
    if size_four:
        selected.append(max(size_four, key=lambda row: float(row["evaluation_accuracy"])))
    return selected


def get_candidate_corruption_key(category: str) -> str:
    if category in DEFAULT_CORRUPTIONS:
        return category
    if category.startswith("clean+"):
        suffix = category.split("+", 1)[1]
        if suffix in DEFAULT_CORRUPTIONS:
            return suffix
    return ""


def get_clean_ratio_for_category(category: str, mixed_clean_ratio: float) -> float:
    if category == "clean":
        return 1.0
    if category in DEFAULT_CORRUPTIONS:
        return 0.0
    if category.startswith("clean+"):
        return mixed_clean_ratio
    return 0.0


def build_client_indices(
    rows: List[Dict[str, str]], corruptions: List[str]
) -> Tuple[Dict[str, Dict[str, Dict[str, str]]], Dict[str, List[Dict[str, str]]]]:
    clean_final_by_client: Dict[str, Dict[str, str]] = {}
    corrupt_candidates_by_category: Dict[str, List[Dict[str, str]]] = {corruption: [] for corruption in corruptions}
    corrupt_final_by_client: Dict[str, Dict[str, Dict[str, str]]] = {corruption: {} for corruption in corruptions}

    for row in rows:
        if row["category"] == "clean" and row["checkpoint_kind"] == "final_snapshot":
            clean_final_by_client[row["original_client_id"]] = row
        candidate_corruption = get_candidate_corruption_key(row["category"])
        if candidate_corruption and candidate_corruption in corruptions:
            corrupt_candidates_by_category[candidate_corruption].append(row)
            if row["category"] == candidate_corruption and row["checkpoint_kind"] == "final_snapshot":
                corrupt_final_by_client[candidate_corruption][row["original_client_id"]] = row

    return {"clean": clean_final_by_client, "corrupt_final": corrupt_final_by_client}, corrupt_candidates_by_category


def build_qos_distance(candidate: Dict[str, str], ideal: Dict[str, float], weights: Dict[str, float]) -> float:
    accuracy = float(candidate["evaluation_accuracy"])
    latency = float(candidate["latency_seconds"])
    reliability = float(candidate["reliability_score"])

    acc_scale = max(ideal["accuracy"], 1e-6)
    lat_scale = max(ideal["latency"], 1e-6)
    rel_scale = max(ideal["reliability"], 1e-6)

    d_acc = abs(accuracy - ideal["accuracy"]) / acc_scale
    d_lat = abs(latency - ideal["latency"]) / lat_scale
    d_rel = abs(reliability - ideal["reliability"]) / rel_scale
    return weights["accuracy"] * d_acc + weights["latency"] * d_lat + weights["reliability"] * d_rel


def parse_class_distribution(row: Dict[str, str]) -> List[float]:
    raw = json.loads(row["functional_class_distribution"])
    if isinstance(raw, dict):
        size = max(int(key) for key in raw.keys()) + 1 if raw else 0
        values = [0.0] * size
        for key, value in raw.items():
            values[int(key)] = float(value)
    else:
        values = [float(value) for value in raw]
    total = sum(values)
    if total <= 0:
        return values
    return [value / total for value in values]


def distribution_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    size = max(len(vec_a), len(vec_b))
    padded_a = vec_a + [0.0] * (size - len(vec_a))
    padded_b = vec_b + [0.0] * (size - len(vec_b))
    distance = l1_distance(padded_a, padded_b) / 2.0
    return max(0.0, 1.0 - distance)


def qos_similarity_ratio(candidate: Dict[str, str], failing_clean_profile: Dict[str, str]) -> float:
    ideal_profile = {
        "accuracy": float(failing_clean_profile["evaluation_accuracy"]),
        "latency": float(failing_clean_profile["latency_seconds"]),
        "reliability": float(failing_clean_profile["reliability_score"]),
    }
    qos_weights = {"accuracy": 0.5, "reliability": 0.3, "latency": 0.2}
    distance = build_qos_distance(candidate, ideal_profile, qos_weights)
    return 100.0 / (1.0 + distance)


def data_similarity_ratio(
    candidate: Dict[str, str],
    failing_clean_profile: Dict[str, str],
    failing_corrupt_profile: Dict[str, str],
    corruption: str,
    target_clean_ratio: float,
    mixed_clean_ratio: float,
) -> float:
    candidate_corruption = get_candidate_corruption_key(candidate["category"])
    corruption_match = 1.0 if candidate_corruption == corruption else 0.0
    candidate_clean_ratio = get_clean_ratio_for_category(candidate["category"], mixed_clean_ratio)
    clean_ratio_similarity = max(0.0, 1.0 - abs(candidate_clean_ratio - target_clean_ratio))
    class_distribution_similarity = distribution_similarity(
        parse_class_distribution(candidate),
        parse_class_distribution(failing_clean_profile),
    )
    prediction_similarity = distribution_similarity(
        parse_distribution(candidate),
        parse_distribution(failing_corrupt_profile),
    )
    score = (
        0.35 * corruption_match
        + 0.25 * clean_ratio_similarity
        + 0.20 * class_distribution_similarity
        + 0.20 * prediction_similarity
    )
    return 100.0 * max(0.0, min(1.0, score))


def requirement_similarity_ratio(
    candidate: Dict[str, str],
    failing_clean_profile: Dict[str, str],
    failing_corrupt_profile: Dict[str, str],
    corruption: str,
    target_clean_ratio: float,
    mixed_clean_ratio: float,
) -> Tuple[float, float, float]:
    qos_similarity = qos_similarity_ratio(candidate, failing_clean_profile)
    data_similarity = data_similarity_ratio(
        candidate,
        failing_clean_profile,
        failing_corrupt_profile,
        corruption,
        target_clean_ratio,
        mixed_clean_ratio,
    )
    final_similarity = 0.45 * qos_similarity + 0.55 * data_similarity
    return final_similarity, qos_similarity, data_similarity


def parse_distribution(row: Dict[str, str]) -> List[float]:
    return [float(x) for x in json.loads(row["prediction_distribution"])]


def l1_distance(vec_a: List[float], vec_b: List[float]) -> float:
    return sum(abs(a - b) for a, b in zip(vec_a, vec_b))


def contextual_score(candidate: Dict[str, str], failing_clean_profile: Dict[str, str], remaining_context_profiles: List[Dict[str, str]]) -> float:
    replacement_target = {
        "accuracy": float(failing_clean_profile["evaluation_accuracy"]),
        "latency": float(failing_clean_profile["latency_seconds"]),
        "reliability": float(failing_clean_profile["reliability_score"]),
    }
    replacement_distance = build_qos_distance(
        candidate, replacement_target, {"accuracy": 0.5, "latency": 0.2, "reliability": 0.3}
    )

    if remaining_context_profiles:
        context_target = {
            "accuracy": mean(float(row["evaluation_accuracy"]) for row in remaining_context_profiles),
            "latency": mean(float(row["latency_seconds"]) for row in remaining_context_profiles),
            "reliability": mean(float(row["reliability_score"]) for row in remaining_context_profiles),
        }
        context_qos_distance = build_qos_distance(
            candidate, context_target, {"accuracy": 0.5, "latency": 0.2, "reliability": 0.3}
        )
        remaining_distributions = [parse_distribution(row) for row in remaining_context_profiles]
        avg_distribution = [
            mean(values[idx] for values in remaining_distributions)
            for idx in range(len(remaining_distributions[0]))
        ]
        context_distribution_distance = l1_distance(parse_distribution(candidate), avg_distribution) / len(avg_distribution)
    else:
        context_qos_distance = 0.0
        context_distribution_distance = 0.0

    return 0.45 * replacement_distance + 0.35 * context_qos_distance + 0.20 * context_distribution_distance


def utility_score(candidate: Dict[str, str]) -> float:
    accuracy = float(candidate["evaluation_accuracy"])
    reliability = float(candidate["reliability_score"])
    latency = float(candidate["latency_seconds"])
    return 0.5 * accuracy + 0.35 * reliability - 0.15 * latency / 10.0


def functional_match(candidate: Dict[str, str], failing_clean_profile: Dict[str, str]) -> bool:
    return (
        candidate["functional_task_type"] == failing_clean_profile["functional_task_type"]
        and candidate["functional_input_modality"] == failing_clean_profile["functional_input_modality"]
        and candidate["functional_output_type"] == failing_clean_profile["functional_output_type"]
        and candidate["functional_label_space"] == failing_clean_profile["functional_label_space"]
    )


def choose_mlaas_candidate(candidate_pool: List[Dict[str, str]], failing_clean_profile: Dict[str, str], failing_corrupt_profile: Dict[str, str]) -> Tuple[Dict[str, str], str]:
    stage1 = [candidate for candidate in candidate_pool if functional_match(candidate, failing_clean_profile)]
    stage2 = [
        candidate
        for candidate in stage1
        if float(candidate["evaluation_accuracy"]) >= float(failing_corrupt_profile["evaluation_accuracy"])
        and float(candidate["reliability_score"]) >= float(failing_corrupt_profile["reliability_score"])
        and float(candidate["latency_seconds"]) <= 1.25 * float(failing_clean_profile["latency_seconds"])
    ]
    if stage2:
        return max(stage2, key=utility_score), "strict_qos_rules"
    if stage1:
        return max(stage1, key=utility_score), "functional_rules_only"
    return max(candidate_pool, key=utility_score), "fallback_best_utility"


def expand_candidate_pool(candidate_pool: List[Dict[str, str]], multiplier: int) -> List[Dict[str, str]]:
    if multiplier <= 1:
        return list(candidate_pool)
    expanded: List[Dict[str, str]] = []
    for row in candidate_pool:
        expanded.append(row)
        for replica_index in range(1, multiplier):
            replica = dict(row)
            replica["service_id"] = f"{row['service_id']}__dup{replica_index}"
            replica["service_alias"] = f"{row['service_alias']}__dup{replica_index}"
            expanded.append(replica)
    return expanded


def candidate_preference_score(
    method: str,
    candidate: Dict[str, str],
    failing_clean_profile: Dict[str, str],
    failing_corrupt_profile: Dict[str, str],
    remaining_context_profiles: List[Dict[str, str]],
) -> Tuple[float, str]:
    if method == "semantic":
        qos_weights = {"accuracy": 0.5, "reliability": 0.3, "latency": 0.2}
        ideal_profile = {
            "accuracy": float(failing_clean_profile["evaluation_accuracy"]),
            "latency": float(failing_clean_profile["latency_seconds"]),
            "reliability": float(failing_clean_profile["reliability_score"]),
        }
        return -build_qos_distance(candidate, ideal_profile, qos_weights), "semantic_qos_distance"
    if method == "context":
        return -contextual_score(candidate, failing_clean_profile, remaining_context_profiles), "contextual_fit"
    if method == "mlaas":
        if functional_match(candidate, failing_clean_profile):
            if (
                float(candidate["evaluation_accuracy"]) >= float(failing_corrupt_profile["evaluation_accuracy"])
                and float(candidate["reliability_score"]) >= float(failing_corrupt_profile["reliability_score"])
                and float(candidate["latency_seconds"]) <= 1.25 * float(failing_clean_profile["latency_seconds"])
            ):
                return 100.0 + utility_score(candidate), "strict_qos_rules"
            return 50.0 + utility_score(candidate), "functional_rules_only"
        return utility_score(candidate), "fallback_best_utility"
    raise ValueError(f"Unsupported method: {method}")


def select_candidate_combination(
    method: str,
    candidate_pool: List[Dict[str, str]],
    failing_profiles: List[Tuple[str, Dict[str, str], Dict[str, str], List[Dict[str, str]]]],
    combination_beam_width: int,
) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    ranked_lists: List[List[Tuple[Dict[str, str], float, str]]] = []
    for _, failing_clean_profile, failing_corrupt_profile, remaining_context_profiles in failing_profiles:
        ranked = [
            (candidate, *candidate_preference_score(method, candidate, failing_clean_profile, failing_corrupt_profile, remaining_context_profiles))
            for candidate in candidate_pool
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        ranked_lists.append(ranked[:combination_beam_width])

    best_tuple = None
    best_score = float("-inf")
    combinations_evaluated = 0
    for combo in product(*ranked_lists):
        service_ids = [item[0]["service_id"] for item in combo]
        if len(service_ids) != len(set(service_ids)):
            continue
        combinations_evaluated += 1
        combo_score = sum(item[1] for item in combo)
        if combo_score > best_score:
            best_score = combo_score
            best_tuple = combo

    if best_tuple is None:
        best_tuple = tuple(ranked[0] for ranked in ranked_lists)
        combinations_evaluated = 1

    chosen_candidates = [item[0] for item in best_tuple]
    selection_rules = [item[2] for item in best_tuple]
    return chosen_candidates, {
        "selection_rule": "+".join(selection_rules),
        "selection_score": best_score,
        "combination_count_evaluated": combinations_evaluated,
    }


def select_candidate(method: str, candidate_pool: List[Dict[str, str]], failing_clean_profile: Dict[str, str], failing_corrupt_profile: Dict[str, str], remaining_context_profiles: List[Dict[str, str]]) -> Tuple[Dict[str, str], Dict[str, object]]:
    if method == "semantic":
        qos_weights = {"accuracy": 0.5, "reliability": 0.3, "latency": 0.2}
        ideal_profile = {
            "accuracy": float(failing_clean_profile["evaluation_accuracy"]),
            "latency": float(failing_clean_profile["latency_seconds"]),
            "reliability": float(failing_clean_profile["reliability_score"]),
        }
        ranked = sorted(candidate_pool, key=lambda candidate: build_qos_distance(candidate, ideal_profile, qos_weights))
        candidate = ranked[0]
        return candidate, {
            "selection_rule": "semantic_qos_distance",
            "selection_score": build_qos_distance(candidate, ideal_profile, qos_weights),
        }
    if method == "context":
        ranked = sorted(
            candidate_pool,
            key=lambda candidate: contextual_score(candidate, failing_clean_profile, remaining_context_profiles),
        )
        candidate = ranked[0]
        return candidate, {
            "selection_rule": "contextual_fit",
            "selection_score": contextual_score(candidate, failing_clean_profile, remaining_context_profiles),
        }
    if method == "mlaas":
        candidate, selection_rule = choose_mlaas_candidate(candidate_pool, failing_clean_profile, failing_corrupt_profile)
        return candidate, {
            "selection_rule": selection_rule,
            "selection_score": utility_score(candidate),
        }
    raise ValueError(f"Unsupported method: {method}")


def output_dir_for_method(method: str) -> Path:
    return OUTPUTS_ROOT / method


def output_prefix_for_method(method: str) -> str:
    return {
        "semantic": "semantic",
        "context": "contextual",
        "mlaas": "mlaas",
        "tta": "tta",
    }[method]


def load_checkpoint_state(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint["model_state"] if "model_state" in checkpoint else checkpoint


def build_standard_model(state_dict):
    model = MNISTCNN()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def compute_mean_logits(model: MNISTCNN, dataloader: DataLoader) -> torch.Tensor:
    model.eval()
    logits_sum = None
    total = 0
    with torch.no_grad():
        for images, _ in dataloader:
            logits = model(images)
            logits_sum = logits.sum(dim=0) if logits_sum is None else logits_sum + logits.sum(dim=0)
            total += images.size(0)
    if logits_sum is None or total == 0:
        raise ValueError("Probe loader is empty")
    return logits_sum / total


def sample_size_weights(client_sizes: Dict[str, int]) -> Dict[str, float]:
    total = float(sum(client_sizes.values()))
    return {client_name: size / total for client_name, size in client_sizes.items()}


def protected_similarity_weights(
    server_model: MNISTCNN,
    client_states: Dict[str, Dict[str, torch.Tensor]],
    client_sizes: Dict[str, int],
    probe_loader: DataLoader,
    beta: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    mu_server = compute_mean_logits(server_model, probe_loader)
    sample_weights_map = sample_size_weights(client_sizes)
    distances: Dict[str, float] = {}
    scores: Dict[str, float] = {}

    for client_name, state in client_states.items():
        client_model = build_standard_model(state)
        mu_client = compute_mean_logits(client_model, probe_loader)
        distance = torch.norm(mu_server - mu_client, p=2).item()
        distances[client_name] = float(distance)
        scores[client_name] = math.exp(-beta * distance) * sample_weights_map[client_name]

    total_score = float(sum(scores.values()))
    weights = {client_name: score / total_score for client_name, score in scores.items()}
    return weights, distances


def apply_weighted_client_parameters(
    client_states: Dict[str, Dict[str, torch.Tensor]],
    weights: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    if not client_states:
        raise ValueError("No client states provided for aggregation")

    aggregated_state: Dict[str, torch.Tensor] = {}
    param_names = sorted({param_name for state in client_states.values() for param_name in state.keys()})
    for param_name in param_names:
        available_clients = [client_name for client_name, state in client_states.items() if param_name in state]
        if not available_clients:
            continue
        if param_name.endswith("num_batches_tracked"):
            aggregated_state[param_name] = client_states[available_clients[0]][param_name].clone()
            continue
        total_weight = sum(weights[client_name] for client_name in available_clients)
        aggregated_state[param_name] = sum(
            client_states[client_name][param_name] * (weights[client_name] / total_weight)
            for client_name in available_clients
        )
    return aggregated_state


def aggregate_states_with_weights(states: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    if not states:
        raise ValueError("No states provided for aggregation")
    aggregated: Dict[str, torch.Tensor] = {}
    for param_name in states[0].keys():
        if param_name.endswith("num_batches_tracked"):
            aggregated[param_name] = states[0][param_name].clone()
            continue
        acc = None
        for state, weight in zip(states, weights):
            term = state[param_name] * weight
            acc = term if acc is None else acc + term
        aggregated[param_name] = acc
    return aggregated


def state_key_groups(state: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
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
) -> Tuple[float, float, float]:
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
    weight_pref: Tuple[float, float],
    bn_pref: Tuple[float, float],
) -> Dict[str, torch.Tensor]:
    weight_keys, bn_keys = state_key_groups(performing_state)
    out: Dict[str, torch.Tensor] = {}
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
    return out


def build_probe_dataset(corruptions: List[str], probe_samples_per_corruption: int, seed: int, mnist_c_root: Path) -> Dataset:
    probe_datasets = []
    rng = np.random.default_rng(seed)
    per_digit = max(1, probe_samples_per_corruption // 10)

    for corruption in sorted(set(corruptions)):
        dataset = LocalMNISTC(root=mnist_c_root, corruption=corruption, split="test")
        labels = np.asarray(dataset.labels)
        selected_indices = []
        for digit in range(10):
            digit_indices = np.where(labels == digit)[0]
            rng.shuffle(digit_indices)
            selected_indices.extend(digit_indices[:per_digit].tolist())
        probe_datasets.append(Subset(dataset, selected_indices))

    return ConcatDataset(probe_datasets)


def get_sample_count_from_clean_profile(clean_profile: Dict[str, str]) -> int:
    return int(clean_profile["functional_num_samples"])


def find_profile_row(rows: List[Dict[str, str]], *, category: str, client_name: str, checkpoint_kind: str = "final_snapshot") -> Dict[str, str]:
    for row in rows:
        if (
            row["category"] == category
            and row["original_client_id"] == client_name
            and row["checkpoint_kind"] == checkpoint_kind
        ):
            return row
    raise KeyError(f"Could not find profile row for category={category}, client={client_name}, kind={checkpoint_kind}")


def synthesize_tta_artifact_from_mixed_profile(output_dir: Path, client_name: str, corruption: str) -> None:
    profile_rows = read_csv(CLIENT_PROFILES_CSV)
    clean_row = find_profile_row(profile_rows, category="clean", client_name=client_name)
    adapted_row = find_profile_row(profile_rows, category=f"clean+{corruption}", client_name=client_name)

    clean_model = load_model_from_checkpoint(Path(clean_row["weights_file"]))
    adapted_model = load_model_from_checkpoint(Path(adapted_row["weights_file"]))
    clean_state = clean_model.state_dict()
    adapted_state = adapted_model.state_dict()
    bn_before = tent_grad_module.snapshot_bn_parameters(clean_model)
    bn_after = tent_grad_module.snapshot_bn_parameters(adapted_model)
    bn_mean_var = {
        layer_name: {
            "mean": layer_values["running_mean"],
            "var": layer_values["running_var"],
        }
        for layer_name, layer_values in bn_after.items()
    }
    prediction_distribution = json.loads(adapted_row["prediction_distribution"])
    parameter_updates = {
        name: (adapted_state[name].detach().cpu() - clean_state[name].detach().cpu()).reshape(-1).tolist()
        for name in adapted_state.keys()
        if "num_batches_tracked" not in name
    }
    adapted_weights = {
        name: parameter.detach().cpu().reshape(-1).tolist()
        for name, parameter in adapted_state.items()
        if "running_" not in name and "num_batches_tracked" not in name
    }
    adapted_bn_affine = tent_grad_module.extract_bn_affine_parameters(bn_after)

    source_checkpoint = torch.load(Path(adapted_row["weights_file"]), map_location="cpu")
    torch.save(source_checkpoint, output_dir / "adapted_client_model.pt")
    torch.save(adapted_weights, output_dir / "wdm_adapted_weights.pt")
    torch.save(parameter_updates, output_dir / "ucs_adapted_layer_updates.pt")
    torch.save(bn_mean_var, output_dir / "bndas_adapted_bn_mean_var.pt")
    torch.save(adapted_bn_affine, output_dir / "bnuas_adapted_bn_gamma_beta.pt")
    torch.save(prediction_distribution, output_dir / "pdam_adapted_prediction_distribution.pt")
    torch.save([], output_dir / "bn_batch_statistics.pt")

    write_json(
        output_dir / "config.json",
        {
            "client_name": client_name,
            "corruption": corruption,
            "adaptation_source": "mixed_clean_corruption_checkpoint",
            "source_profile_service_id": adapted_row["service_id"],
            "source_checkpoint": adapted_row["weights_file"],
            "reference_clean_service_id": clean_row["service_id"],
        },
    )
    write_json(output_dir / "bn_parameters_before.json", bn_before)
    write_json(output_dir / "bn_parameters_after.json", bn_after)
    write_json(output_dir / "bn_statistics_summary.json", {})
    write_json(output_dir / "online_metrics.json", [])
    write_json(output_dir / "wdm_adapted_weights.json", adapted_weights)
    write_json(output_dir / "ucs_adapted_layer_updates.json", parameter_updates)
    write_json(output_dir / "bndas_adapted_bn_mean_var.json", bn_mean_var)
    write_json(output_dir / "bnuas_adapted_bn_gamma_beta.json", adapted_bn_affine)
    write_json(output_dir / "pdam_adapted_prediction_distribution.json", prediction_distribution)
    write_json(
        output_dir / "summary.json",
        {
            "client_name": client_name,
            "corruption": corruption,
            "num_target_samples": int(adapted_row["functional_num_samples"]),
            "baseline_accuracy": float(clean_row["evaluation_accuracy"]),
            "adapted_accuracy": float(adapted_row["evaluation_accuracy"]),
            "num_adaptation_batches": 0,
            "adaptation_source": "mixed_clean_corruption_checkpoint",
            "output_dir": str(output_dir),
        },
    )


def adapt_client_with_tent_grad_limited(
    client_name: str,
    corruption: str,
    output_dir: Path,
    tta_batch_size: int,
    tta_learning_rate: float,
    tta_max_batches: Optional[int],
    tta_max_target_samples: Optional[int],
    seed: int,
) -> None:
    tent_grad_module.set_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[tta] adapting {client_name} on {corruption}", flush=True)

    model = tent_grad_module.load_client_model(fl_run_dir=FL_RUN_DIR, client_name=client_name)
    before_model = deepcopy(model)
    bn_before = tent_grad_module.snapshot_bn_parameters(model)

    dataset = tent_grad_module.LocalMNISTC(
        root=PROJECT_ROOT / "data" / "mnist_c",
        corruption=corruption,
        split="test",
        allowed_digits=None,
    )
    dataset = maybe_cap_dataset(dataset, tta_max_target_samples or 0, seed=seed)
    dataloader = DataLoader(dataset, batch_size=tta_batch_size, shuffle=False, num_workers=0)
    print(f"[tta] target samples: {len(dataset)}", flush=True)

    baseline_accuracy = tent_grad_module.evaluate_accuracy(model, dataloader)
    print(f"[tta] baseline accuracy: {baseline_accuracy:.4f}", flush=True)
    model = tent_grad_module.configure_model_for_tent_grad(model)
    parameters, parameter_names = tent_grad_module.collect_tent_grad_parameters(model)
    optimizer = torch.optim.Adam(parameters, lr=tta_learning_rate)
    recorder = tent_grad_module.BNStatisticsRecorder(model)

    online_metrics = []
    for batch_index, (images, labels) in enumerate(dataloader):
        if tta_max_batches is not None and batch_index >= tta_max_batches:
            break
        recorder.begin_step()
        optimizer.zero_grad()
        logits = model(images)
        entropy = tent_grad_module.softmax_entropy(logits).mean()
        entropy.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_accuracy = (predictions == labels).float().mean().item()
        online_metrics.append(
            {
                "batch_index": batch_index,
                "entropy": float(entropy.item()),
                "accuracy_before_update_output": float(batch_accuracy),
                "batch_size": int(labels.size(0)),
            }
        )

    recorder.close()
    print(f"[tta] finished adaptation batches: {len(online_metrics)}", flush=True)
    adapted_accuracy = tent_grad_module.evaluate_accuracy(model, dataloader)
    adapted_prediction_distribution = tent_grad_module.evaluate_prediction_distribution(model, dataloader)
    bn_after = tent_grad_module.snapshot_bn_parameters(model)
    bn_summary = recorder.summarize()
    model_parameter_updates = tent_grad_module.compute_model_parameter_updates(before_model=before_model, after_model=model)
    adapted_model_weights = tent_grad_module.snapshot_model_parameters(model)
    adapted_bn_affine = tent_grad_module.extract_bn_affine_parameters(bn_after)
    adapted_bn_mean_var = {
        layer_name: {
            "mean": layer_summary["mean_of_batch_means"],
            "var": layer_summary["mean_of_batch_vars"],
        }
        for layer_name, layer_summary in bn_summary.items()
    }

    torch.save(
        {"client_name": client_name, "corruption": corruption, "model_state": model.state_dict()},
        output_dir / "adapted_client_model.pt",
    )
    torch.save(recorder.records, output_dir / "bn_batch_statistics.pt")
    torch.save(adapted_model_weights, output_dir / "wdm_adapted_weights.pt")
    torch.save(model_parameter_updates, output_dir / "ucs_adapted_layer_updates.pt")
    torch.save(adapted_bn_mean_var, output_dir / "bndas_adapted_bn_mean_var.pt")
    torch.save(adapted_bn_affine, output_dir / "bnuas_adapted_bn_gamma_beta.pt")
    torch.save(adapted_prediction_distribution, output_dir / "pdam_adapted_prediction_distribution.pt")

    write_json(
        output_dir / "config.json",
        {
            "fl_run_dir": str(FL_RUN_DIR),
            "client_name": client_name,
            "corruption": corruption,
            "split": "test",
            "batch_size": tta_batch_size,
            "learning_rate": tta_learning_rate,
            "max_batches": tta_max_batches,
            "max_target_samples": tta_max_target_samples,
            "allowed_digits": None,
            "seed": seed,
            "num_target_samples": len(dataset),
            "tent_grad_parameter_names": parameter_names,
            "tent_parameter_names": parameter_names,
        },
    )
    write_json(output_dir / "bn_parameters_before.json", bn_before)
    write_json(output_dir / "bn_parameters_after.json", bn_after)
    write_json(output_dir / "bn_statistics_summary.json", bn_summary)
    write_json(output_dir / "online_metrics.json", online_metrics)
    write_json(output_dir / "wdm_adapted_weights.json", adapted_model_weights)
    write_json(output_dir / "ucs_adapted_layer_updates.json", model_parameter_updates)
    write_json(output_dir / "bndas_adapted_bn_mean_var.json", adapted_bn_mean_var)
    write_json(output_dir / "bnuas_adapted_bn_gamma_beta.json", adapted_bn_affine)
    write_json(output_dir / "pdam_adapted_prediction_distribution.json", adapted_prediction_distribution)
    write_json(
        output_dir / "summary.json",
        {
            "client_name": client_name,
            "corruption": corruption,
            "num_target_samples": len(dataset),
            "baseline_accuracy": baseline_accuracy,
            "adapted_accuracy": adapted_accuracy,
            "num_adaptation_batches": len(online_metrics),
            "batch_size": tta_batch_size,
            "learning_rate": tta_learning_rate,
            "max_batches": tta_max_batches,
            "max_target_samples": tta_max_target_samples,
        },
    )
    print(f"[tta] saved artifacts for {client_name} on {corruption}", flush=True)


def ensure_tta_artifact(
    client_name: str,
    corruption: str,
    tta_batch_size: int,
    tta_learning_rate: float,
    tta_max_batches: Optional[int],
    tta_max_target_samples: Optional[int],
    seed: int,
) -> Path:
    output_dir = TTA_ARTIFACTS_ROOT / f"{client_name}_{corruption}"
    if (output_dir / "adapted_client_model.pt").exists() and (output_dir / "summary.json").exists():
        return output_dir

    legacy_dir = LEGACY_TTA_ARTIFACTS_ROOT / f"{client_name}_{corruption}"
    if legacy_dir.exists() and (legacy_dir / "adapted_client_model.pt").exists() and (legacy_dir / "summary.json").exists():
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(legacy_dir, output_dir)
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    synthesize_tta_artifact_from_mixed_profile(output_dir, client_name, corruption)
    return output_dir


def ensure_required_tta_artifacts(
    tta_batch_size: int,
    tta_learning_rate: float,
    tta_max_batches: Optional[int],
    tta_max_target_samples: Optional[int],
    seed: int,
    corruptions: List[str],
) -> None:
    for client_id in range(5):
        client_name = f"client_{client_id}"
        for corruption in corruptions:
            ensure_tta_artifact(
                client_name,
                corruption,
                tta_batch_size,
                tta_learning_rate,
                tta_max_batches,
                tta_max_target_samples,
                seed,
            )


def ensure_pairwise_compatibility(chunk_size: int) -> Path:
    summary_csv = TTA_COMPAT_ROOT / "summary_tta_benchmark.csv"
    if summary_csv.exists():
        return summary_csv

    TTA_COMPAT_ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "python3",
            str(PROJECT_ROOT / "run_pairwise_compatibility.py"),
            "--adapted-root",
            str(TTA_ARTIFACTS_ROOT),
            "--composition-root",
            str(PROJECT_ROOT / "Performing composition" / "artifacts_expanded"),
            "--output-root",
            str(TTA_COMPAT_ROOT),
            "--mnist-c-root",
            str(PROJECT_ROOT / "data" / "mnist_c"),
            "--chunk-size",
            str(chunk_size),
            "--output-suffix",
            "tta_benchmark",
        ],
        check=True,
        cwd=str(PROJECT_ROOT),
    )
    return summary_csv


def load_compatibility_lookup(summary_csv: Path, threshold_field: str = "predicted_at_0_5") -> Dict[Tuple[str, str], Dict[str, float]]:
    rows = read_csv(summary_csv)
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in rows:
        key = (row["adapted_case"], row["composition_name"])
        grouped.setdefault(key, []).append(row)

    lookup: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, group in grouped.items():
        mean_ttaas = sum(float(row["ttaas"]) for row in group) / len(group)
        mean_pred = sum(float(row.get(threshold_field, 0.0)) for row in group) / len(group)
        mean_truth = sum(float(str(row["ground_truth_composable"]).lower() == "true") for row in group) / len(group)
        lookup[key] = {
            "mean_ttaas": mean_ttaas,
            "mean_predicted": mean_pred,
            "mean_ground_truth": mean_truth,
            "is_composable": mean_pred >= 0.5,
        }
    return lookup


def run_method(
    method: str,
    batch_size: int,
    max_eval_samples: int,
    seed: int,
    candidate_pool_multiplier: int,
    max_substitutions: int,
    combination_beam_width: int,
    target_clean_ratio: float,
    mixed_clean_ratio: float,
    corruptions: List[str],
) -> Dict[str, object]:
    method_start = time.perf_counter()
    composition_rows = read_csv(COMPOSITION_PROFILES_CSV)
    client_rows = read_csv(CLIENT_PROFILES_CSV)
    selected_compositions = select_compositions(composition_rows)
    client_lookup, corrupt_candidates_by_category = build_client_indices(client_rows, corruptions)

    impact_rows: List[Dict[str, object]] = []
    substitution_rows: List[Dict[str, object]] = []
    selection_time_total = 0.0
    aggregation_time_total = 0.0
    evaluation_time_total = 0.0
    corrupt_datasets = {
        corruption: maybe_cap_dataset(
            LocalMNISTC(MNIST_C_ROOT, corruption=corruption, split="test"),
            max_eval_samples=max_eval_samples,
            seed=seed + idx,
        )
        for idx, corruption in enumerate(corruptions)
    }

    for row in selected_compositions:
        composition_clients = json.loads(row["client_names"])
        baseline_accuracy = float(row["evaluation_accuracy"])
        composition_model = load_model_from_checkpoint(Path(row["weights_file"]))

        for corruption in corruptions:
            case_start = time.perf_counter()
            eval_start = time.perf_counter()
            corrupt_dataset = corrupt_datasets[corruption]
            degraded_accuracy, degraded_latency = evaluate_model(composition_model, corrupt_dataset, batch_size=batch_size)
            evaluation_time_total += time.perf_counter() - eval_start
            accuracy_drop = baseline_accuracy - degraded_accuracy
            impact_rows.append(
                {
                    "profile_id": row["profile_id"],
                    "composition_name": row["composition_name"],
                    "combination_size": row["combination_size"],
                    "corruption_category": corruption,
                    "before_accuracy": baseline_accuracy,
                    "before_reliability_score": float(row["reliability_score"]),
                    "before_latency_seconds": float(row["latency_seconds"]),
                    "after_corruption_accuracy": degraded_accuracy,
                    "after_corruption_latency_seconds": degraded_latency,
                    "accuracy_drop": accuracy_drop,
                    "weights_file": row["weights_file"],
                }
            )

            component_degradations = []
            for client_name in composition_clients:
                clean_profile = client_lookup["clean"][client_name]
                corrupt_profile = client_lookup["corrupt_final"][corruption][client_name]
                degradation = float(clean_profile["evaluation_accuracy"]) - float(corrupt_profile["evaluation_accuracy"])
                component_degradations.append((degradation, client_name, clean_profile, corrupt_profile))

            component_degradations.sort(reverse=True, key=lambda item: item[0])
            selected_failures = component_degradations[: max(1, min(max_substitutions, len(component_degradations)))]
            failing_clients = [item[1] for item in selected_failures]
            remaining_clients = [client_name for client_name in composition_clients if client_name not in failing_clients]

            candidate_pool = [
                candidate
                for candidate in corrupt_candidates_by_category[corruption]
                if candidate["original_client_id"] not in composition_clients
            ]
            candidate_pool = expand_candidate_pool(candidate_pool, candidate_pool_multiplier)
            if not candidate_pool:
                continue

            selection_start = time.perf_counter()
            chosen_candidates, selection_meta = select_candidate_combination(
                method,
                candidate_pool,
                [
                    (
                        failing_client,
                        failing_clean_profile,
                        failing_corrupt_profile,
                        [client_lookup["corrupt_final"][corruption][client_name] for client_name in remaining_clients],
                    )
                    for _, failing_client, failing_clean_profile, failing_corrupt_profile in selected_failures
                ],
                combination_beam_width,
            )
            selection_time = time.perf_counter() - selection_start
            selection_time_total += selection_time
            similarity_triplets = [
                requirement_similarity_ratio(
                    candidate,
                    failing_clean_profile,
                    failing_corrupt_profile,
                    corruption,
                    target_clean_ratio,
                    mixed_clean_ratio,
                )
                for candidate, (_, _, failing_clean_profile, failing_corrupt_profile) in zip(chosen_candidates, selected_failures)
            ]
            similarity_ratio = mean(item[0] for item in similarity_triplets)
            qos_similarity = mean(item[1] for item in similarity_triplets)
            data_similarity = mean(item[2] for item in similarity_triplets)

            aggregated_paths = [Path(client_lookup["clean"][client]["weights_file"]) for client in remaining_clients]
            aggregated_counts = [int(client_lookup["clean"][client]["functional_num_samples"]) for client in remaining_clients]
            for candidate in chosen_candidates:
                aggregated_paths.append(Path(candidate["weights_file"]))
                aggregated_counts.append(int(candidate["functional_num_samples"]))

            aggregation_start = time.perf_counter()
            substituted_model = aggregate_client_checkpoints(aggregated_paths, aggregated_counts)
            aggregation_time = time.perf_counter() - aggregation_start
            aggregation_time_total += aggregation_time
            eval_start = time.perf_counter()
            substituted_accuracy, substituted_latency = evaluate_model(substituted_model, corrupt_dataset, batch_size=batch_size)
            evaluation_time = time.perf_counter() - eval_start
            evaluation_time_total += evaluation_time

            recovery_ratio = 0.0
            if accuracy_drop > 0:
                recovery_ratio = max(0.0, substituted_accuracy - degraded_accuracy) / accuracy_drop

            substitution_rows.append(
                {
                    "profile_id": row["profile_id"],
                    "composition_name": row["composition_name"],
                    "combination_size": row["combination_size"],
                    "corruption_category": corruption,
                    "before_accuracy": baseline_accuracy,
                    "after_corruption_accuracy": degraded_accuracy,
                    "after_substitution_accuracy": substituted_accuracy,
                    "accuracy_drop": accuracy_drop,
                    "accuracy_recovery": substituted_accuracy - degraded_accuracy,
                    "recovery_ratio": recovery_ratio,
                    "failing_client": json.dumps(failing_clients),
                    "recommended_substitute_service": json.dumps([candidate["service_id"] for candidate in chosen_candidates]),
                    "recommended_substitute_client": json.dumps([candidate["original_client_id"] for candidate in chosen_candidates]),
                    "recommended_substitute_checkpoint": json.dumps(
                        [candidate.get("source_checkpoint_name", "") for candidate in chosen_candidates]
                    ),
                    "selection_rule": selection_meta["selection_rule"],
                    "selection_score": selection_meta["selection_score"],
                    "candidate_pool_size": len(candidate_pool),
                    "candidate_scan_count": selection_meta["combination_count_evaluated"],
                    "num_substitutions": len(chosen_candidates),
                    "qos_similarity_ratio": qos_similarity,
                    "data_similarity_ratio": data_similarity,
                    "substitution_similarity_ratio": similarity_ratio,
                    "low_similarity_substitute": similarity_ratio < 60.0,
                    "selection_time_seconds": selection_time,
                    "aggregation_time_seconds": aggregation_time,
                    "evaluation_time_seconds": evaluation_time,
                    "case_time_seconds": time.perf_counter() - case_start,
                    "substituted_latency_seconds": substituted_latency,
                    "remaining_clients": json.dumps(remaining_clients),
                    "substituted_client_set": json.dumps(remaining_clients + [candidate["original_client_id"] for candidate in chosen_candidates]),
                }
            )

    out_root = output_dir_for_method(method)
    prefix = output_prefix_for_method(method)
    write_csv(
        out_root / f"{prefix}_corruption_impact.csv",
        impact_rows,
        [
            "profile_id",
            "composition_name",
            "combination_size",
            "corruption_category",
            "before_accuracy",
            "before_reliability_score",
            "before_latency_seconds",
            "after_corruption_accuracy",
            "after_corruption_latency_seconds",
            "accuracy_drop",
            "weights_file",
        ],
    )
    write_csv(
        out_root / f"{prefix}_substitution_results.csv",
        substitution_rows,
        [
            "profile_id",
            "composition_name",
            "combination_size",
            "corruption_category",
            "before_accuracy",
            "after_corruption_accuracy",
            "after_substitution_accuracy",
            "accuracy_drop",
            "accuracy_recovery",
            "recovery_ratio",
            "failing_client",
            "recommended_substitute_service",
            "recommended_substitute_client",
            "recommended_substitute_checkpoint",
            "selection_rule",
            "selection_score",
            "candidate_pool_size",
            "candidate_scan_count",
            "num_substitutions",
            "qos_similarity_ratio",
            "data_similarity_ratio",
            "substitution_similarity_ratio",
            "low_similarity_substitute",
            "selection_time_seconds",
            "aggregation_time_seconds",
            "evaluation_time_seconds",
            "case_time_seconds",
            "substituted_latency_seconds",
            "remaining_clients",
            "substituted_client_set",
        ],
    )
    write_json(
        out_root / f"{prefix}_substitution_metadata.json",
        {
            "method": method,
            "corruptions": corruptions,
            "num_selected_compositions": len(selected_compositions),
            "num_impact_rows": len(impact_rows),
            "num_substitution_rows": len(substitution_rows),
            "max_eval_samples": max_eval_samples,
            "candidate_pool_multiplier": candidate_pool_multiplier,
            "max_substitutions": max_substitutions,
            "combination_beam_width": combination_beam_width,
            "target_clean_ratio": target_clean_ratio,
            "mixed_clean_ratio": mixed_clean_ratio,
            "runtime_seconds": time.perf_counter() - method_start,
            "selection_time_seconds_total": selection_time_total,
            "aggregation_time_seconds_total": aggregation_time_total,
            "evaluation_time_seconds_total": evaluation_time_total,
        },
    )
    return {
        "method": method,
        "num_selected_compositions": len(selected_compositions),
        "num_impact_rows": len(impact_rows),
        "num_substitution_rows": len(substitution_rows),
    }


def run_tta_method(
    batch_size: int,
    max_eval_samples: int,
    seed: int,
    tta_batch_size: int,
    tta_learning_rate: float,
    tta_max_batches: Optional[int],
    tta_max_target_samples: Optional[int],
    compatibility_chunk_size: int,
    probe_samples_per_corruption: int,
    protected_beta: float,
    max_substitutions: int,
    target_clean_ratio: float,
    mixed_clean_ratio: float,
    corruptions: List[str],
) -> Dict[str, object]:
    method_start = time.perf_counter()
    composition_rows = read_csv(COMPOSITION_PROFILES_CSV)
    client_rows = read_csv(CLIENT_PROFILES_CSV)
    selected_compositions = select_compositions(composition_rows)
    client_lookup, _ = build_client_indices(client_rows, corruptions)

    impact_rows: List[Dict[str, object]] = []
    substitution_rows: List[Dict[str, object]] = []
    aggregation_time_total = 0.0
    evaluation_time_total = 0.0
    corrupt_datasets = {
        corruption: maybe_cap_dataset(
            LocalMNISTC(MNIST_C_ROOT, corruption=corruption, split="test"),
            max_eval_samples=max_eval_samples,
            seed=seed + idx,
        )
        for idx, corruption in enumerate(corruptions)
    }

    for row in selected_compositions:
        composition_clients = json.loads(row["client_names"])
        baseline_accuracy = float(row["evaluation_accuracy"])
        composition_model = load_model_from_checkpoint(Path(row["weights_file"]))

        for corruption in corruptions:
            case_start = time.perf_counter()
            eval_start = time.perf_counter()
            corrupt_dataset = corrupt_datasets[corruption]
            degraded_accuracy, degraded_latency = evaluate_model(composition_model, corrupt_dataset, batch_size=batch_size)
            evaluation_time_total += time.perf_counter() - eval_start
            accuracy_drop = baseline_accuracy - degraded_accuracy
            impact_rows.append(
                {
                    "profile_id": row["profile_id"],
                    "composition_name": row["composition_name"],
                    "combination_size": row["combination_size"],
                    "corruption_category": corruption,
                    "before_accuracy": baseline_accuracy,
                    "before_reliability_score": float(row["reliability_score"]),
                    "before_latency_seconds": float(row["latency_seconds"]),
                    "after_corruption_accuracy": degraded_accuracy,
                    "after_corruption_latency_seconds": degraded_latency,
                    "accuracy_drop": accuracy_drop,
                    "weights_file": row["weights_file"],
                }
            )

            component_degradations = []
            for client_name in composition_clients:
                clean_profile = client_lookup["clean"][client_name]
                corrupt_profile = client_lookup["corrupt_final"][corruption][client_name]
                degradation = float(clean_profile["evaluation_accuracy"]) - float(corrupt_profile["evaluation_accuracy"])
                component_degradations.append((degradation, client_name, clean_profile, corrupt_profile))

            component_degradations.sort(reverse=True, key=lambda item: item[0])
            selected_failures = component_degradations[: max(1, min(max_substitutions, len(component_degradations)))]
            failing_clients = [item[1] for item in selected_failures]
            adapted_cases = [f"{client_name}_{corruption}" for client_name in failing_clients]

            remaining_clients = [client_name for client_name in composition_clients if client_name not in failing_clients]
            protected_accuracy = degraded_accuracy
            protected_latency = degraded_latency
            composable = True
            protected_weights = {}
            protected_distances = {}
            similarity_triplets = []
            for _, failing_client, failing_clean_profile, _ in selected_failures:
                adapted_profile = find_profile_row(client_rows, category=f"clean+{corruption}", client_name=failing_client)
                similarity_triplets.append(
                    requirement_similarity_ratio(
                        adapted_profile,
                        failing_clean_profile,
                        client_lookup["corrupt_final"][corruption][failing_client],
                        corruption,
                        target_clean_ratio,
                        mixed_clean_ratio,
                    )
                )
            similarity_ratio = mean(item[0] for item in similarity_triplets)
            qos_similarity = mean(item[1] for item in similarity_triplets)
            data_similarity = mean(item[2] for item in similarity_triplets)

            if composable:
                aggregation_start = time.perf_counter()
                if not remaining_clients:
                    failing_client = failing_clients[0]
                    adapted_dir = TTA_ARTIFACTS_ROOT / f"{failing_client}_{corruption}"
                    adapted_model_path = adapted_dir / "adapted_client_model.pt"
                    if not adapted_model_path.exists():
                        raise FileNotFoundError(f"Missing existing adapted model: {adapted_model_path}")
                    protected_state = load_checkpoint_state(adapted_model_path)
                    protected_weights = {failing_client: 1.0}
                    protected_distances = {}
                else:
                    performing_states = []
                    for client_name in remaining_clients:
                        clean_profile = client_lookup["clean"][client_name]
                        performing_states.append(load_checkpoint_state(Path(clean_profile["weights_file"])))
                    performing_state = aggregate_states_with_weights(
                        performing_states,
                        [1.0 / len(performing_states)] * len(performing_states),
                    )

                    failing_client = failing_clients[0]
                    adapted_dir = TTA_ARTIFACTS_ROOT / f"{failing_client}_{corruption}"
                    adapted_model_path = adapted_dir / "adapted_client_model.pt"
                    if not adapted_model_path.exists():
                        raise FileNotFoundError(f"Missing existing adapted model: {adapted_model_path}")
                    tta_state = load_checkpoint_state(adapted_model_path)

                    weight_keys, bn_keys = state_key_groups(performing_state)
                    model_pref = pairwise_preference_weights(performing_state, tta_state, weight_keys, beta=protected_beta)
                    bn_pref = pairwise_preference_weights(performing_state, tta_state, bn_keys, beta=protected_beta)
                    protected_state = family_specific_hybrid_aggregation(
                        performing_state,
                        tta_state,
                        weight_pref=(model_pref[0], model_pref[1]),
                        bn_pref=(bn_pref[0], bn_pref[1]),
                    )
                    protected_weights = {
                        "performing_weight": model_pref[0],
                        "tta_weight": model_pref[1],
                        "bn_performing_weight": bn_pref[0],
                        "bn_tta_weight": bn_pref[1],
                    }
                    protected_distances = {
                        "model_pref_distance": model_pref[2],
                        "bn_pref_distance": bn_pref[2],
                    }
                protected_model = build_standard_model(protected_state)
                aggregation_time = time.perf_counter() - aggregation_start
                aggregation_time_total += aggregation_time
                eval_start = time.perf_counter()
                protected_accuracy, protected_latency = evaluate_model(
                    protected_model,
                    corrupt_dataset,
                    batch_size=batch_size,
                )
                evaluation_time = time.perf_counter() - eval_start
                evaluation_time_total += evaluation_time
            else:
                aggregation_time = 0.0
                evaluation_time = 0.0

            recovery_ratio = 0.0
            if accuracy_drop > 0:
                recovery_ratio = max(0.0, protected_accuracy - degraded_accuracy) / accuracy_drop

            substitution_rows.append(
                {
                    "profile_id": row["profile_id"],
                    "composition_name": row["composition_name"],
                    "combination_size": row["combination_size"],
                    "corruption_category": corruption,
                    "before_accuracy": baseline_accuracy,
                    "after_corruption_accuracy": degraded_accuracy,
                    "after_substitution_accuracy": protected_accuracy,
                    "accuracy_drop": accuracy_drop,
                    "accuracy_recovery": protected_accuracy - degraded_accuracy,
                    "recovery_ratio": recovery_ratio,
                    "failing_client": json.dumps(failing_clients),
                    "recommended_substitute_service": json.dumps(adapted_cases),
                    "recommended_substitute_client": json.dumps(failing_clients),
                    "recommended_substitute_checkpoint": json.dumps(
                        [str(TTA_ARTIFACTS_ROOT / adapted_case / "adapted_client_model.pt") for adapted_case in adapted_cases]
                    ),
                    "selection_rule": "tta_existing_model_then_simple_aggregation",
                    "selection_score": 1.0,
                    "candidate_pool_size": 1,
                    "candidate_scan_count": 1,
                    "num_substitutions": len(failing_clients),
                    "qos_similarity_ratio": qos_similarity,
                    "data_similarity_ratio": data_similarity,
                    "substitution_similarity_ratio": similarity_ratio,
                    "low_similarity_substitute": similarity_ratio < 60.0,
                    "selection_time_seconds": 0.0,
                    "aggregation_time_seconds": aggregation_time,
                    "evaluation_time_seconds": evaluation_time,
                    "case_time_seconds": time.perf_counter() - case_start,
                    "substituted_latency_seconds": protected_latency,
                    "remaining_clients": json.dumps(remaining_clients),
                    "substituted_client_set": json.dumps(remaining_clients + failing_clients),
                    "compatibility_mean_ttaas": 1.0,
                    "compatibility_predicted_composable": composable,
                    "protected_weights": json.dumps(protected_weights),
                    "protected_probe_distances": json.dumps(protected_distances),
                }
            )

    out_root = output_dir_for_method("tta")
    prefix = output_prefix_for_method("tta")
    write_csv(
        out_root / f"{prefix}_corruption_impact.csv",
        impact_rows,
        [
            "profile_id",
            "composition_name",
            "combination_size",
            "corruption_category",
            "before_accuracy",
            "before_reliability_score",
            "before_latency_seconds",
            "after_corruption_accuracy",
            "after_corruption_latency_seconds",
            "accuracy_drop",
            "weights_file",
        ],
    )
    write_csv(
        out_root / f"{prefix}_substitution_results.csv",
        substitution_rows,
        [
            "profile_id",
            "composition_name",
            "combination_size",
            "corruption_category",
            "before_accuracy",
            "after_corruption_accuracy",
            "after_substitution_accuracy",
            "accuracy_drop",
            "accuracy_recovery",
            "recovery_ratio",
            "failing_client",
            "recommended_substitute_service",
            "recommended_substitute_client",
            "recommended_substitute_checkpoint",
            "selection_rule",
            "selection_score",
            "candidate_pool_size",
            "candidate_scan_count",
            "num_substitutions",
            "qos_similarity_ratio",
            "data_similarity_ratio",
            "substitution_similarity_ratio",
            "low_similarity_substitute",
            "selection_time_seconds",
            "aggregation_time_seconds",
            "evaluation_time_seconds",
            "case_time_seconds",
            "substituted_latency_seconds",
            "remaining_clients",
            "substituted_client_set",
            "compatibility_mean_ttaas",
            "compatibility_predicted_composable",
            "protected_weights",
            "protected_probe_distances",
        ],
    )
    write_json(
        out_root / f"{prefix}_substitution_metadata.json",
        {
            "method": "tta",
            "corruptions": corruptions,
            "num_selected_compositions": len(selected_compositions),
            "num_impact_rows": len(impact_rows),
            "num_substitution_rows": len(substitution_rows),
            "max_eval_samples": max_eval_samples,
            "tta_batch_size": tta_batch_size,
            "tta_learning_rate": tta_learning_rate,
            "tta_max_batches": tta_max_batches,
            "tta_max_target_samples": tta_max_target_samples,
            "compatibility_chunk_size": compatibility_chunk_size,
            "probe_samples_per_corruption": probe_samples_per_corruption,
            "protected_beta": protected_beta,
            "max_substitutions": max_substitutions,
            "target_clean_ratio": target_clean_ratio,
            "mixed_clean_ratio": mixed_clean_ratio,
            "composability_assumption": "same_client_adaptation_is_composable",
            "runtime_seconds": time.perf_counter() - method_start,
            "aggregation_time_seconds_total": aggregation_time_total,
            "evaluation_time_seconds_total": evaluation_time_total,
        },
    )
    return {
        "method": "tta",
        "num_selected_compositions": len(selected_compositions),
        "num_impact_rows": len(impact_rows),
        "num_substitution_rows": len(substitution_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["semantic", "context", "mlaas", "tta", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-eval-samples", type=int, default=5000)
    parser.add_argument("--candidate-pool-multiplier", type=int, default=1)
    parser.add_argument("--max-substitutions", type=int, default=1)
    parser.add_argument("--combination-beam-width", type=int, default=12)
    parser.add_argument("--target-clean-ratio", type=float, default=0.0)
    parser.add_argument("--mixed-clean-ratio", type=float, default=0.8)
    parser.add_argument("--tta-batch-size", type=int, default=128)
    parser.add_argument("--tta-learning-rate", type=float, default=1e-3)
    parser.add_argument("--tta-max-batches", type=int, default=0)
    parser.add_argument("--tta-max-target-samples", type=int, default=2000)
    parser.add_argument("--compatibility-chunk-size", type=int, default=1000)
    parser.add_argument("--probe-samples-per-corruption", type=int, default=200)
    parser.add_argument("--protected-beta", type=float, default=2.0)
    parser.add_argument("--corruptions", nargs="+", default=DEFAULT_CORRUPTIONS)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(1)

    methods = ["semantic", "context", "mlaas", "tta"] if args.method == "all" else [args.method]
    summaries = []
    for method in methods:
        if method == "tta":
            summaries.append(
                run_tta_method(
                    batch_size=args.batch_size,
                    max_eval_samples=args.max_eval_samples,
                    seed=args.seed,
                    tta_batch_size=args.tta_batch_size,
                    tta_learning_rate=args.tta_learning_rate,
                    tta_max_batches=args.tta_max_batches or None,
                    tta_max_target_samples=args.tta_max_target_samples or None,
                    compatibility_chunk_size=args.compatibility_chunk_size,
                    probe_samples_per_corruption=args.probe_samples_per_corruption,
                    protected_beta=args.protected_beta,
                    max_substitutions=args.max_substitutions,
                    target_clean_ratio=args.target_clean_ratio,
                    mixed_clean_ratio=args.mixed_clean_ratio,
                    corruptions=args.corruptions,
                )
            )
        else:
            summaries.append(
                run_method(
                    method,
                    batch_size=args.batch_size,
                    max_eval_samples=args.max_eval_samples,
                    seed=args.seed,
                    candidate_pool_multiplier=args.candidate_pool_multiplier,
                    max_substitutions=args.max_substitutions,
                    combination_beam_width=args.combination_beam_width,
                    target_clean_ratio=args.target_clean_ratio,
                    mixed_clean_ratio=args.mixed_clean_ratio,
                    corruptions=args.corruptions,
                )
            )
    print(json.dumps({"runs": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
