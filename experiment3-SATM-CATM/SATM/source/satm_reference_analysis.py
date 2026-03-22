import argparse
import csv
import importlib
import importlib.util
import itertools
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parent
FL_SCRIPT = PROJECT_ROOT / "FL" / "train_fedavg_mnist.py"
TTA_PACKAGE_ROOT = PROJECT_ROOT / "TTA techniques"
DEFAULT_FL_RUN_DIR = PROJECT_ROOT / "FL" / "artifacts" / "mnist_fl_baseline_5clients_run"
PRECOMPUTED_TTA_ROOT = PROJECT_ROOT / "TTA techniques" / "artifacts"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiment_artifacts" / "reviewer_composition_table_with_substitution_v2"
DEFAULT_MNIST_C_ROOT = PROJECT_ROOT / "data" / "mnist_c"
DEFAULT_SUBSTITUTION_ROOT = PROJECT_ROOT / "federated_artifact_corrupt"
FALLBACK_SUBSTITUTION_MAP = {
    "brightness": ("clean_translate", "translate"),
    "glass_blur": ("clean_fog", "fog"),
    "impulse_noise": ("clean_spatter", "spatter"),
    "stripe": ("clean_stripe", "stripe"),
    "zigzag": ("clean_zigzag", "zigzag"),
}


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fl_module = load_module(FL_SCRIPT, "fl_reviewer_substitution_v2")
if str(TTA_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTA_PACKAGE_ROOT))
tent_module = importlib.import_module("tta_techniques.tent_adapter")


class ConcatDataset(Dataset):
    def __init__(self, datasets_list: List[Dataset]) -> None:
        self.datasets_list = datasets_list
        self.offsets = []
        total = 0
        for dataset in datasets_list:
            self.offsets.append(total)
            total += len(dataset)
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int):
        for dataset, offset in zip(self.datasets_list, self.offsets):
            if index < offset + len(dataset):
                return dataset[index - offset]
        raise IndexError(index)


def make_clean_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def load_clean_test_dataset():
    return datasets.MNIST(
        root=PROJECT_ROOT / "data",
        train=False,
        download=False,
        transform=make_clean_transform(),
    )


def maybe_subset_dataset(dataset: Dataset, max_samples: int):
    if len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, list(range(max_samples)))


def build_eval_dataset(mnist_c_root: Path, corruption: str, clean_samples: int, corruption_samples: int):
    clean = maybe_subset_dataset(load_clean_test_dataset(), clean_samples)
    corrupted = maybe_subset_dataset(
        tent_module.LocalMNISTC(
            root=mnist_c_root,
            corruption=corruption,
            split="test",
            allowed_digits=None,
        ),
        corruption_samples,
    )
    return ConcatDataset([clean, corrupted])


def build_clean_eval_dataset(clean_samples: int):
    return maybe_subset_dataset(load_clean_test_dataset(), clean_samples)


def build_corruption_eval_dataset(mnist_c_root: Path, corruption: str, corruption_samples: int):
    return maybe_subset_dataset(
        tent_module.LocalMNISTC(
            root=mnist_c_root,
            corruption=corruption,
            split="test",
            allowed_digits=None,
        ),
        corruption_samples,
    )


def load_checkpoint_state(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint["model_state"] if "model_state" in checkpoint else checkpoint


def build_standard_model(state_dict):
    model = fl_module.MNISTCNN()
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_model(model, dataset: Dataset, batch_size: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return tent_module.evaluate_accuracy(model, loader)


def aggregate_states_with_weights(states: List[Dict[str, torch.Tensor]], weights: List[float]):
    out = {}
    keys = states[0].keys()
    for name in keys:
        if name.endswith("num_batches_tracked"):
            out[name] = states[0][name].clone()
            continue
        acc = None
        for state, weight in zip(states, weights):
            term = state[name] * weight
            acc = term if acc is None else acc + term
        out[name] = acc
    return out


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


def flatten_subset(state: Dict[str, torch.Tensor], keys: Iterable[str]) -> torch.Tensor:
    parts = [state[key].detach().cpu().reshape(-1).float() for key in keys]
    if not parts:
        return torch.zeros(1)
    return torch.cat(parts)


def pairwise_preference_weights(
    performing_state: Dict[str, torch.Tensor],
    tta_state: Dict[str, torch.Tensor],
    keys: Iterable[str],
    beta: float,
) -> Tuple[float, float, float]:
    perf_vec = flatten_subset(performing_state, keys)
    tta_vec = flatten_subset(tta_state, keys)
    distance = torch.norm(perf_vec - tta_vec, p=2).item()
    tta_score = float(torch.exp(torch.tensor(-beta * distance)).item())
    perf_weight = 1.0 / (1.0 + tta_score)
    tta_weight = tta_score / (1.0 + tta_score)
    return perf_weight, tta_weight, distance


def discover_corruptions(tta_root: Path, client_names: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for client_name in client_names:
        prefix = f"{client_name}_"
        corruptions = []
        for path in sorted(tta_root.glob(f"{prefix}*")):
            if path.is_dir() and (path / "adapted_client_model.pt").exists() and (path / "summary.json").exists():
                corruptions.append(path.name[len(prefix) :])
        out[client_name] = corruptions
    return out


def generate_cases(client_names: List[str], client_to_corruptions: Dict[str, List[str]], min_len: int, max_len: int):
    cases = []
    for tta_client in client_names:
        remaining = [name for name in client_names if name != tta_client]
        for corruption in client_to_corruptions.get(tta_client, []):
            for length in range(min_len, max_len + 1):
                for combo in itertools.combinations(remaining, length):
                    cases.append(
                        {
                            "case": f"perf_{'_'.join(name.split('_')[1] for name in combo)}_plus_{tta_client.split('_')[1]}_on_{corruption}",
                            "performing_clients": list(combo),
                            "performing_length": length,
                            "tta_client": tta_client,
                            "tta_corruption": corruption,
                        }
                    )
    return cases


def resolve_substitution_paths(sub_root: Path, corruption: str) -> Tuple[Optional[Path], Optional[Path]]:
    exact_twenty = sub_root / f"clean_{corruption}" / "global" / "global_model.pt"
    exact_fifty = sub_root / corruption / "global" / "global_model.pt"
    if exact_twenty.exists() and exact_fifty.exists():
        return exact_twenty, exact_fifty
    fallback_twenty_name, fallback_fifty_name = FALLBACK_SUBSTITUTION_MAP.get(corruption, (None, None))
    fallback_twenty = (sub_root / fallback_twenty_name / "global" / "global_model.pt") if fallback_twenty_name else None
    fallback_fifty = (sub_root / fallback_fifty_name / "global" / "global_model.pt") if fallback_fifty_name else None
    return (
        fallback_twenty if fallback_twenty and fallback_twenty.exists() else None,
        fallback_fifty if fallback_fifty and fallback_fifty.exists() else None,
    )


def maybe_eval_state(checkpoint_path: Optional[Path], dataset: Dataset, batch_size: int) -> Optional[float]:
    if checkpoint_path is None:
        return None
    state = load_checkpoint_state(checkpoint_path)
    model = build_standard_model(state)
    return evaluate_model(model, dataset, batch_size=batch_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fl-run-dir", default=str(DEFAULT_FL_RUN_DIR))
    parser.add_argument("--mnist-c-root", default=str(DEFAULT_MNIST_C_ROOT))
    parser.add_argument("--substitution-root", default=str(DEFAULT_SUBSTITUTION_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--clean-eval-samples", type=int, default=50)
    parser.add_argument("--corruption-eval-samples", type=int, default=50)
    parser.add_argument("--min-performing-length", type=int, default=1)
    parser.add_argument("--max-performing-length", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(min(4, torch.get_num_threads()))
    fl_run_dir = Path(args.fl_run_dir).resolve()
    mnist_c_root = Path(args.mnist_c_root).resolve()
    substitution_root = Path(args.substitution_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    client_indices = fl_module.load_json(fl_run_dir / "client_indices.json")
    client_names = sorted(client_indices.keys())
    base_states = {name: load_checkpoint_state(fl_run_dir / "clients" / f"{name}.pt") for name in client_names}

    client_to_corruptions = discover_corruptions(PRECOMPUTED_TTA_ROOT, client_names)
    cases = generate_cases(client_names, client_to_corruptions, args.min_performing_length, args.max_performing_length)

    rows = []
    for case in cases:
        corruption = case["tta_corruption"]
        dataset = build_eval_dataset(mnist_c_root, corruption, args.clean_eval_samples, args.corruption_eval_samples)
        clean_dataset = build_clean_eval_dataset(args.clean_eval_samples)
        corruption_dataset = build_corruption_eval_dataset(mnist_c_root, corruption, args.corruption_eval_samples)

        performing_clients = case["performing_clients"]
        tta_client = case["tta_client"]

        initial_client_states = [base_states[name] for name in performing_clients] + [base_states[tta_client]]
        initial_full_composition = aggregate_states_with_weights(
            initial_client_states,
            [1.0 / len(initial_client_states)] * len(initial_client_states),
        )

        performing_states = [base_states[name] for name in performing_clients]
        performing_state = aggregate_states_with_weights(
            performing_states,
            [1.0 / len(performing_states)] * len(performing_states),
        )

        tta_dir = PRECOMPUTED_TTA_ROOT / f"{tta_client}_{corruption}"
        tta_state = deepcopy(base_states[tta_client])
        tta_state.update(load_checkpoint_state(tta_dir / "adapted_client_model.pt"))
        summary = fl_module.load_json(tta_dir / "summary.json")

        weight_keys, bn_keys = state_key_groups(performing_state)
        model_pref = pairwise_preference_weights(performing_state, tta_state, weight_keys, beta=args.beta)
        bn_pref = pairwise_preference_weights(performing_state, tta_state, bn_keys, beta=args.beta)
        sum_tta = min(1.0, model_pref[1] + bn_pref[1])
        sum_perf = 1.0 - sum_tta

        fedavg_after_tta = aggregate_states_with_weights([performing_state, tta_state], [0.5, 0.5])
        protected_v2_after_tta = aggregate_states_with_weights([performing_state, tta_state], [sum_perf, sum_tta])

        initial_model = build_standard_model(initial_full_composition)
        initial_acc = evaluate_model(initial_model, dataset, batch_size=args.batch_size)
        initial_clean_acc = evaluate_model(initial_model, clean_dataset, batch_size=args.batch_size)
        initial_corruption_acc = evaluate_model(initial_model, corruption_dataset, batch_size=args.batch_size)
        fedavg_model = build_standard_model(fedavg_after_tta)
        fedavg_acc = evaluate_model(fedavg_model, dataset, batch_size=args.batch_size)
        fedavg_corruption_acc = evaluate_model(fedavg_model, corruption_dataset, batch_size=args.batch_size)
        protected_v2_model = build_standard_model(protected_v2_after_tta)
        protected_v2_acc = evaluate_model(protected_v2_model, dataset, batch_size=args.batch_size)
        protected_v2_corruption_acc = evaluate_model(protected_v2_model, corruption_dataset, batch_size=args.batch_size)

        sub20_path, sub50_path = resolve_substitution_paths(substitution_root, corruption)
        sub20_acc = maybe_eval_state(sub20_path, dataset, batch_size=args.batch_size)
        sub50_acc = maybe_eval_state(sub50_path, dataset, batch_size=args.batch_size)

        row = {
            "case": case["case"],
            "performing_clients": ",".join(performing_clients),
            "performing_length": case["performing_length"],
            "tta_client": tta_client,
            "tta_corruption": corruption,
            "clean_samples": args.clean_eval_samples,
            "corruption_samples": args.corruption_eval_samples,
            "tta_local_before": summary["baseline_accuracy"],
            "tta_local_after": summary["adapted_accuracy"],
            "initial_composition_accuracy": initial_acc,
            "initial_clean_accuracy": initial_clean_acc,
            "initial_corruption_accuracy": initial_corruption_acc,
            "fedavg_after_tta_accuracy": fedavg_acc,
            "fedavg_corruption_accuracy": fedavg_corruption_acc,
            "protected_v2_after_tta_accuracy": protected_v2_acc,
            "protected_v2_corruption_accuracy": protected_v2_corruption_acc,
            "substitution_20_similarity_accuracy": sub20_acc,
            "substitution_50_similarity_accuracy": sub50_acc,
            "protected_v2_gain_vs_initial": protected_v2_acc - initial_acc,
            "protected_v2_gain_vs_fedavg": protected_v2_acc - fedavg_acc,
            "sub20_gain_vs_initial": (sub20_acc - initial_acc) if sub20_acc is not None else None,
            "sub50_gain_vs_initial": (sub50_acc - initial_acc) if sub50_acc is not None else None,
            "sub20_gain_vs_protected_v2": (sub20_acc - protected_v2_acc) if sub20_acc is not None else None,
            "sub50_gain_vs_protected_v2": (sub50_acc - protected_v2_acc) if sub50_acc is not None else None,
            "substitution_20_source": str(sub20_path) if sub20_path else "",
            "substitution_50_source": str(sub50_path) if sub50_path else "",
            "model_pref_performing_weight": model_pref[0],
            "model_pref_tta_weight": model_pref[1],
            "bn_pref_performing_weight": bn_pref[0],
            "bn_pref_tta_weight": bn_pref[1],
            "sum_pref_performing_weight": sum_perf,
            "sum_pref_tta_weight": sum_tta,
        }
        rows.append(row)

    with (output_root / "reviewer_composition_with_substitution_v2.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    with (output_root / "reviewer_composition_with_substitution_v2.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
