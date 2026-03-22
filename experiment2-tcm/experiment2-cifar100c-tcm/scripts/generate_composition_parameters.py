import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FL.train_fedavg_cifar100 import build_model_for_dataset, get_dataset_spec, load_run_dataset_name


DEFAULT_FL_RUN_DIR = PROJECT_ROOT / "FL" / "artifacts" / "cifar100_fl_baseline_5clients_run_clean"
OUTPUT_ROOT = PROJECT_ROOT / "Performing composition" / "artifacts"
COMPOSITIONS = {
    "C2": ["client_1"],
    "C2_C3": ["client_1", "client_2"],
    "C2_C3_C4": ["client_1", "client_2", "client_3"],
    "C2_C3_C4_C5": ["client_1", "client_2", "client_3", "client_4"],
}
REFERENCE_GLOBAL_ROUND = 10
ALL_CLIENTS = ["client_0", "client_1", "client_2", "client_3", "client_4"]
CLIENT_LABELS = {
    "client_0": "C1",
    "client_1": "C2",
    "client_2": "C3",
    "client_3": "C4",
    "client_4": "C5",
}


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def bn_module_types():
    return (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


class BNStatisticsRecorder:
    def __init__(self, model: nn.Module) -> None:
        self.step_index = -1
        self.records = []
        self.handles = []

        for layer_name, module in model.named_modules():
            if isinstance(module, bn_module_types()):
                self.handles.append(module.register_forward_pre_hook(self._make_hook(layer_name)))

    def _make_hook(self, layer_name: str):
        def hook(module, inputs):
            activations = inputs[0].detach()
            if activations.ndim == 2:
                dims = (0,)
            else:
                dims = (0, 2, 3)

            self.records.append(
                {
                    "step": self.step_index,
                    "layer": layer_name,
                    "batch_mean": activations.mean(dim=dims).cpu(),
                    "batch_var": activations.var(dim=dims, unbiased=False).cpu(),
                }
            )

        return hook

    def begin_step(self) -> None:
        self.step_index += 1

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def summarize(self) -> Dict[str, Dict[str, object]]:
        summary: Dict[str, Dict[str, object]] = {}
        for record in self.records:
            layer_name = record["layer"]
            layer_summary = summary.setdefault(layer_name, {"batch_means": [], "batch_vars": []})
            layer_summary["batch_means"].append(record["batch_mean"])
            layer_summary["batch_vars"].append(record["batch_var"])

        result = {}
        for layer_name, values in summary.items():
            means_tensor = torch.stack(values["batch_means"])
            vars_tensor = torch.stack(values["batch_vars"])
            result[layer_name] = {
                "num_steps": int(means_tensor.shape[0]),
                "mean_of_batch_means": means_tensor.mean(dim=0).tolist(),
                "std_of_batch_means": means_tensor.std(dim=0, unbiased=False).tolist(),
                "mean_of_batch_vars": vars_tensor.mean(dim=0).tolist(),
                "std_of_batch_vars": vars_tensor.std(dim=0, unbiased=False).tolist(),
            }
        return result


@torch.no_grad()
def evaluate_prediction_distribution(model: nn.Module, dataloader: DataLoader) -> List[float]:
    model.eval()
    probability_sum = None
    total_examples = 0
    for images, _ in dataloader:
        probabilities = model(images).softmax(dim=1)
        if probability_sum is None:
            probability_sum = probabilities.sum(dim=0)
        else:
            probability_sum += probabilities.sum(dim=0)
        total_examples += images.size(0)

    if probability_sum is None or total_examples == 0:
        return []
    return (probability_sum / total_examples).cpu().tolist()


def load_checkpoint_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint["model_state"], checkpoint


def resolve_reference_round(fl_run_dir: Path, requested_round: int) -> int:
    requested_path = fl_run_dir / "global" / f"global_model_round_{requested_round}.pt"
    if requested_path.exists():
        return requested_round

    available_rounds = []
    for path in (fl_run_dir / "global").glob("global_model_round_*.pt"):
        try:
            available_rounds.append(int(path.stem.rsplit("_", 1)[1]))
        except ValueError:
            continue
    if not available_rounds:
        raise FileNotFoundError(f"No global round checkpoints were found in {fl_run_dir / 'global'}")
    return max(available_rounds)


def flatten_state_delta(before_state: Dict[str, torch.Tensor], after_state: Dict[str, torch.Tensor]):
    return {
        name: (after_state[name].detach().cpu() - before_state[name].detach().cpu()).reshape(-1).tolist()
        for name in after_state.keys()
        if "num_batches_tracked" not in name
    }


def extract_bn_affine_from_model(model: nn.Module):
    bn_affine = {}
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and module.affine:
            bn_affine[module_name] = {
                "gamma": module.weight.detach().cpu().reshape(-1).tolist(),
                "beta": module.bias.detach().cpu().reshape(-1).tolist(),
            }
    return bn_affine


def weighted_average_state(states: List[Dict[str, torch.Tensor]], weights: List[float]):
    total = float(sum(weights))
    averaged = {}
    for key in states[0].keys():
        sample_tensor = states[0][key]
        if sample_tensor.is_floating_point():
            accumulator = torch.zeros_like(sample_tensor, dtype=torch.float32)
            for state, weight in zip(states, weights):
                accumulator += state[key].detach().cpu().to(torch.float32) * (weight / total)
            averaged[key] = accumulator.to(sample_tensor.dtype)
        else:
            accumulator = torch.zeros_like(sample_tensor, dtype=torch.float64)
            for state, weight in zip(states, weights):
                accumulator += state[key].detach().cpu().to(torch.float64) * (weight / total)
            averaged[key] = accumulator.round().to(sample_tensor.dtype)
    return averaged


def build_model_from_state(state_dict: Dict[str, torch.Tensor], dataset_name: str) -> nn.Module:
    model = build_model_for_dataset(dataset_name)
    model.load_state_dict(state_dict)
    return model


def load_clean_test_dataloader(dataset_name: str):
    dataset_spec = get_dataset_spec(dataset_name)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(dataset_spec["mean"], dataset_spec["std"]),
        ]
    )
    test_dataset = dataset_spec["torchvision_class"](
        root=PROJECT_ROOT / "Data",
        train=False,
        download=False,
        transform=transform,
    )
    return DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0), len(test_dataset)


def configure_composition_for_bn_observation(model: nn.Module) -> nn.Module:
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
    return model


@torch.no_grad()
def collect_bn_mean_var(model: nn.Module, dataloader: DataLoader):
    recorder = BNStatisticsRecorder(model)
    for images, _ in dataloader:
        recorder.begin_step()
        _ = model(images)
    recorder.close()
    summary = recorder.summarize()
    return {
        layer_name: {
            "mean": layer_summary["mean_of_batch_means"],
            "var": layer_summary["mean_of_batch_vars"],
        }
        for layer_name, layer_summary in summary.items()
    }, summary, recorder.records


def generate_one(composition_name: str, client_names: List[str], output_root: Path, fl_run_dir: Path):
    output_dir = output_root / composition_name
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = load_run_dataset_name(fl_run_dir)
    dataset_spec = get_dataset_spec(dataset_name)
    clean_dataset_label = f"clean_{dataset_name}_test"
    reference_round = resolve_reference_round(fl_run_dir, REFERENCE_GLOBAL_ROUND)

    client_states = []
    client_weights = []
    for client_name in client_names:
        state, checkpoint = load_checkpoint_model(fl_run_dir / "clients" / f"{client_name}.pt")
        client_states.append(state)
        client_weights.append(float(checkpoint["train_metrics"]["num_samples"]))

    composition_state = weighted_average_state(client_states, client_weights)
    reference_before_state, _ = load_checkpoint_model(
        fl_run_dir / "global" / f"global_model_round_{reference_round}.pt"
    )

    composition_model = build_model_from_state(composition_state, dataset_name)
    composition_weights = {
        name: parameter.detach().cpu().reshape(-1).tolist()
        for name, parameter in composition_model.state_dict().items()
        if "running_" not in name and "num_batches_tracked" not in name
    }
    composition_updates = flatten_state_delta(reference_before_state, composition_state)
    composition_bn_affine = extract_bn_affine_from_model(composition_model)

    dataloader, num_target_samples = load_clean_test_dataloader(dataset_name)

    prediction_model = build_model_from_state(composition_state, dataset_name)
    prediction_distribution = evaluate_prediction_distribution(prediction_model, dataloader)

    bn_model = build_model_from_state(composition_state, dataset_name)
    bn_model = configure_composition_for_bn_observation(bn_model)
    bn_mean_var, bn_statistics_summary, bn_batch_records = collect_bn_mean_var(bn_model, dataloader)

    torch.save(
        {
            "composition_name": composition_name,
            "client_names": client_names,
            "dataset": clean_dataset_label,
            "dataset_name": dataset_name,
            "model_state": composition_state,
        },
        output_dir / "composition_model.pt",
    )
    torch.save(composition_weights, output_dir / "wdm_composition_weights.pt")
    torch.save(composition_updates, output_dir / "ucs_composition_layer_updates.pt")
    torch.save(bn_mean_var, output_dir / "bndas_composition_bn_mean_var.pt")
    torch.save(composition_bn_affine, output_dir / "bnuas_composition_bn_gamma_beta.pt")
    torch.save(prediction_distribution, output_dir / "pdam_composition_prediction_distribution.pt")
    torch.save(bn_batch_records, output_dir / "bn_batch_statistics.pt")

    save_json(output_dir / "wdm_composition_weights.json", composition_weights)
    save_json(output_dir / "ucs_composition_layer_updates.json", composition_updates)
    save_json(output_dir / "bndas_composition_bn_mean_var.json", bn_mean_var)
    save_json(output_dir / "bnuas_composition_bn_gamma_beta.json", composition_bn_affine)
    save_json(output_dir / "pdam_composition_prediction_distribution.json", prediction_distribution)
    save_json(output_dir / "bn_statistics_summary.json", bn_statistics_summary)
    save_json(
        output_dir / "config.json",
        {
            "composition_name": composition_name,
            "client_names": client_names,
            "dataset": clean_dataset_label,
            "dataset_name": dataset_name,
            "fl_run_dir": str(fl_run_dir),
            "reference_global_round": reference_round,
            "aggregation": "sample-weighted-fedavg",
        },
    )
    save_json(
        output_dir / "summary.json",
        {
            "composition_name": composition_name,
            "client_names": client_names,
            "dataset": clean_dataset_label,
            "dataset_name": dataset_name,
            "num_target_samples": num_target_samples,
            "separate_artifacts": {
                "wdm_weights": "wdm_composition_weights.json",
                "ucs_layer_updates": "ucs_composition_layer_updates.json",
                "bndas_bn_mean_var": "bndas_composition_bn_mean_var.json",
                "bnuas_bn_gamma_beta": "bnuas_composition_bn_gamma_beta.json",
                "pdam_prediction_distribution": "pdam_composition_prediction_distribution.json",
            },
            "output_dir": str(output_dir),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("legacy", "expanded"),
        default="legacy",
        help="Generate the original four compositions or all non-empty clean-service subsets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output directory. Defaults depend on the selected mode.",
    )
    parser.add_argument(
        "--fl-run-dir",
        type=Path,
        default=DEFAULT_FL_RUN_DIR,
        help="FL artifact directory containing the client/global checkpoints to compose.",
    )
    args = parser.parse_args()

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    fl_run_dir = args.fl_run_dir.resolve()
    if args.mode == "legacy":
        output_root = args.output_root or OUTPUT_ROOT
        compositions = COMPOSITIONS
    else:
        output_root = args.output_root or (PROJECT_ROOT / "Performing composition" / "artifacts_expanded")
        compositions = {}
        for subset_size in range(1, len(ALL_CLIENTS) + 1):
            for subset in itertools.combinations(ALL_CLIENTS, subset_size):
                composition_name = "_".join(CLIENT_LABELS[client_name] for client_name in subset)
                compositions[composition_name] = list(subset)

    output_root.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for composition_name, client_names in compositions.items():
        generate_one(
            composition_name=composition_name,
            client_names=client_names,
            output_root=output_root,
            fl_run_dir=fl_run_dir,
        )
        print(f"{composition_name} | clean_{load_run_dataset_name(fl_run_dir)}_test", flush=True)
        index_rows.append(
            {
                "composition_name": composition_name,
                "client_names": client_names,
                "contains_original_c1": "client_0" in client_names,
                "num_clients": len(client_names),
                "output_dir": str(output_root / composition_name),
            }
        )

    save_json(output_root / "index.json", index_rows)


if __name__ == "__main__":
    main()
