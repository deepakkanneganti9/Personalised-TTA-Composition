import json
import os
import sys
import argparse
import itertools
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TTA_DIR = PROJECT_ROOT / "TTA techniques"
if str(TTA_DIR) not in sys.path:
    sys.path.insert(0, str(TTA_DIR))

from tta_techniques.tent_grad_adapter import (  # noqa: E402
    BNStatisticsRecorder,
    LocalMNISTC,
    MNISTCNN,
    evaluate_prediction_distribution,
    save_json,
    snapshot_model_parameters,
)


FL_RUN_DIR = PROJECT_ROOT / "FL" / "artifacts" / "mnist_fl_baseline_5clients_run"
OUTPUT_ROOT = PROJECT_ROOT / "Performing composition" / "artifacts"
COMPOSITIONS = {
    "C2": ["client_1"],
    "C2_C3": ["client_1", "client_2"],
    "C2_C3_C4": ["client_1", "client_2", "client_3"],
    "C2_C3_C4_C5": ["client_1", "client_2", "client_3", "client_4"],
}
REFERENCE_GLOBAL_ROUND = 9
ALL_CLIENTS = ["client_0", "client_1", "client_2", "client_3", "client_4"]
CLIENT_LABELS = {
    "client_0": "C1",
    "client_1": "C2",
    "client_2": "C3",
    "client_3": "C4",
    "client_4": "C5",
}


def load_checkpoint_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint["model_state"], checkpoint


def flatten_state_delta(before_state: Dict[str, torch.Tensor], after_state: Dict[str, torch.Tensor]):
    return {
        name: (after_state[name].detach().cpu() - before_state[name].detach().cpu()).reshape(-1).tolist()
        for name in after_state.keys()
        if "num_batches_tracked" not in name
    }


def extract_bn_affine_from_state(state_dict: Dict[str, torch.Tensor]):
    bn_affine = {}
    for key, value in state_dict.items():
        if key.endswith(".weight"):
            prefix = key.rsplit(".", 1)[0]
            bias_key = f"{prefix}.bias"
            if bias_key in state_dict and ("features.1" in prefix or "features.5" in prefix or "bn" in prefix.lower()):
                bn_affine[prefix] = {
                    "gamma": state_dict[key].detach().cpu().reshape(-1).tolist(),
                    "beta": state_dict[bias_key].detach().cpu().reshape(-1).tolist(),
                }
    return bn_affine


def weighted_average_state(states: List[Dict[str, torch.Tensor]], weights: List[float]):
    total = float(sum(weights))
    averaged = {}
    for key in states[0].keys():
        averaged[key] = sum(state[key].detach().cpu() * (weight / total) for state, weight in zip(states, weights))
    return averaged


def build_model_from_state(state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    model = MNISTCNN()
    model.load_state_dict(state_dict)
    return model


def load_clean_test_dataloader():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_dataset = datasets.MNIST(
        root=PROJECT_ROOT / "data",
        train=False,
        download=False,
        transform=transform,
    )
    return DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0), len(test_dataset)


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
    for batch_index, (images, _) in enumerate(dataloader):
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


def generate_one(composition_name: str, client_names: List[str], output_root: Path):
    output_dir = output_root / composition_name
    output_dir.mkdir(parents=True, exist_ok=True)

    client_states = []
    client_weights = []
    for client_name in client_names:
        state, checkpoint = load_checkpoint_model(FL_RUN_DIR / "clients" / f"{client_name}.pt")
        client_states.append(state)
        if "num_samples" in checkpoint:
            client_weights.append(float(checkpoint["num_samples"]))
        else:
            client_weights.append(float(checkpoint["train_metrics"]["num_samples"]))

    composition_state = weighted_average_state(client_states, client_weights)
    reference_before_state, _ = load_checkpoint_model(
        FL_RUN_DIR / "global" / f"global_model_round_{REFERENCE_GLOBAL_ROUND}.pt"
    )

    composition_weights = snapshot_model_parameters(build_model_from_state(composition_state))
    composition_updates = flatten_state_delta(reference_before_state, composition_state)
    composition_bn_affine = extract_bn_affine_from_state(composition_state)

    dataloader, num_target_samples = load_clean_test_dataloader()

    prediction_model = build_model_from_state(composition_state)
    prediction_distribution = evaluate_prediction_distribution(prediction_model, dataloader)

    bn_model = build_model_from_state(composition_state)
    bn_model = configure_composition_for_bn_observation(bn_model)
    bn_mean_var, bn_statistics_summary, bn_batch_records = collect_bn_mean_var(bn_model, dataloader)

    torch.save(
        {
            "composition_name": composition_name,
            "client_names": client_names,
            "dataset": "clean_mnist_test",
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
            "dataset": "clean_mnist_test",
            "fl_run_dir": str(FL_RUN_DIR),
            "reference_global_round": REFERENCE_GLOBAL_ROUND,
            "aggregation": "sample-weighted-fedavg",
        },
    )
    save_json(
        output_dir / "summary.json",
        {
            "composition_name": composition_name,
            "client_names": client_names,
            "dataset": "clean_mnist_test",
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
    args = parser.parse_args()

    torch.set_num_threads(min(4, os.cpu_count() or 1))
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
        generate_one(composition_name=composition_name, client_names=client_names, output_root=output_root)
        print(f"{composition_name} | clean_mnist_test", flush=True)
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
