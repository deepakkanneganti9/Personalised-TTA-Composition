import csv
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parent
FL_SCRIPT = PROJECT_ROOT / "FL" / "train_fedavg_mnist.py"
TTA_PACKAGE_ROOT = PROJECT_ROOT / "TTA techniques"
MNIST_C_ROOT = PROJECT_ROOT / "data" / "mnist_c"
MULTI_RESULTS = PROJECT_ROOT / "experiment_artifacts" / "reviewer_composition_multi_tta_v2" / "reviewer_composition_multi_tta_v2.csv"
SINGLE_RESULTS = PROJECT_ROOT / "experiment_artifacts" / "reviewer_composition_table_with_substitution_v2" / "reviewer_composition_with_substitution_v2.csv"
SUBSTITUTION_ROOT = PROJECT_ROOT / "federated_artifact_corrupt"
STMU_FILTERED_RESULTS = PROJECT_ROOT / "STMU" / "artifacts" / "filtered_same_scale" / "filtered_same_scale.csv"
OUTPUT_ROOT = PROJECT_ROOT / "experiment_artifacts" / "filtered_improved_random_substitution"


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fl_module = load_module(FL_SCRIPT, "fl_filtered_improved_random_substitution")
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


def build_eval_dataset(corruption: str, clean_samples: int, corruption_samples: int):
    clean = maybe_subset_dataset(load_clean_test_dataset(), clean_samples)
    corrupted = maybe_subset_dataset(
        tent_module.LocalMNISTC(
            root=MNIST_C_ROOT,
            corruption=corruption,
            split="test",
            allowed_digits=None,
        ),
        corruption_samples,
    )
    return ConcatDataset([clean, corrupted])


def load_checkpoint_state(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint["model_state"] if "model_state" in checkpoint else checkpoint


def build_standard_model(state_dict):
    model = fl_module.MNISTCNN()
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_model(model, dataset: Dataset, batch_size: int = 128):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return tent_module.evaluate_accuracy(model, loader)


def deterministic_pick(paths: List[Path], token: str) -> Path:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(paths)
    return paths[index]


def composition_id_from_performing_clients(performing_clients: str) -> str:
    ids = [client.strip().split("_")[1] for client in performing_clients.split(",") if client.strip()]
    return "composition_" + "_".join(ids)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    multi_rows = list(csv.DictReader(MULTI_RESULTS.open()))
    single_lookup = {
        row["case"]: {
            "deployment_accuracy": float(row["initial_clean_accuracy"]),
            "data_shift_accuracy": float(row["initial_corruption_accuracy"]),
        }
        for row in csv.DictReader(SINGLE_RESULTS.open())
    }
    stmu_lookup = {
        (row["composition_id"], row["corruption"]): {
            "stmu_gala_mixed_accuracy": float(row["composition_gala_mixed_accuracy"]),
            "stmu_gala_gain_mixed": float(row["composition_gala_gain_mixed"]),
        }
        for row in csv.DictReader(STMU_FILTERED_RESULTS.open())
    }

    improved_rows = [row for row in multi_rows if float(row["protected_v2_gain_vs_initial_mixed"]) > 0]

    sub20_pool = sorted((SUBSTITUTION_ROOT / name / "global" / "global_model.pt") for name in ["clean_fog", "clean_spatter", "clean_stripe", "clean_translate", "clean_zigzag"])
    sub50_pool = sorted((SUBSTITUTION_ROOT / name / "global" / "global_model.pt") for name in ["fog", "spatter", "stripe", "translate", "zigzag"])

    out_rows = []
    for row in improved_rows:
        case = row["case"]
        corruption = row["tta_corruption"]
        dataset = build_eval_dataset(corruption, int(row["clean_samples"]), int(row["corruption_samples"]))

        sub20_path = deterministic_pick(sub20_pool, f"{row['tta_method']}::{case}::sub20")
        sub50_path = deterministic_pick(sub50_pool, f"{row['tta_method']}::{case}::sub50")
        sub20_acc = evaluate_model(build_standard_model(load_checkpoint_state(sub20_path)), dataset)
        sub50_acc = evaluate_model(build_standard_model(load_checkpoint_state(sub50_path)), dataset)

        deployment = single_lookup[case]["deployment_accuracy"]
        data_shift = single_lookup[case]["data_shift_accuracy"]
        initial_mixed = float(row["initial_mixed_accuracy"])
        protected_v2 = float(row["protected_v2_mixed_accuracy"])
        composition_id = composition_id_from_performing_clients(row["performing_clients"])
        stmu = stmu_lookup[(composition_id, corruption)]

        out_rows.append(
            {
                "tta_method": row["tta_method"],
                "case": case,
                "performing_length": int(row["performing_length"]),
                "tta_corruption": corruption,
                "deployment_accuracy": deployment,
                "data_shift_accuracy": data_shift,
                "initial_mixed_accuracy": initial_mixed,
                "protected_v2_mixed_accuracy": protected_v2,
                "stmu_gala_mixed_accuracy": stmu["stmu_gala_mixed_accuracy"],
                "substitution_20_random_accuracy": sub20_acc,
                "substitution_50_random_accuracy": sub50_acc,
                "protected_v2_gain_vs_initial": protected_v2 - initial_mixed,
                "stmu_gala_gain_vs_initial": stmu["stmu_gala_mixed_accuracy"] - initial_mixed,
                "sub20_gain_vs_initial": sub20_acc - initial_mixed,
                "sub50_gain_vs_initial": sub50_acc - initial_mixed,
                "sub20_source": str(sub20_path),
                "sub50_source": str(sub50_path),
            }
        )

    with (OUTPUT_ROOT / "filtered_improved_random_substitution.json").open("w", encoding="utf-8") as handle:
        json.dump(out_rows, handle, indent=2)
    with (OUTPUT_ROOT / "filtered_improved_random_substitution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)


if __name__ == "__main__":
    main()
