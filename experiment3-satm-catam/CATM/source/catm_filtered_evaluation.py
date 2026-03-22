import csv
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch


STMU_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STMU_ROOT.parent
SL_RESULTS = PROJECT_ROOT / "experiment_artifacts" / "reviewer_composition_multi_tta_v2" / "reviewer_composition_multi_tta_v2.csv"
COMPOSITION_MODEL_ROOT = STMU_ROOT / "artifacts" / "composition_baselines" / "composition_models"
OUTPUT_ROOT = STMU_ROOT / "artifacts" / "filtered_same_scale"


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


multi_tta_module = load_module(STMU_ROOT / "evaluate_composition_multi_tta_methods.py", "stmu_filtered_multi_tta")
gala_module = load_module(STMU_ROOT / "evaluate_composition_gala_tta.py", "stmu_filtered_gala")


def composition_id_from_clients(performing_clients: str) -> str:
    ids = [client.strip().split("_")[1] for client in performing_clients.split(",") if client.strip()]
    return "composition_" + "_".join(ids)


def load_improved_contexts() -> Set[Tuple[str, str]]:
    rows = list(csv.DictReader(SL_RESULTS.open()))
    contexts = set()
    for row in rows:
        if float(row["protected_v2_gain_vs_initial_mixed"]) > 0:
            contexts.add((composition_id_from_clients(row["performing_clients"]), row["tta_corruption"]))
    return contexts


def evaluate_method_rows() -> List[Dict]:
    clean_samples = 50
    corruption_samples = 50
    contexts = load_improved_contexts()
    composition_to_corruptions: Dict[str, Set[str]] = {}
    for composition_id, corruption in contexts:
        composition_to_corruptions.setdefault(composition_id, set()).add(corruption)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    clean_dataset = multi_tta_module.load_clean_test_dataset(clean_samples)
    corruption_cache = {}
    rows: List[Dict] = []

    for composition_path in sorted(COMPOSITION_MODEL_ROOT.glob("composition_*.pt")):
        checkpoint = torch.load(composition_path, map_location="cpu")
        state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
        clients = checkpoint.get("clients", [])
        composition_id = composition_path.stem
        composition_length = len(clients)

        for corruption in sorted(composition_to_corruptions.get(composition_id, set())):

            if corruption not in corruption_cache:
                corruption_cache[corruption] = multi_tta_module.load_corruption_test_dataset(corruption, corruption_samples)
            corruption_dataset = corruption_cache[corruption]
            mixed_dataset = multi_tta_module.EvalDataset([clean_dataset, corruption_dataset])

            base_model = multi_tta_module.build_model(deepcopy(state_dict))
            before_clean_acc = multi_tta_module.evaluate_model(base_model, clean_dataset)
            before_corruption_acc = multi_tta_module.evaluate_model(base_model, corruption_dataset)
            before_mixed_acc = multi_tta_module.evaluate_model(base_model, mixed_dataset)

            method_models = {
                "composition_tent": multi_tta_module.adapt_with_tent(deepcopy(state_dict), corruption_dataset),
                "composition_tta_bn": multi_tta_module.adapt_with_tta_bn(deepcopy(state_dict), corruption_dataset),
                "composition_tta_memo": multi_tta_module.adapt_with_tta_memo(deepcopy(state_dict), corruption_dataset),
            }

            full_model = gala_module.adapt_full_layers(
                deepcopy(state_dict),
                corruption_dataset,
                learning_rate=1e-3,
                batch_size=1,
            )
            gala_model, gala_stats = gala_module.adapt_gala_layers(
                deepcopy(state_dict),
                corruption_dataset,
                learning_rate=1e-3,
                batch_size=1,
                reset_window=20,
                threshold=0.75,
            )
            method_models["composition_full_adapt"] = full_model
            method_models["composition_gala"] = gala_model

            row = {
                "composition_id": composition_id,
                "composition_clients": ",".join(clients),
                "composition_length": composition_length,
                "corruption": corruption,
                "clean_samples": clean_samples,
                "corruption_samples": corruption_samples,
                "before_clean_accuracy": before_clean_acc,
                "before_corruption_accuracy": before_corruption_acc,
                "before_mixed_accuracy": before_mixed_acc,
            }

            for method_name, model in method_models.items():
                clean_acc = multi_tta_module.evaluate_model(model, clean_dataset)
                corruption_acc = multi_tta_module.evaluate_model(model, corruption_dataset)
                mixed_acc = multi_tta_module.evaluate_model(model, mixed_dataset)
                row[f"{method_name}_clean_accuracy"] = clean_acc
                row[f"{method_name}_corruption_accuracy"] = corruption_acc
                row[f"{method_name}_mixed_accuracy"] = mixed_acc
                row[f"{method_name}_gain_mixed"] = mixed_acc - before_mixed_acc
                row[f"{method_name}_gain_corruption"] = corruption_acc - before_corruption_acc

            for k, v in gala_stats.items():
                row[f"gala_{k}"] = v

            rows.append(row)
    return rows


def main():
    rows = evaluate_method_rows()
    with (OUTPUT_ROOT / "filtered_same_scale.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    with (OUTPUT_ROOT / "filtered_same_scale.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
