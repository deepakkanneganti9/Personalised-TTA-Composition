# FL Baseline

This folder contains the reusable federated learning baseline trained on clean CIFAR-10 only.

## What it does

- Loads the local `Data/` CIFAR-10 dataset
- Builds a mild non-IID split across 5 clients
- Trains a standard FedAvg model with BatchNorm layers
- Saves reusable artifacts for later TTA and compatibility experiments

## Outputs

Running the training script creates:

- `FL/artifacts/cifar10_fl_baseline_5clients_run/config.json`
- `FL/artifacts/cifar10_fl_baseline_5clients_run/client_preferences.json`
- `FL/artifacts/cifar10_fl_baseline_5clients_run/client_indices.json`
- `FL/artifacts/cifar10_fl_baseline_5clients_run/split_summary.json`
- `FL/artifacts/cifar10_fl_baseline_5clients_run/metrics.json`
- `FL/artifacts/cifar10_fl_baseline_5clients_run/rounds/round_<xx>.json`
- `FL/artifacts/cifar10_fl_baseline_5clients_run/global/global_model_round_<r>.pt`
- `FL/artifacts/cifar10_fl_baseline_5clients_run/global/global_model.pt`
- `FL/artifacts/cifar10_fl_baseline_5clients_run/clients/client_<id>_round_<r>.pt`
- `FL/artifacts/cifar10_fl_baseline_5clients_run/clients/client_<id>.pt`
- `FL/artifacts/cifar10_fl_baseline_5clients_run/summary.json`

## Run

```bash
python3 FL/train_fedavg_cifar10.py --mode full --device cpu --run-dir FL/artifacts/cifar10_fl_baseline_5clients_run
```
