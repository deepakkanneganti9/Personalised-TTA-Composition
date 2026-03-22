# FL Corrupt

This folder contains the federated-learning runner for MNIST-C corruptions.
It also supports mixed dataset specs such as `clean+fog` or `clean+zigzag`.

## Default training setup

- `10` clients per corruption
- `5` FedAvg rounds
- `2` local epochs
- artifact root: `FL_C/artifacts/federated_artifact_corrupt/`

## Default corruptions

- `zigzag`
- `stripe`
- `impulse_noise`
- `glass_blur`
- `brightness`
- `translate`
- `spatter`
- `fog`

## Outputs per corruption

Each corruption gets its own folder containing:

- `config.json`
- `client_preferences.json`
- `client_indices.json`
- `split_summary.json`
- `dataset_summary.json`
- `metrics.json`
- `rounds/round_*.json`
- `global/global_model_round_*.pt`
- `global/global_model.pt`
- `clients/client_<id>_round_<round>.pt`
- `clients/client_<id>.pt`
- `summary.json`

The top-level artifact folder also stores `index.json` across all corruption runs.

## Run

```bash
python3 FL_C/train_fedavg_mnist.py
```

If you need a single corruption only:

```bash
python3 FL_C/train_fedavg_mnist.py --mode train-corruption --corruption zigzag
```

If you want mixed clean+corrupt services with a shorter run:

```bash
python3 FL_C/train_fedavg_mnist.py \
  --mode train-all \
  --corruptions clean+fog,clean+translate,clean+stripe,clean+zigzag,clean+spatter \
  --num-clients 10 \
  --num-rounds 2 \
  --local-epochs 1 \
  --max-train-samples 30000 \
  --max-test-samples 5000
```

Note: this repository includes `glass_blur` in MNIST-C rather than `gaussian_blur`.
