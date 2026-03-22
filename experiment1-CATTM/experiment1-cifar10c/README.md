# Composition-Aware FL TTA on CIFAR-10

This repository contains a cleaned version of the CIFAR-10 federated learning and test-time adaptation trigger experiments.

The repository is organized around:

- `composition_aware_fl_tta_CIFAR.py`
  - main experiment entrypoint for the proposed composition-aware trigger method
- `baselines/`
  - standalone baseline implementations used for comparison
  - `run_poem_trigger.py`
  - `run_asr_trigger.py`
  - `run_dss_trigger.py`
- `FL/train_fedavg_cifar10.py`
  - reference FedAvg training script used during development
- `Dataset/`
  - CIFAR-10 and corruption data archives
- `outputs_step1_step2_cifar_30k_5r_2e_fixed/`
  - final saved experiment artifacts

## Final Artifacts

The final composition model used by the experiments is:

- `outputs_step1_step2_cifar_30k_5r_2e_fixed/checkpoints/MLS_composition_cipher.pt`

The final compact comparison table currently kept in the repository is:

- `outputs_step1_step2_cifar_30k_5r_2e_fixed/metrics/final_selected_corruption_accuracy_table.csv`

This CSV contains the selected corruption-category accuracy comparison across:

- CATTM / proposed method
- ARS
- DSS
- POEM

## Main Method

To run the proposed CIFAR-10 composition-aware method, use:

```bash
python composition_aware_fl_tta_CIFAR.py \
  --output-dir outputs_experiment \
  --data-root Dataset \
  --mnist-c-root Dataset/CIFAR-10-C/CIFAR-10-C
```

Useful options:

- `--rounds`
- `--local-epochs`
- `--window-size`
- `--train-sample-cap`
- `--skip-step2` ... `--skip-step10`

Example: run only the later evaluation stages with an existing trained model/output directory:

```bash
python composition_aware_fl_tta_CIFAR.py \
  --output-dir outputs_experiment \
  --data-root Dataset \
  --mnist-c-root Dataset/CIFAR-10-C/CIFAR-10-C \
  --skip-step2
```

## Baselines

The baseline methods are kept as standalone scripts in `baselines/`.

### POEM

```bash
python baselines/run_poem_trigger.py
```

### ARS

```bash
python baselines/run_asr_trigger.py
```

### DSS

```bash
python baselines/run_dss_trigger.py
```

These baseline scripts were used to generate per-corruption comparison tables and threshold sweeps.

## Reproducibility Notes

- The experiments were run with fixed seeds and single-threaded numerical settings to avoid OpenMP / runtime instability.
- The main final results in this cleaned repository are preserved as saved artifacts instead of requiring every long simulation to be rerun.
- If you rerun experiments, new artifacts should be written to a new output directory rather than overwriting the preserved final folder.

## Recommended Reading Order

1. Open `composition_aware_fl_tta_CIFAR.py` for the proposed method.
2. Check `baselines/` for the comparison methods.
3. Inspect `outputs_step1_step2_cifar_30k_5r_2e_fixed/metrics/final_selected_corruption_accuracy_table.csv` for the current compact result table.
