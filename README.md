# Test-Time Adaptation in MLaaS Composition for IoT Environments

This repository is organized around the four main ideas presented in the paper:

1. **Algorithm 1: CATTM**
2. **Algorithm 2: TCM**
3. **Algorithm 3: STAM / SATM package**
4. **Algorithm 4: CTAM / CATM package**

The codebase is split into three experiment groups:

- `experiment1-*`
  - Composition-aware triggering experiments for MNIST-C, CIFAR-10-C, and CIFAR-100-C
- `experiment2-tcm/`
  - TTA-aware composability experiments and paper-facing release packages
- `experiment3-satm-catam/`
  - Service-level and composition-level adaptation artifacts for the later-stage paper pipeline

## Repository Reading Guide

If you are new to the repository, the easiest reading order is:

1. Start with **Algorithm 1 / CATTM** in the `experiment1-*` folders
2. Move to **Algorithm 2 / TCM** in `experiment2-tcm/`
3. Then read **Algorithm 3 / STAM** and **Algorithm 4 / CTAM** in `experiment3-satm-catam/`

---

# Algorithm 1: CATTM

**CATTM** is the **Composition-Aware Test-Time Adaptation Trigger Mechanism** described in the paper. In this repository, Algorithm 1 is implemented through the trigger-focused experiments in **Experiment 1**.

These experiments cover:

- **MNIST / MNIST-C**
  - `experiment1-mnistc/`
- **CIFAR-10 / CIFAR-10-C**
  - `experiment1-cifar10c/`
- **CIFAR-100 / CIFAR-100-C**
  - `experiment1-cifar100c/`

## Main Algorithm 1 Files

- `experiment1-mnistc/composition_aware_fl_tta_mnist.py`
  - main CATTM pipeline for the MNIST-C experiment
- `experiment1-cifar10c/composition_aware_fl_tta_CIFAR.py`
  - main CATTM pipeline for the CIFAR-10-C experiment
- `experiment1-cifar100c/composition_aware_fl_tta_CIFAR.py`
  - main CATTM pipeline for the CIFAR-100-C experiment

## Baseline Trigger Files

The trigger baselines used for comparison are also kept in Experiment 1:

- `run_poem_trigger.py`
- `run_asr_trigger.py`
- `run_dss_trigger.py`

You can find them here:

- `experiment1-cifar10c/baselines/`
- `experiment1-cifar100c/baseline/`
- `experiment1-mnistc/baseline/`

## Experiment 1 Outputs and Final Tables

- **MNIST-C**
  - outputs: `experiment1-mnistc/outputs_experiment_w64/`
  - final comparison table: `experiment1-mnistc/outputs_experiment_w64/metrics/final_method_accuracy_table.csv`
- **CIFAR-10-C**
  - outputs: `experiment1-cifar10c/outputs_step1_step2_cifar_30k_5r_2e_fixed/`
  - final comparison table: `experiment1-cifar10c/outputs_step1_step2_cifar_30k_5r_2e_fixed/metrics/final_selected_corruption_accuracy_table.csv`
- **CIFAR-100-C**
  - outputs: `experiment1-cifar100c/outputs_experiment_round30/`
  - final CATTM table: `experiment1-cifar100c/outputs_experiment_round30/metrics/corruption_threshold_accuracy_report_accuracy_tuned.csv`

## Composition Models Used in Algorithm 1

The Experiment 1 READMEs identify the composition-model checkpoints used in the final runs:

- MNIST-C
  - `experiment1-mnistc/outputs_experiment_w64/checkpoints/mls_composition.pt`
- CIFAR-10-C
  - `experiment1-cifar10c/outputs_step1_step2_cifar_30k_5r_2e_fixed/checkpoints/MLS_composition_cipher.pt`
- CIFAR-100-C
  - `experiment1-cifar100c/outputs_experiment_round30/checkpoints/global_model_final.pt`

Note:

- these checkpoint paths are the documented composition models used by the experiments
- in the current Git snapshot, `.pt` files are intentionally not tracked, so some checkpoint files may be referenced in the documentation without being stored in the repository itself

## Experiment 1 READMEs

- `experiment1-cifar10c/README.md`
- `experiment1-cifar100c/README.txt`
- `experiment1-mnistc/README.txt`

---

# Personalized Adaptation

This section corresponds to the paper's **TTA-aware composability model**.

## Algorithm 2: TCM

**TCM** is the **TTA-aware MLaaS Composability Model** from the paper. In this repository, Algorithm 2 is represented by **Experiment 2**, where the main goal is to evaluate whether TTA-adapted services remain composable with the current performing composition.

The Experiment 2 datasets are:

- `experiment2-tcm/experiment2-mnistc-tcm/`
- `experiment2-tcm/experiment2-cifar10c-tcm/`
- `experiment2-tcm/experiment2-cifar100c-tcm/`

## Main Algorithm 2 Files

There is not a single file literally named `algorithm2`, so the closest implementation entry points for Algorithm 2 are the compatibility and final-table pipeline scripts inside each Experiment 2 release package.

Key files:

- `scripts/run_pairwise_compatibility.py`
  - main compatibility evaluation pipeline for TCM-style analysis
- `scripts/compatibility_metrics.py`
  - compatibility metric computation used in the MNIST-C and CIFAR-10-C releases
- `scripts/build_final_tables.py`
  - lightweight paper-facing script that rebuilds the public final tables from saved compatibility summaries and QoS files

Additional adaptation-related files in Experiment 2 include:

- MNIST-C
  - `scripts/tent_grad_adapter.py`
  - `scripts/tta_bn_adapter.py`
  - `scripts/tta_memo_adapter.py`
- CIFAR-10-C and CIFAR-100-C
  - `scripts/run_tent_adaptation.py`
  - `scripts/run_tta_bn_adaptation.py`
  - `scripts/run_tta_memo_adaptation.py`

## Experiment 2 Final Outputs

- MNIST-C
  - `experiment2-tcm/experiment2-mnistc-tcm/results/final/combined_final_tables.csv`
- CIFAR-10-C
  - `experiment2-tcm/experiment2-cifar10c-tcm/results/final/combined_final_table.csv`
- CIFAR-100-C
  - `experiment2-tcm/experiment2-cifar100c-tcm/results/final/combined_final_table.csv`

## Experiment 2 READMEs

- `experiment2-tcm/experiment2-mnistc-tcm/README.md`
- `experiment2-tcm/experiment2-cifar10c-tcm/README.md`
- `experiment2-tcm/experiment2-cifar100c-tcm/README.md`

---

# Algorithm 3 and Algorithm 4

The final part of the repository contains the service-level and composition-level adaptation artifacts used in the later paper pipeline.

## Algorithm 3: STAM

In the paper, **Algorithm 3** is the **Service-Level TTA Composition Model (STAM)**.

In this repository, the service-level package is stored under:

- `experiment3-satm-catam/SATM/`

The naming in the repository uses `SATM`, while the paper text refers to the service-level model as `STAM`. The key implementation file for Algorithm 3 is:

- `experiment3-satm-catam/SATM/source/algorithm3_satm.py`

Supporting files for Algorithm 3:

- `experiment3-satm-catam/SATM/source/satm_substitution_analysis.py`
- `experiment3-satm-catam/SATM/source/satm_reference_analysis.py`
- `experiment3-satm-catam/SATM/run_satm.py`

Important SATM/STAM result files:

- `experiment3-satm-catam/SATM/results/satm_multi_tta_results_50_50.csv`
- `experiment3-satm-catam/SATM/results/satm_substitution_results_50_50.csv`
- `experiment3-satm-catam/SATM/results/satm_preference_weights_50_50.csv`

## Algorithm 4: CTAM

In the paper, **Algorithm 4** is the **Composition-Level TTA Composition Model (CTAM)**.

In this repository, the composition-level package is stored under:

- `experiment3-satm-catam/CATM/`

The corresponding implementation file for Algorithm 4 is:

- `experiment3-satm-catam/CATM/source/algorithm4_catm.py`

Supporting files for Algorithm 4:

- `experiment3-satm-catam/CATM/source/catm_composition_tta_methods.py`
- `experiment3-satm-catam/CATM/source/catm_filtered_evaluation.py`
- `experiment3-satm-catam/CATM/run_catm.py`

Important CATM/CTAM result files:

- `experiment3-satm-catam/CATM/results/catm_filtered_results_50_50.csv`
- `experiment3-satm-catam/CATM/results/catm_packet_summary.json`

## Shared Experiment 3 Assets and Final Tables

Shared files for Experiment 3:

- `experiment3-satm-catam/assets/`
  - packaged FL, CATM, and substitution assets
- `experiment3-satm-catam/common/`
  - shared FL and TTA utility code

Final summary tables:

- `experiment3-satm-catam/results/final_table_50_50.csv`
- `experiment3-satm-catam/results/final_table_by_corruption_50_50.csv`
- `experiment3-satm-catam/results/final_table_by_corruption_and_length_50_50.csv`
- `experiment3-satm-catam/results/final_table_tta_bn_by_corruption_and_length_50_50.csv`

Utility scripts:

- `experiment3-satm-catam/run_packet_results.py`
- `experiment3-satm-catam/run_paper_pipeline.py`
- `experiment3-satm-catam/generate_final_table_50_50.py`
- `experiment3-satm-catam/generate_final_table_by_corruption_50_50.py`
- `experiment3-satm-catam/generate_final_table_by_corruption_and_length_50_50.py`

## Experiment 3 READMEs

- `experiment3-satm-catam/README.md`
- `experiment3-satm-catam/SATM/README.md`
- `experiment3-satm-catam/CATM/README.md`

---

## Quick File Map

- **Algorithm 1 / CATTM**
  - `experiment1-mnistc/composition_aware_fl_tta_mnist.py`
  - `experiment1-cifar10c/composition_aware_fl_tta_CIFAR.py`
  - `experiment1-cifar100c/composition_aware_fl_tta_CIFAR.py`
- **Algorithm 2 / TCM**
  - `experiment2-tcm/*/scripts/run_pairwise_compatibility.py`
  - `experiment2-tcm/*/scripts/build_final_tables.py`
- **Algorithm 3 / STAM (repository package: SATM)**
  - `experiment3-satm-catam/SATM/source/algorithm3_satm.py`
- **Algorithm 4 / CTAM (repository package: CATM)**
  - `experiment3-satm-catam/CATM/source/algorithm4_catm.py`

This outer README is intended to help readers move from the paper's algorithm numbering to the exact experiment folders and source files in the repository.
