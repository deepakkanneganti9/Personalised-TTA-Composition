# Paper Artifact Structure

This folder contains the paper artifact organized around the two main techniques:

- `SATM`
- `CATM`

The artifact is designed around precomputed experiment assets and precomputed result files. The main purpose of this package is:

1. to expose the actual source files that implement the proposed methods
2. to keep the exact experiment assets used for the final tables
3. to regenerate the final summary tables from the packaged result files

## Folder Overview

### `SATM/`

Contains the source files and packaged result files related to the SATM method.

Main source files:
- `SATM/source/algorithm4_satm.py`
  - main SATM logic used for the multi-TTA `50/50` experiment
  - this is the primary implementation corresponding to **Algorithm 4** in the paper
- `SATM/source/satm_substitution_analysis.py`
  - substitution-side comparison used in the final filtered analysis
- `SATM/source/satm_reference_analysis.py`
  - reference evaluation logic used while assembling the final SATM comparison

Packaged SATM result files:
- `SATM/results/satm_multi_tta_results_50_50.csv`
- `SATM/results/satm_substitution_results_50_50.csv`
- `SATM/results/satm_preference_weights_50_50.csv`

### `CATM/`

Contains the source files and packaged result files related to the CATM method.

Main source files:
- `CATM/source/algorithm3_catm.py`
  - main incremental selective composition adaptation logic
  - this is the primary implementation corresponding to **Algorithm 3** in the paper
- `CATM/source/catm_composition_tta_methods.py`
  - composition-level TTA baselines used alongside CATM
- `CATM/source/catm_filtered_evaluation.py`
  - filtered evaluation over the selected composition/corruption contexts

Packaged CATM result files:
- `CATM/results/catm_filtered_results_50_50.csv`

### `assets/`

Contains the copied experiment assets required to understand the setup and inspect the precomputed experiment state:

- `assets/FL/`
  - federated baseline checkpoints
- `assets/data/`
  - clean MNIST and the corruption subsets used in the final paper analysis
- `assets/TTA/`
  - precomputed client-level TTA artifacts for `tent`, `tta_bn`, and `tta_memo`
- `assets/CATM/`
  - composition baselines and composition-level CATM/TTA artifacts
- `assets/substitution/`
  - substitution-model artifacts used in the substitution comparisons

### `common/`

Shared source code copied into the artifact for transparency:

- `common/FL/train_fedavg_mnist.py`
- `common/TTA/tta_techniques/*`

### `results/`

Contains the final summary tables generated from the packaged results:

- `results/final_table_50_50.csv`
- `results/final_table_by_corruption_50_50.csv`
- `results/final_table_by_corruption_and_length_50_50.csv`
- `results/final_table_tta_bn_by_corruption_and_length_50_50.csv`

## Table Generation

To regenerate the summary tables from the packaged result files:

```bash
python paper_release/run_packet_results.py
```

This command does not retrain models. It rebuilds the summary tables from the packaged SATM and CATM result files already stored in this artifact.
