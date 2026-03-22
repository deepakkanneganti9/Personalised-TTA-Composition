# CATM

This folder contains the CATM source files and the packaged CATM result files.

## Main Logic

- `source/algorithm4_catm.py`
  - primary CATM implementation
  - corresponds to **Algorithm 4** in the paper
- `source/catm_composition_tta_methods.py`
  - composition-level TTA baselines used for comparison
- `source/catm_filtered_evaluation.py`
  - filtered evaluation over the selected composition/corruption contexts

## Result Files

- `results/catm_filtered_results_50_50.csv`
  - filtered CATM result file used in the final summary tables

## Utility Script

- `run_catm.py`
  - validates the packaged CATM result files and reports the packet summary
