# SATM

This folder contains the SATM source files and the packaged SATM result files.

## Main Logic

- `source/algorithm3_satm.py`
  - primary SATM implementation
  - corresponds to **Algorithm 3** in the paper
- `source/satm_substitution_analysis.py`
  - substitution analysis used in the final filtered comparison
- `source/satm_reference_analysis.py`
  - supporting SATM evaluation logic used during the final comparison setup

## Result Files

- `results/satm_multi_tta_results_50_50.csv`
  - main `50/50` SATM experiment results across `tent`, `tta_bn`, and `tta_memo`
- `results/satm_substitution_results_50_50.csv`
  - filtered substitution comparison used in the final summary tables
- `results/satm_preference_weights_50_50.csv`
  - preference weights used in the filtered SATM analysis

## Utility Script

- `run_satm.py`
  - validates the packaged SATM result files and reports the packet summary
