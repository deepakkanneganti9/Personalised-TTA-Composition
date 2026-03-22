# CIFAR-100 / CIFAR-100-C Paper Release

This directory is a paper-facing release package built from the already saved CIFAR-100 artifacts and results in the original repository. It does not delete or modify the original project, and it does not rerun the expensive FL, TTA, or pairwise compatibility stages.

## Layout

- `artifacts/`
  - canonical saved checkpoints and model artifacts
  - `fl/` contains the clean federated learning run
  - `tta/` contains saved TTA artifacts for `TTA-Grad`, `TTA-BN`, and `TTA-MEMO`
  - `composition/` contains the saved performing-composition artifacts
- `results/`
  - `compatibility/` contains the saved chunk-level compatibility summaries for each TTA method
  - `qos/` contains normalized QoS CSV files pointing to the canonical artifact paths in this release
  - `final/` contains the public-facing final tables
- `scripts/`
  - canonical scripts copied from the original repository
  - `build_final_tables.py` is the lightweight release script for regenerating the final paper tables
- `manifest.json`
  - compact dependency manifest for the release package

## Dependency Flow

The simplified dependency flow is:

1. saved artifacts
2. saved compatibility summaries
3. saved QoS CSV files
4. lightweight final-table regeneration

No heavy stage is rerun in this release package. The final tables are rebuilt only from the already saved compatibility summaries and the saved QoS descriptors.

## Regenerating Final Tables

From this `paper_release/` directory, run:

```bash
python scripts/build_final_tables.py
```

This regenerates:

- `results/final/TTA-Grad.csv`
- `results/final/TTA-BN.csv`
- `results/final/TTA-MEMO.csv`
- `results/final/combined_final_table.csv`

The public-facing tables intentionally expose only:

- `method`
- `technique`
- `accuracy`
- `precision`
- `recall`
- `f1`

Threshold-selection logic remains internal to the code and is not exposed in the paper-facing tables.

## Notes

- The original repository remains untouched.
- QoS CSVs in this release were rewritten to point to the canonical copied checkpoints under `artifacts/`.
- Exploratory threshold-analysis files are preserved only in the internal regeneration outputs, not in the public-facing final tables.
