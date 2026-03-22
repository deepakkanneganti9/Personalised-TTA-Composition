# CIFAR-10 / CIFAR-10-C Paper Release

This release directory is a paper-facing snapshot built from already saved CIFAR-10 / CIFAR-10-C artifacts and results. It does not modify the original repository and it does not rerun the expensive FL, TTA, or pairwise compatibility pipelines.

## Layout

- `artifacts/`
  - Canonical saved checkpoints and artifact folders.
  - Includes the 20-round clean FL run, the reported TTA artifact folders, and the expanded clean performing-composition artifacts.
- `results/compatibility/`
  - Canonical chunk-level compatibility summaries for each reported TTA method.
- `results/qos/`
  - Unified QoS CSV files derived from the existing QoS exports, with `weights_file` values rewritten to point at the canonical artifact paths inside this release.
- `results/final/`
  - Public paper-facing final tables with only outcome columns.
- `scripts/`
  - Core pipeline scripts plus `build_final_tables.py`, which rebuilds the public final tables from saved summaries and saved QoS files only.

## Dependency Flow

Saved artifacts -> saved compatibility summaries -> saved QoS CSVs -> lightweight final-table regeneration

This means readers can regenerate the final tables without retraining FL models, rerunning TTA adaptation, or recomputing chunk-level pairwise compatibility.

## Canonical Inputs

- Clean FL checkpoints:
  - `artifacts/fl/cifar10_fl_baseline_5clients_run_20rounds/`
- TTA artifact folders:
  - `artifacts/tta/tent/`
  - `artifacts/tta/tta_bn/`
  - `artifacts/tta/tta_memo/`
- Performing compositions:
  - `artifacts/compositions/expanded/`
- Compatibility summaries:
  - `results/compatibility/tent/summary.csv`
  - `results/compatibility/tta_bn/summary.csv`
  - `results/compatibility/tta_memo/summary.csv`

## Regenerating Final Tables

From the `paper_release/` directory:

```bash
python3 scripts/build_final_tables.py
```

This writes:

- `results/final/tent_final_table.csv`
- `results/final/tta_bn_final_table.csv`
- `results/final/tta_memo_final_table.csv`
- `results/final/combined_final_table.csv`

The public tables intentionally keep only:

- `method`
- `technique`
- `accuracy`
- `precision`
- `recall`
- `f1`

Threshold analysis details remain internal to the code and the manifest JSON rather than being exposed in the public-facing CSVs.
