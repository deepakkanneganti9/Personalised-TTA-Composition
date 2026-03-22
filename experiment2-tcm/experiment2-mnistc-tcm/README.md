# Paper Release Layout

This release layout is a cleaned, paper-facing view of the project. It is built from the existing saved artifacts and results only. No new FL training, TTA adaptation, or compatibility simulations are required to inspect the reported outcomes.

## Structure

- `artifacts/`
  - canonical saved checkpoints and artifacts
  - includes:
    - clean FL checkpoints
    - adapted TTA artifacts for `tta_grad`, `tta_bn`, and `tta_memo`
    - performing-composition artifacts

- `results/compatibility/`
  - canonical chunk-level compatibility summaries
  - one folder per TTA method
  - each method contains:
    - `summary.csv`
    - `summary_with_baselines.csv`

- `results/qos/`
  - unified QoS datasets
  - `adapted_qos_all.csv` combines the adapted-service QoS records for all three TTA methods
  - `performing_compositions_qos.csv` stores the QoS records for clean performing compositions
  - these files reference the model checkpoints already stored under `artifacts/` instead of duplicating copied weight files

- `results/final/`
  - clean final result tables without internal threshold columns
  - includes one final table for each method and one combined table

- `scripts/`
  - canonical scripts used in the pipeline
  - kept for traceability and reproducibility from the saved artifacts

## Reproduction Flow

The simplified flow is:

1. Use the saved FL checkpoints and TTA artifacts from `artifacts/`
2. Use the saved performing-composition artifacts from `artifacts/performing_compositions/`
3. Use the saved chunk-level compatibility summaries in `results/compatibility/`
4. Use the saved QoS CSVs in `results/qos/`
5. Inspect the final paper-facing results in `results/final/`

This release is intentionally organized around saved artifacts and saved outputs so that readers do not need to rerun expensive simulations just to inspect the reported results.

## Canonical Final Outputs

- `results/final/grad/final_table.csv`
- `results/final/tta_bn/final_table.csv`
- `results/final/tta_memo/final_table.csv`
- `results/final/combined_final_tables.csv`

These are the public-facing final tables. Internal threshold exploration files are intentionally excluded from this release layout.
