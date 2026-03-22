# Paper Release Manifest

This folder is a self-contained packet for reproducing the final paper tables from precomputed results only.

## Naming

- `STAM`: the final summed-preference aggregation method previously called `Protected V2`
- `CATM`: the composition-level selective adaptation method previously referred to through the STMU filtered same-scale evaluation

## Entry Points

- `STM/run_stam.py`
  - validates the packaged STAM result files
- `CATM/run_catm.py`
  - validates the packaged CATM result files
- `run_packet_results.py`
  - rebuilds the final `50/50` paper table from the packaged precomputed results only
- `run_paper_pipeline.py`
  - reruns the paper-relevant `50/50` pipeline and writes the final table into this folder
- `generate_final_table_50_50.py`
  - assembles the final summary table from the regenerated intermediate results

## Final Result Sources

- `results/reviewer_composition_multi_tta_v2.csv`
  - full `50/50` STAM table across `tent`, `tta_bn`, and `tta_memo`
- `results/filtered_same_scale.csv`
  - filtered `50/50` composition-level TTA and CATM results
- `results/filtered_improved_random_substitution.csv`
  - filtered `50/50` substitution comparison
- `results/final_table_50_50.csv`
  - final paper-facing `50/50` summary table
- `results/final_table_50_50.md`
  - the same table in Markdown form
- `results/final_table_by_corruption_50_50.csv`
  - final `50/50` summary table grouped by corruption
- `results/final_table_by_corruption_50_50.md`
  - the same corruption-wise table in Markdown form
- `results/final_table_by_corruption_and_length_50_50.csv`
  - final `50/50` summary table grouped by corruption and composition length
- `results/final_table_by_corruption_and_length_50_50.md`
  - the same corruption+length table in Markdown form
- `results/filtered_improved_full_table_30_70.csv`
  - filtered `30/70` final comparison table
- `results/stam_filtered_preference_weights_50_50.csv`
  - filtered `50/50` STAM preference weights used for the improved-case analysis

## Reproduction

From the repository root:

```bash
python paper_release/run_packet_results.py
```

This is the lightest reviewer path. It uses only the precomputed packet results already included under:

- `paper_release/STM/results`
- `paper_release/CATM/results`

If you want to refresh the upstream packaged results first, then run:

```bash
python paper_release/run_paper_pipeline.py
```

This regenerates the final `50/50` paper table and places it in:

- `paper_release/results/final_table_50_50.csv`
- `paper_release/results/final_table_50_50.md`

The pipeline uses precomputed FL checkpoints, precomputed client-level TTA artifacts, and precomputed substitution models that are already stored in the repository.

## Filtered Counts

- Total STAM cases before filtering: `900`
- Filtered improved STAM rows: `223`
- Unique filtered composition/corruption contexts: `57`
- Filtered rows by TTA method:
  - `tent`: `73`
  - `tta_bn`: `74`
  - `tta_memo`: `76`
- Filtered rows by performing length:
  - `1`: `52`
  - `2`: `112`
  - `3`: `38`
  - `4`: `21`

## Sample Sizes

- STAM filtered evaluation: `50 clean + 50 corrupt`
- CATM filtered same-scale evaluation: `50 clean + 50 corrupt`
- Final `30/70` table: `30 clean + 70 corrupt`

## Upstream Dependencies

- `../../FL/artifacts/mnist_fl_baseline_5clients_run`
- `../../TTA techniques/artifacts`
- `../../TTA techniques/artifacts_tta_bn`
- `../../TTA techniques/artifacts_tta_memo`
- `../../STMU/artifacts/composition_baselines`
- `../../data/MNIST`
- `../../data/mnist_c`
- `../../federated_artifact_corrupt`
