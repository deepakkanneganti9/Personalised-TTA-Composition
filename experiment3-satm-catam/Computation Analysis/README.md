# Paper Submission Bundle

This folder is a cleaned reproducibility bundle for the final MNIST benchmark used in the paper discussion.

## Scope of this bundle

This package contains the exact code, weights, profiles, outputs, and final tables needed to reproduce the reported **translate-only final lightweight benchmark** with:

- Semantic substitution
- Context substitution
- MLaaS substitution
- TTA with reviewer-style aggregation logic

The final TTA aggregation logic follows:
- `scripts/evaluate_reviewer_composition_table.py`

## What is included

### 1. Shared assets
- `shared/client_profiles/`
  - client profile CSV
  - weight metadata
  - profiled client weights
- `shared/profile/`
  - composition QoS profile CSV
  - composition weight metadata
  - composition weights

### 2. Training provenance
- `training/FL_C/`
  - training and profile-generation scripts
- `training/artifacts/clean_translate/`
  - the last FL_C training run used by this experiment

### 3. Benchmark scripts
- `scripts/run_traditional_adaptive_composition.py`
- `scripts/evaluate_fair_mixed_protocol.py`
- `scripts/collect_runtime_summary.py`
- `scripts/plot_computation_time_scaling.py`
- `scripts/evaluate_reviewer_composition_table.py`
- `scripts/run_pairwise_compatibility.py`
- `scripts/compatibility_metrics.py`

### 4. Per-method folders
- `methods/semantic/`
- `methods/context/`
- `methods/mlaas/`
- `methods/tta/`

Each method folder contains:
- the saved method outputs
- a small runner wrapper (`run_<method>.py`)

### 5. Final result tables
The reviewer-facing tables are in:
- `generated_tables/`

Important files:
- `generated_tables/final_submission_results.csv`
- `generated_tables/final_accuracy_table.csv`
- `generated_tables/timing_similarity_table.csv`
- `generated_tables/by_corruption_table.csv`
- `generated_tables/overall_computation_time_table.csv`
- `generated_tables/overall_time_per_case_table.csv`
- `generated_tables/composition_size_time_table.csv`

### 6. Supporting outputs
- `outputs/fair_mixed/`
- `outputs/plots/`
- `outputs/timing/`

## Final experiment configuration

This bundle reflects the final lightweight setup:
- corruption refreshed: `translate`
- evaluation mix: `20% clean / 80% translate`
- evaluation sample count: `100`
- single-service failure only
- no multi-substitution search
- TTA uses the reviewer-style aggregation logic

## How to rerun

From this folder:

### Run one method
- Semantic:
  - `python3 methods/semantic/run_semantic.py`
- Context:
  - `python3 methods/context/run_context.py`
- MLaaS:
  - `python3 methods/mlaas/run_mlaas.py`
- TTA:
  - `python3 methods/tta/run_tta.py`

### Regenerate the mixed evaluation summary
- `python3 scripts/evaluate_fair_mixed_protocol.py --corruptions translate --clean-ratios 0.2 --total-samples 100 --batch-size 64`

### Regenerate the computation-time scaling plot
- `python3 scripts/plot_computation_time_scaling.py`

### Regenerate all final reviewer tables from the included saved outputs
- `python3 scripts/build_submission_tables.py`

## What the final tables mean

- `final_submission_results.csv`
  - one single consolidated CSV containing all reported final tables in long-form format
- `final_accuracy_table.csv`
  - healthy, degraded, after-adaptation accuracy and recovery metrics
- `timing_similarity_table.csv`
  - online benchmark timing and similarity statistics
- `overall_computation_time_table.csv`
  - full substitution-side preparation cost plus method runtime, compared to lightweight TTA runtime
- `composition_size_time_table.csv`
  - timing summarized by composition-size coverage

## Notes for reviewers

1. This bundle is intentionally cleaned to include only the files needed for the final reported benchmark.
2. The final reported benchmark is **translate-only** for the refreshed lightweight run.
3. The TTA result in this bundle is the corrected result using reviewer-style aggregation logic.
4. This bundle is independently reproducible for table regeneration from the included saved outputs via `scripts/build_submission_tables.py`.
5. It is not a full raw-data retraining package; it is a cleaned benchmark-results package for reproducing the reported reviewer tables.
