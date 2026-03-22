Experiment 1 CIFAR100

Final conference-ready structure

1. composition_aware_fl_tta_CIFAR.py
Main CATTM / composition-aware trigger implementation for CIFAR-100.

2. baseline/
Contains only the baseline method scripts:
- run_poem_trigger.py
- run_asr_trigger.py
- run_dss_trigger.py

3. Dataset/
Only the CIFAR-100 data kept for the final setup:
- CIFAR-100 clean
- CIFAR-100-C

4. outputs_experiment_round30/
Final experiment outputs used for the current CIFAR-100 results.
Important subfolders:
- checkpoints/
- historical_bank/
- pools/
- stream_definitions/
- window_results/
- metrics/
- plots/

Final model used

The final CATTM / paper results in this repository use:
- outputs_experiment_round30/checkpoints/global_model_final.pt

That checkpoint was prepared from:
- outputs_experiment_round30/checkpoints/global_model_round_30_source.pt

So the final experiment is the round-30 CIFAR-100 setup, not the earlier full_prepared or smoke outputs.

Final result tables

The main CATTM corruption-wise table used for the latest tuned results is:
- outputs_experiment_round30/metrics/corruption_threshold_accuracy_report_accuracy_tuned.csv

The selected baseline result tables copied for the final comparison are:
- outputs_experiment_round30/metrics/baselines/poem_final.csv
- outputs_experiment_round30/metrics/baselines/ars_final.csv
- outputs_experiment_round30/metrics/baselines/dss_final_tau098.csv
- outputs_experiment_round30/metrics/baselines/cattm_final_accuracy_tuned.csv

Notes

- Older duplicate folders, smoke runs, image tables, tools, and FL training folders were removed.
- The repository is now centered on the final CIFAR-100 round-30 experiment only.
