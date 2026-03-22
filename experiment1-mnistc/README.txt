CATTM on MNIST / MNIST-C

This repository contains the final accepted experiment setup for the composition-aware test-time adaptation trigger (CATTM) and three baseline trigger implementations: POEM, ARS, and DSS.

Repository contents
- composition_aware_fl_tta_mnist.py: main CATTM pipeline
- baseline/: baseline method scripts
- Datasets/: local MNIST and MNIST-C data
- outputs_experiment_w64/checkpoints/mls_composition.pt: final saved global model
- outputs_experiment_w64/historical_bank/: historical clean and mild-corruption contexts
- outputs_experiment_w64/metrics/final_method_accuracy_table.csv: final comparison table

Final accepted setup
- FL rounds: 5
- local epochs: 3
- window size: 64
- Step 4 contexts: train split, first 10,000 samples
- Step 5 scan: train split, 20,000 samples per corruption
- final model name: mls_composition.pt

How to regenerate CATTM results
Run the main script and reuse the saved checkpoint and historical bank:

KMP_DUPLICATE_LIB_OK=TRUE python3 composition_aware_fl_tta_mnist.py --output-dir outputs_experiment_w64 --window-size 64 --skip-step2 --skip-step3 --skip-step4 --pool-split train --max-pool-samples 20000

How to regenerate the final table
Run:

python3 export_final_method_table.py

This regenerates outputs_experiment_w64/metrics/final_method_accuracy_table.csv.
