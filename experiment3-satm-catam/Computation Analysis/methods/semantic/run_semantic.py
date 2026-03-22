#!/usr/bin/env python3
import os, subprocess, sys
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
script=os.path.join(ROOT,'scripts','run_traditional_adaptive_composition.py')
cmd=['python3', script, '--method', 'semantic', '--corruptions', 'translate', '--max-eval-samples', '100', '--target-clean-ratio', '0.2', '--mixed-clean-ratio', '0.9', '--tta-max-target-samples', '100', '--probe-samples-per-corruption', '100']
raise SystemExit(subprocess.call(cmd))
