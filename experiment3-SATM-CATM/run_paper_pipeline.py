import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_step(script: Path):
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


def copy_result(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_step(ROOT / "evaluate_reviewer_composition_with_substitution_v2.py")
    run_step(ROOT / "evaluate_reviewer_composition_multi_tta_v2.py")
    run_step(ROOT / "STMU" / "evaluate_filtered_stmu_same_scale.py")
    run_step(ROOT / "evaluate_filtered_improved_random_substitution.py")
    run_step(Path(__file__).resolve().parent / "generate_final_table_50_50.py")

    copy_result(
        ROOT / "experiment_artifacts" / "reviewer_composition_multi_tta_v2" / "reviewer_composition_multi_tta_v2.csv",
        RESULTS_DIR / "reviewer_composition_multi_tta_v2.csv",
    )
    copy_result(
        ROOT / "experiment_artifacts" / "reviewer_composition_multi_tta_v2" / "reviewer_composition_multi_tta_v2.json",
        RESULTS_DIR / "reviewer_composition_multi_tta_v2.json",
    )
    copy_result(
        ROOT / "STMU" / "artifacts" / "filtered_same_scale" / "filtered_same_scale.csv",
        RESULTS_DIR / "filtered_same_scale.csv",
    )
    copy_result(
        ROOT / "STMU" / "artifacts" / "filtered_same_scale" / "filtered_same_scale.json",
        RESULTS_DIR / "filtered_same_scale.json",
    )
    copy_result(
        ROOT / "experiment_artifacts" / "filtered_improved_random_substitution" / "filtered_improved_random_substitution.csv",
        RESULTS_DIR / "filtered_improved_random_substitution.csv",
    )
    copy_result(
        ROOT / "experiment_artifacts" / "filtered_improved_random_substitution" / "filtered_improved_random_substitution.json",
        RESULTS_DIR / "filtered_improved_random_substitution.json",
    )


if __name__ == "__main__":
    main()
