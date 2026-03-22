from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    runpy.run_path(str(ROOT / "generate_final_table_50_50.py"), run_name="__main__")
    runpy.run_path(str(ROOT / "generate_final_table_by_corruption_50_50.py"), run_name="__main__")
    runpy.run_path(str(ROOT / "generate_final_table_by_corruption_and_length_50_50.py"), run_name="__main__")
