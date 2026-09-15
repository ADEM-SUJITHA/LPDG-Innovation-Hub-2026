import subprocess
import sys
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

BASELINE_FILE = ROOT / "predictions_baseline.csv"
FINAL_FILE = ROOT / "predictions.csv"


def run_command(command):
    print("\nRunning:", " ".join(command))
    result = subprocess.run(command, cwd=ROOT)

    if result.returncode != 0:
        raise SystemExit(f"\nCommand failed with exit code {result.returncode}")


def main():
    if not DATA_DIR.exists():
        raise SystemExit("ERROR: data folder was not found.")

    print("LPDG Innovation Hub 2026 - Prediction Pipeline")
    print("=" * 50)

    # Step 1: Generate the provided 3-sigma baseline
    run_command([
        sys.executable,
        "baseline_3sigma.py",
        "--data",
        "data",
        "--out",
        "predictions_baseline.csv"
    ])

    # The improvement analysis reads predictions.csv,
    # so temporarily use the baseline as its input.
    shutil.copy2(BASELINE_FILE, FINAL_FILE)

    # Step 2: Apply the improved evidence-based ranking
    run_command([
        sys.executable,
        "analysis/improved_analysis.py"
    ])

    # Step 3: Validate final submission
    run_command([
        sys.executable,
        "validate_submission.py",
        "predictions.csv"
    ])

    print("\nPipeline completed successfully.")
    print("Final submission: predictions.csv")


if __name__ == "__main__":
    main()