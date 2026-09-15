import subprocess
import sys
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

ML_PREDICTIONS = ROOT / "predictions_ml.csv"
FINAL_FILE = ROOT / "predictions.csv"


def run_command(command):
    print("\nRunning:", " ".join(command))
    result = subprocess.run(command, cwd=ROOT)

    if result.returncode != 0:
        raise SystemExit(f"\nCommand failed with exit code {result.returncode}")


def main():
    if not DATA_DIR.exists():
        raise SystemExit("ERROR: data folder was not found.")

    print("LPDG Innovation Hub 2026 - Machine Learning Pipeline")
    print("=" * 55)

    # Step 1: Build historical ML features and labels
    run_command([
        sys.executable,
        "analysis/create_ml_baseline_features.py"
    ])

    # Step 2: Train the final ML model and generate predictions
    run_command([
        sys.executable,
        "analysis/final_ml_predictions.py"
    ])

    # Step 3: Copy ML predictions to the required filename
    if not ML_PREDICTIONS.exists():
        raise SystemExit("ERROR: predictions_ml.csv was not generated.")

    shutil.copy2(ML_PREDICTIONS, FINAL_FILE)

    # Step 4: Validate the final submission
    run_command([
        sys.executable,
        "validate_submission.py",
        "predictions.csv"
    ])

    print("\nPipeline completed successfully.")
    print("Final submission: predictions.csv")


if __name__ == "__main__":
    main()