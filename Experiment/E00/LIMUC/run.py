"""E00 LIMUC paper reproduction.

This wrapper does not reimplement training. It runs the original repository
3_train.py with the settings stated in the FOROH paper for LIMUC.
"""

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TRAIN = REPO_ROOT / "3_train.py"
OUTPUT = REPO_ROOT / "Result" / "E00" / "LIMUC"


def main():
    cmd = [
        sys.executable,
        str(TRAIN),
        "--method", "FOROH",
        "--dataset", "limuc",
        "--backbone", "resnet50",
        "--proj-dim", "128",
        "--n-folds", "10",
        "--fold", "-1",
        "--epochs", "50",
        "--batch-size", "64",
        "--lr", "1e-4",
        "--lr-head", "1e-3",
        "--weight-decay", "1e-4",
        "--img-size", "224",
        "--freeze-layers", "2",
        "--patience", "10",
        "--optimizer", "adamw",
        "--scheduler", "cosine",
        "--seed", "42",
        "--output-dir", str(OUTPUT),
    ]
    print("[E00/LIMUC] FOROH paper reproduction")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
