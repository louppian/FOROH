"""Experiment 01: coordinate necessity, using the original 3_train.py pipeline."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "Experiment" / "train.py"

# Keep the original 3_train.py defaults. Only the Phase-1 variant and output
# directory change between controls.
RUNS = [
    ("E01A", "euclidean_huber"),
    ("E01B", "normalized_cosine"),
    ("E01C", "foroh"),
]


if __name__ == "__main__":
    for exp_id, variant in RUNS:
        out = ROOT / "Result" / "01_Coordinate_Necessity" / exp_id
        cmd = [
            sys.executable,
            str(TRAIN),
            "--method", "FOROH",
            "--phase1-variant", variant,
            "--dataset", "limuc",
            "--backbone", "resnet50",
            "--fold", "0",
            "--seed", "42",
            "--exp", "1",
            "--output-dir", str(out),
        ]
        print("\n$", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
