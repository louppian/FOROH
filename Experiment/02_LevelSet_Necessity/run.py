"""Experiment 02: level-set versus point prototype on original 3_train.py pipeline."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "Experiment" / "train.py"

RUNS = [
    ("E02A", "point_prototype"),
    ("E02B", "foroh"),
]


if __name__ == "__main__":
    for exp_id, variant in RUNS:
        out = ROOT / "Result" / "02_LevelSet_Necessity" / exp_id
        cmd = [
            sys.executable,
            str(TRAIN),
            "--method", "FOROH",
            "--phase1-variant", variant,
            "--dataset", "limuc",
            "--backbone", "resnet50",
            "--fold", "0",
            "--seed", "42",
            "--exp", "2",
            "--output-dir", str(out),
        ]
        print("\n$", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
