"""Run the complete canonical Phase-1 experiment suite.

This is a convenience entrypoint only. E01 and E02 retain their own canonical
run.py files, and each of those runs all five folds with the shared protocol:
public fold IDs 1..5, experiment seeds 1..5 matched to fold ID, and one fixed
CV split seed of 1.
"""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ROOT / "Experiment" / "01_Coordinate_Necessity" / "run.py",
    ROOT / "Experiment" / "02_LevelSet_Necessity" / "run.py",
]


def main():
    for run in RUNS:
        cmd = [sys.executable, str(run)]
        print("\n$", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
