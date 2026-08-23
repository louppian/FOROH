"""Run the complete E02 level-set necessity experiment.

One command always runs both E02 variants across all five folds. Public fold
IDs are 1..5. CV split seed and training/experiment seed are both fixed to
12345 inside train.py.
"""

from pathlib import Path

from train import run_variant


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RESULT_ROOT = REPO_ROOT / "Result" / "02_LevelSet_Necessity"

COMMON = dict(
    proj_dim=128,
    epochs=50,
    batch_size=128,
    lr=1e-4,
    lr_head=1e-4,
    weight_decay=1e-4,
    img_size=224,
    freeze_layers=2,
    patience=10,
    num_workers=4,
)

EXPERIMENTS = [
    ("E02A", "point_prototype"),
    ("E02B", "foroh"),
]


def main():
    for exp_id, variant in EXPERIMENTS:
        run_variant(variant, RESULT_ROOT / exp_id, **COMMON)


if __name__ == "__main__":
    main()
