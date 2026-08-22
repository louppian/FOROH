"""Run the complete E01 coordinate-necessity experiment.

This file is the only paper-run entrypoint for E01. It always runs all five
folds for the three controlled variants and writes one 5-fold results.json per
variant.
"""

from pathlib import Path

from train import run_variant


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RESULT_ROOT = REPO_ROOT / "Result" / "01_Coordinate_Necessity"

COMMON = dict(
    n_folds=5,
    seed=42,
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
    ("E01A", "euclidean_huber"),
    ("E01B", "normalized_cosine"),
    ("E01C", "foroh"),
]


def main():
    for exp_id, variant in EXPERIMENTS:
        run_variant(variant, RESULT_ROOT / exp_id, **COMMON)


if __name__ == "__main__":
    main()
