"""Run the complete E01 coordinate-necessity experiment.

Official paper entrypoint. One command always runs E01A/E01B/E01C over all
five folds. Fold IDs are 1..5 and each fold uses the same-numbered experiment
seed. The CV partition itself uses one fixed split seed inside train.py.
"""

from pathlib import Path

from train import run_variant


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RESULT_ROOT = REPO_ROOT / "Result" / "01_Coordinate_Necessity"

COMMON = dict(proj_dim=128, epochs=50, batch_size=128, lr=1e-4, lr_head=1e-3, weight_decay=1e-4, img_size=224,
    freeze_layers=2, patience=10, num_workers=4,)

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
