"""Experiment 01: angular coordinate necessity.

Settings live here directly; there is no YAML layer.
"""

from Experiment.common.train import run_experiment

COMMON = dict(
    dataset="limuc",
    backbone="resnet50",
    fold=0,
    n_folds=10,
    split_seed=42,
    seed=42,
    proj_dim=128,
    dropout=0.3,
    optimizer="adamw",
    lr_backbone=1e-4,
    lr_head=1e-3,
    weight_decay=1e-4,
    batch_size=64,
    epochs=50,
    scheduler="cosine",
    patience=10,
    freeze_layers=2,
    img_size=224,
    huber_delta=0.5,
    class_weighting=False,
    num_workers=4,
    output_dir="Result/01_Coordinate_Necessity",
)

EXPERIMENTS = [
    dict(COMMON, experiment_id="E01A", method="euclidean_huber"),
    dict(COMMON, experiment_id="E01B", method="normalized_cosine"),
    dict(COMMON, experiment_id="E01C", method="foroh"),
]


if __name__ == "__main__":
    for experiment in EXPERIMENTS:
        run_experiment(experiment)
