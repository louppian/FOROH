# 01 Coordinate Necessity

## Question

Does angular parameterization provide benefit beyond matched scalar regression and normalization?

## Models

- `E01A` Matched Euclidean Huber
- `E01B` Normalized Cosine Regression
- `E01C` FOROH Angular Regression

All three use the same ResNet50 backbone, the same two-layer projector, the same projection dimension, and the same number of trainable direction/readout parameters. E01A is not clamped during training; clipping and rounding are evaluation-only.

## Gate setting

- Dataset: LIMUC
- Fold: 0 of a patient-level 10-fold split
- Split seed: 42
- Training seed: 42
- Backbone: ResNet50
- Projection dimension: 128
- Optimizer: AdamW
- Backbone LR: 1e-4
- Head LR: 1e-3
- Batch size: 64
- Scheduler: cosine
- Patience: 10
- Frozen stages: 2
- Huber delta: 0.5
- Class weighting: OFF

## Required preflight

Before GPU training:

```bash
python Experiment/common/preflight.py
```

The preflight must report `PHASE-1 PREFLIGHT: PASS`. It checks LIMUC patient mapping, train/validation patient disjointness, and matched head parameter counts.

## Run

```bash
bash Experiment/01_Coordinate_Necessity/run.sh
```

## Primary metrics

MAE, QWK, Macro-F1, class-wise recall.

## Output

`Result/01_Coordinate_Necessity/`

Each run writes `config.yaml`, `run_manifest.json`, `metrics.json`, `predictions.csv`, `history.json`, and `checkpoint.pt`.

## Status

**IMPLEMENTED — NOT YET RUN/VALIDATED ON LOCAL DATA.**

Do not use the historical `03_Score_Function` script as a substitute. It is archived under `Experiment/Legacy_Paper_Reproduction/`.
