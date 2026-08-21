# 01 Coordinate Necessity

## Question

Does angular parameterization provide benefit beyond matched scalar regression and normalization?

## Models

- `E01A` Matched Euclidean Huber
- `E01B` Normalized Cosine Regression
- `E01C` FOROH Angular Regression

## Fixed first-gate setting

- LIMUC
- ResNet50
- patient-level 10-fold split, fold 0
- split seed 42, training seed 42
- same 2-layer projector, projection dimension 128, dropout 0.3
- AdamW, backbone LR 1e-4, head LR 1e-3
- batch 64, cosine schedule, early stopping patience 10
- Huber delta 0.5
- class weighting OFF

All settings are written directly in `run.py`; there is no YAML config layer.

## Run

```bash
python Experiment/common/preflight.py
python Experiment/01_Coordinate_Necessity/run.py
```

`run.sh` is only a thin shell wrapper around the same `run.py`.

## Output

`Result/01_Coordinate_Necessity/`

Each completed run writes `config.json`, `run_manifest.json`, `metrics.json`, `history.json`, `predictions.csv`, and `checkpoint.pt`.

## Status

**Core implementation complete; current-code execution pending.** Run preflight before GPU training. Do not substitute the historical cosine/arccos reproduction script under `Legacy_Paper_Reproduction`.
