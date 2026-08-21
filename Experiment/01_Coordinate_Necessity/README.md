# 01 Coordinate Necessity

## Question

Does angular parameterization provide benefit beyond matched scalar regression and normalization?

## Models

- `E01A` Matched Euclidean Huber
- `E01B` Normalized Cosine Regression
- `E01C` FOROH Angular Regression

## Implementation basis

This experiment now runs through `Experiment/train.py`, which imports the original root `3_train.py` and patches only the Phase-1 head/loss. The original dataset loader, augmentation, optimizer, scheduler, early stopping, metrics, and checkpoint flow are retained.

The root `3_train.py` itself is not modified.

## Original defaults retained

- LIMUC / ResNet50
- fold 0 of the original 5-fold setup
- seed 42
- projector dimension 128, dropout 0.3
- AdamW
- backbone LR 1e-4
- head LR 1e-4
- batch size 128
- 50 epochs
- cosine scheduler
- freeze layers 2
- Huber delta 0.5
- no frequency weighting unless explicitly requested

The three controls use the same original training path. Euclidean Huber uses the same 2-layer projector and an unclipped bias-free scalar readout; clipping occurs only at evaluation. Cosine and FOROH differ only in the score map from the normalized dot product.

## Run

```bash
python Experiment/01_Coordinate_Necessity/run.py
```

`run.sh` is a shell wrapper around `run.py`.

## Output

New reruns are written under:

`Result/01_Coordinate_Necessity/E01A|E01B|E01C/...`

Previous results produced by the retired modular engine must not be mixed with these reruns.

## Status

**Rewritten on the original code path; rerun required.**
