# 02 Level-Set Necessity

## Question

Is constraining only the ordinal polar coordinate better than collapsing each grade to a hyperspherical point?

## Models

- `E02A` Hyperspherical Point Prototype
- `E02B` FOROH Level Set

## Implementation basis

This experiment uses the same `Experiment/train.py` wrapper over the original root `3_train.py` as Experiment 01. The original data, transform, optimizer, scheduler, early-stopping, metric, and save paths are retained; only the Phase-1 head/loss/point decoding are extended.

The point-prototype control uses the same backbone, projector, hypersphere, learnable severity axis, and grade angular positions as FOROH. A fixed meridian reference supplies the orthogonal direction without adding another trainable vector.

Training minimizes Huber on the full spherical geodesic distance to the target grade prototype. Evaluation uses nearest-prototype decoding. FOROH continues to use the original polar-angle score and rounding.

## Original defaults retained

Same as Experiment 01: LIMUC / ResNet50 / original 5-fold fold 0 / seed 42 / batch 128 / AdamW / LR 1e-4 for backbone and head / 50 epochs / cosine schedule / freeze layers 2.

## Run

```bash
python Experiment/02_LevelSet_Necessity/run.py
```

## Output

New reruns are written under:

`Result/02_LevelSet_Necessity/E02A|E02B/...`

Previous results produced by the retired modular engine must not be mixed with these reruns.

## Status

**Rewritten on the original code path; rerun required.**
