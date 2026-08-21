# 02 Level-Set Necessity

## Question

Is constraining only the ordinal polar coordinate better than collapsing each grade to a hyperspherical point?

## Models

- `E02A` Hyperspherical Point Prototype
- `E02B` FOROH Level Set

Both use the same ResNet50 backbone, projector, hypersphere dimension, learnable severity axis, optimizer, split, and grade angular positions.

The point control uses prototypes on one meridian:

```text
a_y = cos(theta_y) w + sin(theta_y) q
```

`q` is derived from a fixed random buffer and is not trainable, so E02A has the same trainable parameter count as FOROH. E02A is trained by Huber loss on full spherical geodesic error expressed in grade units; FOROH is trained by Huber loss on polar score error. This avoids an extra tunable prototype-loss coefficient.

## Gate setting

- Dataset: LIMUC
- Fold: 0 of patient-level 10-fold split
- Split seed: 42
- Training seed: 42
- Backbone: ResNet50
- Projection dimension: 128
- Class weighting: OFF
- Remaining optimizer/training settings identical to Experiment 01

## Required preflight

```bash
python Experiment/common/preflight.py
```

## Run

```bash
bash Experiment/02_LevelSet_Necessity/run.sh
```

## Evaluation

Initial gate: MAE, QWK, Macro-F1, class-wise recall.

Representation analysis (same-grade angular spread / between-grade separation) is still pending and should be added only after the numerical gate runs successfully.

## Output

`Result/02_LevelSet_Necessity/`

## Status

**CORE TRAINING IMPLEMENTED — REPRESENTATION ANALYSIS NOT YET IMPLEMENTED — NOT YET RUN.**

Proceed to broad ordinal positioning only after 01/02 results are interpretable.
