# 02 Level-Set Necessity

## Question

Is constraining only the ordinal polar coordinate better than collapsing each grade to a hyperspherical point?

## Models

- `E02A` Hyperspherical Point Prototype
- `E02B` FOROH Level Set

Use the same hypersphere, dimension, backbone, projector, optimizer, and grade angular positions. Point prototypes must lie on a common meridian so the only intended difference is whether azimuth is fixed or free.

## Evaluation

MAE, QWK, Macro-F1, same-grade angular spread, between-grade separation, generalization gap, and later small-n degradation.

## Output

`Result/02_LevelSet_Necessity/<run-id>/`

## Status

**Implementation pending.** This is the second Phase-1 gate. Proceed to broad baseline positioning only after 01/02 are interpretable.
