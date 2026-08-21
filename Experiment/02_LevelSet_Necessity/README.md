# 02 Level-Set Necessity

## Question

Is constraining only the ordinal polar coordinate better than collapsing each grade to a hyperspherical point?

## Models

- `E02A` Hyperspherical Point Prototype
- `E02B` FOROH Level Set

The point-prototype control uses the same backbone/projector/hypersphere and grade angular positions as FOROH. The prototype meridian reference is fixed rather than adding an extra learnable direction parameter. Point supervision uses the full spherical geodesic error with the same Huber delta, avoiding an auxiliary-loss weight confound.

## Fixed first-gate setting

Same LIMUC / ResNet50 / fold 0 / patient-level 10-fold / optimizer / projector settings as Experiment 01.

All settings are written directly in `run.py`; there is no YAML config layer.

## Run

```bash
python Experiment/common/preflight.py
python Experiment/02_LevelSet_Necessity/run.py
```

`run.sh` is only a thin shell wrapper.

## Output

`Result/02_LevelSet_Necessity/`

## Status

**Core numeric comparison implemented; current-code execution pending.** Representation-spread analysis is still pending and should be added only after the numeric gate is stable.
