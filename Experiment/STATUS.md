# FOROH Experiment Status

`PLAN.md` defines the research questions. This file records implementation and execution status.

## Current gate

The project is at Phase-1 method validity. Experiments 01 and 02 have been rewritten to use the original root `3_train.py` training pipeline rather than the temporary modular engine.

| ID | Experiment | Code status | Execution status | Next condition |
|---|---|---|---|---|
| 00 | Historical inventory / checkpoint verification | DONE | DONE | Re-run only after major migration |
| 01 | Coordinate Necessity | REWRITTEN ON ORIGINAL PIPELINE | RERUN REQUIRED | Run E01A/E01B/E01C fold 0 |
| 02 | Level-Set Necessity | REWRITTEN ON ORIGINAL PIPELINE | RERUN REQUIRED | Run E02A/E02B after 01 |
| 03 | Ordinal Positioning | PLANNED | NOT RUN | Start after 01/02 gate |
| 04 | Small-N | PLANNED | NOT RUN | Later |
| 05 | Imbalance | PLANNED | NOT RUN | Later |
| 06 | Equal Spacing | PLANNED | NOT RUN | Later |
| 07 | Single Axis | PLANNED | NOT RUN | Later |
| 08 | Score Transfer | PLANNED | NOT RUN | Later |
| 09 | Residual Probing | PLANNED | NOT RUN | Later |
| 10 | Temporal | DEFERRED | NOT RUN | Follow-up |

## Canonical Phase-1 execution path

```text
Experiment/01_Coordinate_Necessity/run.py
Experiment/02_LevelSet_Necessity/run.py
        ↓
Experiment/train.py
        ↓
original root 3_train.py
```

`Experiment/train.py` imports the original trainer and patches only the Phase-1 head/loss/evaluation extension points. The root `3_train.py` itself remains unchanged.

The previous `Experiment/common/train.py`, `Model/`, and `Dataset/` implementation is no longer the canonical Phase-1 path and should not be used for the rerun.

## Original defaults restored

The rerun intentionally follows the original `3_train.py` defaults:

- LIMUC / ResNet50
- 5 folds, fold 0 for the first gate
- seed 42
- batch size 128
- backbone LR 1e-4
- head LR 1e-4
- AdamW
- 50 epochs
- cosine scheduler
- freeze layers 2
- projector dimension 128
- Huber delta 0.5
- frequency weighting OFF unless explicitly requested

## Phase-1 definitions

Experiment 01:

- E01A: matched projector + unclipped bias-free Euclidean scalar Huber
- E01B: normalized cosine score `((1-u·w)/2) * Cmax`
- E01C: FOROH score `acos(u·w)/pi * Cmax`

Experiment 02:

- E02A: fixed-meridian hyperspherical point prototypes; full geodesic Huber training; nearest-prototype decoding
- E02B: FOROH level-set angular regression

## Execution

```bash
python Experiment/01_Coordinate_Necessity/run.py
python Experiment/02_LevelSet_Necessity/run.py
```

Results created by the previous modular engine are retained only as development evidence. They must not be compared directly with the new original-pipeline reruns.
