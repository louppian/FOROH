# FOROH Experiment Status

`PLAN.md` defines the research questions. `RUN_PROTOCOL.md` defines the mandatory execution/seed convention.

## Current gate

The project is at Phase-1 method validity.

| ID | Experiment | Code status | Execution status | Next condition |
|---|---|---|---|---|
| 00 | Historical inventory / checkpoint verification | DONE | DONE | Re-run only after major migration |
| 01 | Coordinate Necessity | SELF-CONTAINED | CANONICAL RERUN REQUIRED | Run new 1..5 fold/seed protocol |
| 02 | Level-Set Necessity | SELF-CONTAINED | CANONICAL RERUN REQUIRED | Run after/with E01 |
| 03 | Ordinal Positioning | PLANNED | NOT RUN | Must follow RUN_PROTOCOL.md |
| 04 | Small-N | PLANNED | NOT RUN | Must follow RUN_PROTOCOL.md where CV applies |
| 05 | Imbalance | PLANNED | NOT RUN | Must follow RUN_PROTOCOL.md where CV applies |
| 06 | Equal Spacing | PLANNED | NOT RUN | Later |
| 07 | Single Axis | PLANNED | NOT RUN | Later |
| 08 | Score Transfer | PLANNED | NOT RUN | Later |
| 09 | Residual Probing | PLANNED | NOT RUN | Later |
| 10 | Temporal | DEFERRED | NOT RUN | Follow-up |

## Mandatory run protocol

For every official CV run:

- exactly 5 folds;
- one command runs all five folds;
- public fold IDs are `1,2,3,4,5`;
- experiment/training seed equals fold ID, therefore seeds are `1,2,3,4,5`;
- CV split seed is fixed to `1` across all five folds;
- RNG state is reset before model construction for each fold;
- checkpoints are `fold1.pt` through `fold5.pt`;
- `results.json` contains all five folds plus mean/std and seed metadata.

The split seed is deliberately not changed per fold. A different split seed per fold would produce five unrelated partitions rather than one valid 5-fold CV partition.

## Canonical Phase-1 paths

```text
Experiment/01_Coordinate_Necessity/run.py
        ↓
Experiment/01_Coordinate_Necessity/train.py
        ├── models.py
        ├── dataset.py
        └── metrics.py

Experiment/02_LevelSet_Necessity/run.py
        ↓
Experiment/02_LevelSet_Necessity/train.py
        ├── models.py
        ├── dataset.py
        └── metrics.py
```

The obsolete shared Phase-1 wrapper/engine and old E01/E02 result directories have been removed from `reproduce-paper`.

## Fixed Phase-1 training setting

- LIMUC / ResNet50
- patient-level 5-fold split
- split seed 1
- experiment seeds 1,2,3,4,5 matched to folds 1,2,3,4,5
- batch size 128
- backbone LR 1e-4
- head LR 1e-4
- AdamW
- 50 epochs
- cosine scheduler
- freeze layers 2
- projector dimension 128
- Huber delta 0.5

## Result status

All pre-protocol E01/E02 results have been removed from the active branch. Canonical Phase-1 results must be regenerated using:

```bash
python Experiment/01_Coordinate_Necessity/run.py
python Experiment/02_LevelSet_Necessity/run.py
```
