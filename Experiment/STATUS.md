# FOROH Experiment Status

`PLAN.md` defines the research questions. This file records implementation and execution status.

## Current gate

The project is at Phase-1 method validity.

| ID | Experiment | Code status | Execution status | Next condition |
|---|---|---|---|---|
| 00 | Historical inventory / checkpoint verification | DONE | DONE | Re-run only after major migration |
| 01 | Coordinate Necessity | SELF-CONTAINED ORIGINAL-SPLIT IMPLEMENTATION | EXISTING 5-FOLD RESULTS RECOVERED | Run one reproduction check before deleting legacy engine/results |
| 02 | Level-Set Necessity | ORIGINAL-PIPELINE WRAPPER | PENDING | Migrate after E01 check |
| 03 | Ordinal Positioning | PLANNED | NOT RUN | Start after 01/02 gate |
| 04 | Small-N | PLANNED | NOT RUN | Later |
| 05 | Imbalance | PLANNED | NOT RUN | Later |
| 06 | Equal Spacing | PLANNED | NOT RUN | Later |
| 07 | Single Axis | PLANNED | NOT RUN | Later |
| 08 | Score Transfer | PLANNED | NOT RUN | Later |
| 09 | Residual Probing | PLANNED | NOT RUN | Later |
| 10 | Temporal | DEFERRED | NOT RUN | Follow-up |

## E01 canonical execution path

E01 no longer depends on `Experiment/train.py`, `Experiment/common`, `Model/`, or `Dataset/`.

```text
Experiment/01_Coordinate_Necessity/run.py
        ↓
Experiment/01_Coordinate_Necessity/train.py
        ├── models.py
        ├── dataset.py
        └── metrics.py
```

These files were split from the original root `3_train.py` mechanics. The root `3_train.py` remains untouched as the historical source reference.

## E01 fixed setting

- LIMUC / ResNet50
- patient-level 5-fold split
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

E01 variants:

- E01A: matched projector + unclipped bias-free Euclidean scalar Huber
- E01B: normalized cosine score `((1-u·w)/2) * Cmax`
- E01C: FOROH score `acos(u·w)/pi * Cmax`

## Existing recovered 5-fold result

| Model | MAE | QWK | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Euclidean Huber | 0.2439 ± 0.0028 | 0.8476 ± 0.0049 | 0.7622 ± 0.0032 | 0.6852 ± 0.0058 |
| Normalized Cosine | 0.2401 ± 0.0057 | 0.8497 ± 0.0044 | 0.7669 ± 0.0059 | 0.7032 ± 0.0080 |
| FOROH | 0.2403 ± 0.0033 | 0.8517 ± 0.0021 | 0.7654 ± 0.0036 | 0.7022 ± 0.0031 |

## Required migration check

Before deleting the previous shared engine and duplicate result directories, execute the new self-contained E01 implementation and verify split counts, parameter counts, best-epoch behavior, and metrics against the preserved checkpoint generation path.

```bash
python Experiment/01_Coordinate_Necessity/run.py
```

After this check passes, delete the obsolete E01 result generations and remove shared-engine code only where repo-wide dependency search confirms no active consumer. E02 is still using the original-pipeline wrapper and must be migrated before `Experiment/train.py` can be removed.
