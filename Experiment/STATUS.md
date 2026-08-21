# FOROH Experiment Status

This is the operational status board. `PLAN.md` defines why each experiment exists; this file records what is implemented and what has actually been executed.

## Current gate

The project is at **Phase-1 method validity**. Experiments 01 and 02 must be interpretable before broad baseline or stress-test expansion.

| ID | Experiment | Code status | Execution status | Next condition |
|---|---|---|---|---|
| 00 | Historical inventory / checkpoint verification | DONE | DONE on previous local copy | Re-run after major local file migration only |
| 01 | Coordinate Necessity | IMPLEMENTED | NOT RUN after current fixes | Preflight PASS, then E01A/E01B/E01C fold 0 |
| 02 | Level-Set Necessity | CORE IMPLEMENTED | NOT RUN after current fixes | Run after/with 01; representation analysis afterward |
| 03 | Ordinal Positioning | NOT IMPLEMENTED in new engine | NOT RUN | Start after 01/02 gate |
| 04 | Small-N | PLANNED | NOT RUN | Requires fixed patient manifests + multi-seed runner |
| 05 | Imbalance | PLANNED | NOT RUN | Requires grade-specific retention manifests |
| 06 | Equal Spacing | PLANNED | NOT RUN | Requires controlled continuous-label setting |
| 07 | Single Axis | PLANNED | NOT RUN | Requires synthetic generator + GOL control |
| 08 | Score Transfer | PLANNED | NOT RUN | Requires 06/07 characterization first |
| 09 | Residual Probing | PLANNED | NOT RUN | Requires stable FOROH representation + phenotype labels |
| 10 | Temporal | DEFERRED | NOT RUN | Follow-up work |

## Current implementation style

YAML experiment configs have been removed. New experiments use direct Python settings, matching the simpler original project style:

```text
Experiment/01_Coordinate_Necessity/run.py
Experiment/02_LevelSet_Necessity/run.py
        ↓
Experiment/common/train.py
        ↓
Model/ + Dataset/
```

The experiment-specific values are visible directly in each `run.py`. `Experiment/common/train.py` contains only the shared training/evaluation machinery.

## Scientific fixes applied before execution

1. Matched scalar regression is unclipped during training; clipping/rounding is evaluation-only.
2. E01A/E01B/E01C/E02A use matched projector capacity and matched trainable head parameter counts.
3. Point prototype has no extra learnable azimuth direction.
4. Point prototype uses full spherical geodesic Huber supervision without an auxiliary `lambda`.
5. Class weighting is OFF by default.
6. LIMUC gate uses patient-level 10-fold split; split seed and training seed are separate.
7. Patient mapping is strict: missing/ambiguous mappings abort the run.
8. Validation selects the checkpoint; test is evaluated once afterward.
9. New outputs use JSON/CSV/PT only; no YAML is emitted.

## Required local preflight

From repository root:

```bash
python Experiment/common/preflight.py
```

Expected final line:

```text
PHASE-1 PREFLIGHT: PASS
```

Do not bypass a patient-map failure for paper experiments.

## Immediate execution order

```bash
# 0. CPU/data integrity check
python Experiment/common/preflight.py

# 1. Coordinate necessity: E01A, E01B, E01C
python Experiment/01_Coordinate_Necessity/run.py

# 2. Level-set necessity: E02A, E02B
python Experiment/02_LevelSet_Necessity/run.py
```

## Legacy policy

- `Experiment/Legacy_Paper_Reproduction/`, root `3_train.py`, and `Experiment/train.py` are historical reproduction tools.
- New scientific claims use the direct `run.py -> Experiment/common/train.py` path.
- Historical `outputs/` and new `Result/` must not be mixed without an explicit legacy label.
