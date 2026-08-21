# FOROH Experiment Status

This file is the operational status board. `PLAN.md` defines *why* each experiment exists; this file defines *what is implemented and what has actually been run*.

## Current gate

The project is currently at **Phase-1 method validity**. No broad leaderboard or expensive stress test should be started before Experiments 01 and 02 are interpretable.

| ID | Experiment | Code status | Execution status | Next condition |
|---|---|---|---|---|
| 00 | Historical inventory / checkpoint verification | DONE | DONE on previous local copy | Re-run after major local file migration only |
| 01 | Coordinate Necessity | IMPLEMENTED | NOT RUN after current fixes | Run preflight, then E01A/E01B/E01C fold 0 |
| 02 | Level-Set Necessity | CORE IMPLEMENTED | NOT RUN after current fixes | Run only after/with 01; add representation analysis after numeric gate |
| 03 | Ordinal Positioning (CE/Huber/CORAL/CORN/GOL/FOROH) | NOT IMPLEMENTED in new engine | NOT RUN | Start only if 01/02 support the formulation |
| 04 | Small-N | PLANNED | NOT RUN | Requires fixed patient manifests + multi-seed runner |
| 05 | Imbalance | PLANNED | NOT RUN | Requires grade-specific retention manifests |
| 06 | Equal Spacing | PLANNED | NOT RUN | Requires controlled continuous-label setting |
| 07 | Single Axis | PLANNED | NOT RUN | Requires synthetic generator + GOL control |
| 08 | Score Transfer | PLANNED | NOT RUN | Requires 06/07 characterization first |
| 09 | Residual Probing | PLANNED | NOT RUN | Requires stable FOROH representation + phenotype labels |
| 10 | Temporal | DEFERRED | NOT RUN | Follow-up work |

## Scientific fixes applied before Phase-1 execution

1. **Matched scalar regression no longer clamps during training.** Clipping/rounding is evaluation-only.
2. **Matched trainable capacity:** E01A, E01B, E01C and E02A have the same projector and the same number of trainable head parameters.
3. **Point prototype no longer has an extra learnable azimuth vector.** The meridional reference is a fixed buffer orthogonalized against `w`.
4. **Point prototype loss no longer uses `Huber(score) + lambda * prototype_loss`.** It now uses a single Huber objective on full spherical geodesic error in grade units, removing the extra `lambda` confound.
5. **Class weighting is OFF by default and must be explicitly enabled.** The previous Phase-1 engine silently applied inverse-frequency weights to every method.
6. **LIMUC uses patient-level 10-fold split for the gate**, with `split_seed=42` separated from the model/training seed.
7. **Patient mapping is strict.** Missing or ambiguous filename-to-patient mapping aborts the run instead of creating pseudo-patients.
8. **Test set is evaluated once only after validation-based checkpoint selection.** The new engine has no boundary optimization/test-selection path.
9. **Outputs record split seed, parameter counts, sample paths, and patient IDs when available.**

## Required local preflight

From repository root:

```bash
python Experiment/common/preflight.py
```

Expected final line:

```text
PHASE-1 PREFLIGHT: PASS
```

This must pass before GPU training. A failure in patient mapping is a data-integrity issue and should not be bypassed for paper experiments.

## Immediate execution order

```bash
# 0. CPU/data integrity checks
python Experiment/common/preflight.py

# 1. Geometry necessity
bash Experiment/01_Coordinate_Necessity/run.sh

# 2. Level-set necessity
bash Experiment/02_LevelSet_Necessity/run.sh
```

After these finish, compare the metrics and inspect training stability before expanding folds, datasets, or baselines.

## Legacy code policy

- `Experiment/Legacy_Paper_Reproduction/` and root historical scripts remain evidence/reproduction tools.
- New scientific claims must use `Experiment/common/train.py` plus versioned YAML configs.
- Do not mix results from historical `outputs/` with new `Result/01_*` / `Result/02_*` results without explicitly labeling them as legacy.
