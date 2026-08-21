# FOROH paper reproduction

This directory reproduces the experiments in the order used by the FOROH draft.
The root scripts and historical `outputs/` directory are intentionally left unchanged.

## Paper setup used here

- Primary dataset: LIMUC, patient-level stratified 10-fold CV + held-out test
- APTOS: 5-fold image-level split; the draft also reports fold-0 comparisons
- Metrics: MAE, QWK (primary), Accuracy, Macro F1, class-wise recall
- Optimizer: AdamW
- Backbone LR: 1e-4
- Head LR: 1e-3
- Scheduler: cosine
- Early stopping patience: 10
- Batch size: 64
- Default FOROH: projection dim 128, Huber delta 0.5, learnable axis, projector enabled, freeze_layers=2

## Reproduction order

### 01 Main Comparison

`Experiment/01_Main_Comparison/run.sh`

1. LIMUC 10-fold: FOROH, CE, CORAL, CORN, MSE
2. APTOS fold 0: FOROH vs CE
3. APTOS full 5-fold: FOROH vs CE (setup-level reproduction)

Draft target for LIMUC ResNet-50 10-fold FOROH:

- MAE: 0.239 +/- 0.009
- QWK: 0.851 +/- 0.007
- ACC: 0.767 +/- 0.009
- Macro F1: 0.701 +/- 0.014

Draft target for APTOS ResNet-50 fold 0 FOROH:

- MAE: 0.201
- QWK: 0.915
- ACC: 0.834
- Macro F1: 0.663

### 02 Backbone Scaling

`Experiment/02_Backbone_Scaling/run.sh`

LIMUC fold 0, FOROH vs CE on:

1. ResNet-18
2. Inception-v3
3. ResNet-50

### 03 Score Function

`Experiment/03_Score_Function/run.sh`

LIMUC fold 0, ResNet-50:

1. arccos score: `acos(u.w) / pi * C_max`
2. cosine regression: `(1 - u.w) / 2 * C_max`

The score-function experiment reproduces the numerical ablation only. The draft's
mechanistic explanation of an "implicit curriculum" is marked inside the draft as
requiring revision, so it should not be treated as established by this script.

### 04 Ablation

`Experiment/04_Ablation/run.sh`

Run in this order:

1. Projection dimension: 32, 64, 128, 256, 512
2. Loss: Huber(delta=0.5), L1, SmoothL1, MSE
3. Severity axis: learnable vs fixed random
4. Projector: with vs without
5. Frozen stages: 0, 1, 2, 3, 4

Draft default / fold-0 reference: MAE 0.233, QWK 0.861.

## Running everything

From the repository root:

```bash
bash Experiment/run_all.sh
```

Results are written under matching numbered directories in `Result/`.

## Data locations expected by the current code

```text
data/
├── limuc/
│   ├── train_and_validation_sets/
│   ├── patient_based_classified_images/
│   └── test_set/
├── aptos2019/
│   ├── train.csv
│   └── train_images/
└── kneexray/
    └── KneeXrayData/ClsKLData/kneeKL224/
```

Raw datasets remain ignored by git.

## Important reproducibility notes

1. The original root `run.sh` does **not** match the setup summarized in the draft for the main experiment. It contains later/alternate dataset-specific recipes (e.g. LIMUC Inception-v3 + Adam + ReduceLR + oversampling). The numbered reproduction scripts therefore use the draft setup explicitly.
2. `5_figure.py` is stale relative to the current FOROH head API and should not be used until the numerical checkpoints are reproduced and the analysis script is updated.
3. The draft itself marks the arccos-gradient mechanism and the `sin(theta)` uncertainty interpretation as requiring correction. Numerical experiments can be reproduced independently of those interpretations.
