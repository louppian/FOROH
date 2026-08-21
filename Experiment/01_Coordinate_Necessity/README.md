# 01 Coordinate Necessity

## Question

Does angular parameterization provide benefit beyond matched scalar regression and normalization?

## Phase-1 models

- `E01A` Matched Euclidean Huber
- `E01B` Normalized Cosine Regression
- `E01C` FOROH Angular Regression

## First gate setting

- Dataset: LIMUC
- Backbone: ResNet50
- Same patient split / fold
- Same projector capacity where applicable
- Same optimizer, LR, augmentation, schedule, Huber objective

## Primary metrics

MAE, QWK, Macro-F1, class-wise recall.

## Output

`Result/01_Coordinate_Necessity/<run-id>/`

## Status

**Implementation pending.** Do not substitute the old `03_Score_Function` script: that script only compared cosine vs arccos and is preserved under `Experiment/Legacy_Paper_Reproduction/03_Score_Function/`.

See `Experiment/PLAN.md` for the full design and gate criterion.
