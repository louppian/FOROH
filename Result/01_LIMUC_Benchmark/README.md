# 01 LIMUC Benchmark Results

This directory is reserved for the paper-facing LIMUC benchmark:

- CE.
- CDW-CE with alpha sweep.
- Ours.

The benchmark must use the fixed official fold JSON files under
`Dataset/LIMUC/cross_validation_folds_train_val_info/`. There is no split seed
for E01 results.

Expected run layout:

```text
Result/01_LIMUC_Benchmark/
└── <backbone>/
    └── <loss>/
        ├── config.json
        ├── fold1.pt
        ├── ...
        ├── fold10.pt
        ├── fold1_predictions.csv
        ├── ...
        ├── fold10_predictions.csv
        ├── results.csv
        └── results.json
```
