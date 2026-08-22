# 01 Coordinate Necessity

## Question

Does hyperspherical normalization help beyond matched Euclidean scalar regression, and does the FOROH angular coordinate add value beyond normalized cosine scoring?

## Models

- `E01A` Matched Euclidean Huber
- `E01B` Normalized Cosine Regression
- `E01C` FOROH Angular Regression

## Canonical implementation

E01 is now self-contained. The code was split from the original root `3_train.py` rather than routed through the later shared `Experiment/common` engine.

```text
Experiment/01_Coordinate_Necessity/
├── run.py
├── train.py
├── models.py
├── dataset.py
├── metrics.py
└── README.md
```

The training mechanics follow the original LIMUC/ResNet50 pipeline. The only intended experimental difference is the head/score mapping.

## Fixed setting

- LIMUC
- ResNet50
- patient-level 5-fold split
- seed 42
- projection dimension 128
- dropout 0.3
- AdamW
- backbone LR 1e-4
- head LR 1e-4
- weight decay 1e-4
- batch 128
- 50 epochs
- cosine scheduler
- early stopping patience 10
- freeze layers 2
- Huber delta 0.5

## Run

Run the complete paper experiment with one command:

```bash
python Experiment/01_Coordinate_Necessity/run.py
```

`run.py` always executes E01A/E01B/E01C across folds 0-4 and writes a true 5-fold `results.json` for each model. Individual fold paper runs are intentionally not exposed through this entrypoint, preventing the previous single-fold `results.json` overwrite problem.

## Output

```text
Result/01_Coordinate_Necessity/
├── E01A/FOROH_limuc/
├── E01B/FOROH_limuc/
└── E01C/FOROH_limuc/
```

Each canonical model directory contains `fold0.pt` through `fold4.pt` plus one aggregate `results.json`.

## Existing validated 5-fold result

| Model | MAE | QWK | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Euclidean Huber | 0.2439 ± 0.0028 | 0.8476 ± 0.0049 | 0.7622 ± 0.0032 | 0.6852 ± 0.0058 |
| Normalized Cosine | 0.2401 ± 0.0057 | 0.8497 ± 0.0044 | 0.7669 ± 0.0059 | 0.7032 ± 0.0080 |
| FOROH | 0.2403 ± 0.0033 | 0.8517 ± 0.0021 | 0.7654 ± 0.0036 | 0.7022 ± 0.0031 |

## Migration status

The previous checkpoints/results are preserved. Before deleting the old shared engine and duplicate result directories, run one reproduction check with this self-contained E01 code and compare split counts, best-epoch behavior, and fold metrics against the preserved checkpoints.
