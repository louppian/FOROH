# 01 Coordinate Necessity

## Question

Does hyperspherical normalization help beyond matched Euclidean scalar regression, and does the FOROH angular coordinate add value beyond normalized cosine scoring?

## Models

- `E01A` Matched Euclidean Huber
- `E01B` Normalized Cosine Regression
- `E01C` FOROH Angular Regression

## Canonical implementation

E01 is self-contained and follows the training mechanics of the original root `3_train.py`.

```text
Experiment/01_Coordinate_Necessity/
├── run.py
├── train.py
├── models.py
├── dataset.py
├── metrics.py
└── README.md
```

## Official cross-validation / seed protocol

This protocol is fixed in code and is not exposed as a paper-run option.

- Exactly 5 patient-level folds.
- Public fold IDs are `1, 2, 3, 4, 5`.
- Experiment/training seed equals the public fold ID: `1, 2, 3, 4, 5`.
- CV split seed is fixed to `1` for the entire five-fold partition.
- The split seed must remain fixed across folds; otherwise the experiment is not a genuine 5-fold partition.
- Every fold resets Python, NumPy, PyTorch, and CUDA RNG state before model creation.
- One `run.py` call executes all models and all five folds.
- Canonical checkpoints are named `fold1.pt` through `fold5.pt`.

## Fixed training setting

- LIMUC
- ResNet50
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

```bash
python Experiment/01_Coordinate_Necessity/run.py
```

There is no official single-fold entrypoint.

## Output

```text
Result/01_Coordinate_Necessity/
├── E01A/FOROH_limuc/
│   ├── fold1.pt
│   ├── fold2.pt
│   ├── fold3.pt
│   ├── fold4.pt
│   ├── fold5.pt
│   └── results.json
├── E01B/FOROH_limuc/
└── E01C/FOROH_limuc/
```

`results.json` records fold IDs, zero-based internal split indices, experiment seeds, the fixed split seed, per-fold metrics, mean, and standard deviation.

## Previous result status

The previously recovered 5-fold numbers were produced under the earlier seed-42 protocol. They are retained only as historical/development evidence and are not the canonical result after this protocol change. Canonical E01 numbers must come from a fresh run of the command above.
