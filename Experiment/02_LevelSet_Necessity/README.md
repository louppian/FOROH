# 02 Level-Set Necessity

## Question

Is constraining only the ordinal polar coordinate better than collapsing each grade to a hyperspherical point?

## Models

- `E02A` Hyperspherical Point Prototype
- `E02B` FOROH Level Set

E02 is now self-contained and no longer depends on `Experiment/train.py`.

```text
Experiment/02_LevelSet_Necessity/
├── run.py
├── train.py
├── models.py
├── dataset.py
├── metrics.py
└── README.md
```

The point-prototype control uses the same ResNet50 backbone, projector, hypersphere, and ordinal angular positions as FOROH. It uses full geodesic Huber supervision to the target prototype and nearest-prototype decoding.

## Official cross-validation / seed protocol

The protocol is fixed in code and matches E01 exactly.

- Exactly 5 patient-level folds.
- Public fold IDs: `1, 2, 3, 4, 5`.
- Experiment/training seed equals fold ID: `1, 2, 3, 4, 5`.
- CV split seed is fixed to `1` for the entire five-fold partition.
- Python, NumPy, PyTorch, and CUDA RNG state is reset before every fold.
- One `run.py` call executes both variants and all five folds.
- Canonical checkpoints are `fold1.pt` through `fold5.pt`.

A single fixed split seed is necessary for a genuine 5-fold partition; changing the split seed independently for each fold is not allowed.

## Fixed training setting

Same as E01: LIMUC, ResNet50, proj-dim 128, batch 128, AdamW, backbone/head LR 1e-4, weight decay 1e-4, 50 epochs, cosine scheduler, patience 10, freeze layers 2, Huber delta 0.5.

## Run

```bash
python Experiment/02_LevelSet_Necessity/run.py
```

There is no official single-fold entrypoint.

## Output

```text
Result/02_LevelSet_Necessity/
├── E02A/FOROH_limuc/
│   ├── fold1.pt
│   ├── fold2.pt
│   ├── fold3.pt
│   ├── fold4.pt
│   ├── fold5.pt
│   └── results.json
└── E02B/FOROH_limuc/
```
