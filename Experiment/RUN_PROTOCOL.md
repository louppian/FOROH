# Canonical Experiment Run Protocol

This protocol applies to every official experiment `run.py` unless a future experiment explicitly documents why cross-validation is inapplicable.

## Five-fold rule

- Official cross-validation is exactly 5 folds.
- A paper `run.py` must execute all five folds in one command.
- Official `run.py` files must not expose a single-fold paper mode.
- Public fold IDs are `1, 2, 3, 4, 5`.
- Canonical checkpoint names are `fold1.pt` through `fold5.pt`.

## Seed rule

- Experiment/training seed must equal the public fold ID.
- Therefore fold/seed pairs are exactly `(1,1), (2,2), (3,3), (4,4), (5,5)`.
- Python, NumPy, PyTorch CPU, and PyTorch CUDA RNG state must be reset before model construction for every fold.

## Split rule

- The CV split seed is fixed to `1` for the entire five-fold partition.
- It must not change with fold ID.
- Reason: changing the split seed per fold would generate different CV partitions and would not constitute one genuine five-fold cross-validation.

## Required result metadata

Every canonical `results.json` must record:

- `n_folds = 5`
- public fold IDs `1..5`
- internal zero-based fold indices `0..4`
- experiment seeds `1..5`
- fixed split seed `1`
- per-fold metrics
- mean and standard deviation across the five folds

Every checkpoint must record its public fold ID, internal fold index, experiment seed, and split seed.

## Current application

E01 and E02 implement this protocol directly. Future official experiment runners should copy this protocol rather than inventing a new fold/seed convention.
