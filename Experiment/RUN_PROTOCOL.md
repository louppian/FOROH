# Canonical Experiment Run Protocol

This protocol applies to every official experiment `run.py` unless a future experiment explicitly documents why cross-validation is inapplicable.

## Five-fold rule

- Official cross-validation is exactly 5 folds for E01/E02.
- A paper `run.py` must execute all five folds in one command.
- Official `run.py` files must not expose a single-fold paper mode.
- Public fold IDs are `1, 2, 3, 4, 5`.
- Canonical checkpoint names are `fold1.pt` through `fold5.pt`.

## Seed rule

- Experiment/training seed is fixed to `12345` for every fold.
- Python, NumPy, PyTorch CPU, and PyTorch CUDA RNG state must be reset to `12345` before model construction for every fold.

## Split rule

- The CV split seed is fixed to `12345` for the complete cross-validation partition.
- It must not change with fold ID.

## Required result metadata

Every canonical `results.json` must record:

- the experiment's `n_folds`
- public fold IDs
- internal zero-based fold indices
- experiment seed `12345`
- split seed `12345`
- per-fold metrics
- mean and standard deviation across folds

Every checkpoint must record its public fold ID, internal fold index, experiment seed, and split seed.

## Current application

- E00 reproduction uses seed `12345`; LIMUC uses 10 folds and APTOS uses 5 folds according to the paper setup.
- E01 and E02 use five folds with split seed `12345` and experiment/training seed `12345`.
- Future official experiment runners should use the same fixed seed unless explicitly documented otherwise.
