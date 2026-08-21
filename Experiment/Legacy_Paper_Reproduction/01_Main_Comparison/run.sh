#!/usr/bin/env bash
set -euo pipefail

T="python Experiment/train.py"
COMMON="--backbone resnet50 --optimizer adamw --lr 1e-4 --lr-head 1e-3 --scheduler cosine --batch-size 64 --patience 10 --freeze-layers 2"

# Paper Table 1: LIMUC, patient-level stratified 10-fold CV + held-out test.
for METHOD in FOROH ce coral corn mse; do
  $T --method "$METHOD" --dataset limuc --fold -1 --n-folds 10 $COMMON \
     --output-dir Result/01_Main_Comparison/LIMUC
 done

# Paper APTOS fold-0 comparison (Table / class-wise analysis).
# APTOS has no labeled official test set in this code, so validation fold is evaluation fold.
for METHOD in FOROH ce; do
  $T --method "$METHOD" --dataset aptos --fold 0 --n-folds 5 $COMMON \
     --output-dir Result/01_Main_Comparison/APTOS_fold0
 done

# Optional full 5-fold APTOS reproduction of the setup described in Section 3.1.
for METHOD in FOROH ce; do
  $T --method "$METHOD" --dataset aptos --fold -1 --n-folds 5 $COMMON \
     --output-dir Result/01_Main_Comparison/APTOS_5fold
 done
