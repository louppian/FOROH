#!/usr/bin/env bash
set -euo pipefail

T="python Experiment/train.py"
BASE="--method FOROH --dataset limuc --backbone resnet50 --fold 0 --n-folds 10 --optimizer adamw --lr 1e-4 --lr-head 1e-3 --scheduler cosine --batch-size 64 --patience 10"

for D in 32 64 128 256 512; do
  $T $BASE --freeze-layers 2 --proj-dim "$D" \
     --output-dir "Result/04_Ablation/01_Projection_Dim/d${D}"
done

for LOSS in huber l1 smooth_l1 mse; do
  $T $BASE --freeze-layers 2 --loss-fn "$LOSS" \
     --output-dir "Result/04_Ablation/02_Loss/${LOSS}"
done

$T $BASE --freeze-layers 2 --output-dir Result/04_Ablation/03_Axis/learnable
$T $BASE --freeze-layers 2 --fixed-w --output-dir Result/04_Ablation/03_Axis/fixed_random

$T $BASE --freeze-layers 2 --output-dir Result/04_Ablation/04_Projector/with
$T $BASE --freeze-layers 2 --no-projector --output-dir Result/04_Ablation/04_Projector/without

for FL in 0 1 2 3 4; do
  $T $BASE --freeze-layers "$FL" \
     --output-dir "Result/04_Ablation/05_Freeze/fl${FL}"
done
