#!/usr/bin/env bash
set -euo pipefail

T="python Experiment/train.py"
COMMON="--method FOROH --dataset limuc --backbone resnet50 --fold 0 --n-folds 10 --optimizer adamw --lr 1e-4 --lr-head 1e-3 --scheduler cosine --batch-size 64 --patience 10 --freeze-layers 2"

# Default FOROH score: acos(u.w)/pi * C_max
$T $COMMON --output-dir Result/03_Score_Function/arccos

# Paper cosine-regression ablation: ((1-u.w)/2) * C_max
$T $COMMON --no-arccos --output-dir Result/03_Score_Function/cosine
