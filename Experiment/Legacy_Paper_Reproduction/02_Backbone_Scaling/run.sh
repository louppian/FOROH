#!/usr/bin/env bash
set -euo pipefail

T="python Experiment/train.py"
COMMON="--dataset limuc --fold 0 --n-folds 10 --optimizer adamw --lr 1e-4 --lr-head 1e-3 --scheduler cosine --batch-size 64 --patience 10 --freeze-layers 2"

# Paper Table 2: LIMUC fold 0, FOROH vs CE across three backbones.
for BB in resnet18 inception_v3 resnet50; do
  for METHOD in FOROH ce; do
    $T --method "$METHOD" --backbone "$BB" $COMMON \
       --output-dir Result/02_Backbone_Scaling
  done
done
