#!/usr/bin/env bash
set -euo pipefail

PY="python Experiment/train.py"
COMMON="--batch-size 64 --lr 1e-4 --lr-head 1e-3 --optimizer adamw --scheduler cosine --patience 10 --freeze-layers 2 --seed 42"

run() {
  echo
  echo "============================================================"
  echo "$*"
  echo "============================================================"
  eval "$*"
}

run "$PY --method FOROH --dataset limuc --backbone resnet50 --n-folds 10 --fold -1 $COMMON --output-dir Result/05_Recovery/01_limuc_r50_foroh_10fold"
run "$PY --method ce --dataset limuc --backbone resnet50 --n-folds 10 --fold 0 $COMMON --output-dir Result/05_Recovery/02_limuc_r50_ce_fold0"
run "$PY --method FOROH --dataset aptos --backbone resnet50 --n-folds 5 --fold 0 $COMMON --output-dir Result/05_Recovery/03_aptos_r50_foroh_fold0"
run "$PY --method ce --dataset aptos --backbone resnet50 --n-folds 5 --fold 0 $COMMON --output-dir Result/05_Recovery/04_aptos_r50_ce_fold0"
run "$PY --method FOROH --dataset limuc --backbone resnet18 --n-folds 10 --fold 0 $COMMON --output-dir Result/05_Recovery/05_limuc_r18_foroh_fold0"
run "$PY --method ce --dataset limuc --backbone resnet18 --n-folds 10 --fold 0 $COMMON --output-dir Result/05_Recovery/06_limuc_r18_ce_fold0"
run "$PY --method FOROH --dataset limuc --backbone inception_v3 --n-folds 10 --fold 0 $COMMON --output-dir Result/05_Recovery/07_limuc_iv3_foroh_fold0"
run "$PY --method ce --dataset limuc --backbone inception_v3 --n-folds 10 --fold 0 $COMMON --output-dir Result/05_Recovery/08_limuc_iv3_ce_fold0"

for d in 32 64 256 512; do
  run "$PY --method FOROH --dataset limuc --backbone resnet50 --n-folds 10 --fold 0 --proj-dim $d $COMMON --output-dir Result/05_Recovery/09_proj_d${d}"
done

for loss in mse smooth_l1 l1; do
  run "$PY --method FOROH --dataset limuc --backbone resnet50 --n-folds 10 --fold 0 --loss-fn $loss $COMMON --output-dir Result/05_Recovery/10_loss_${loss}"
done

for f in 0 1 3 4; do
  run "$PY --method FOROH --dataset limuc --backbone resnet50 --n-folds 10 --fold 0 --freeze-layers $f --batch-size 64 --lr 1e-4 --lr-head 1e-3 --optimizer adamw --scheduler cosine --patience 10 --seed 42 --output-dir Result/05_Recovery/11_freeze_${f}"
done

echo "Historical recovery runs completed."
