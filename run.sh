# FOROH 실험 목록 (번호별)

## LIMUC — Iv3, Adam 2e-4 (single LR), ReduceLR, oversample, 10-fold
## 출처: Polat 2023 (IBD, doi:10.1093/ibd/izac226)
# Exp 1: CDW-CE† 원저 보고값 (QWK 0.868±0.006) — 실행 불필요
# Exp 2~8: 우리 실행
python 3_train.py --exp 2  --method ce    --dataset limuc --backbone inception_v3 --optimizer adam --lr 2e-4 --lr-head 2e-4 --scheduler reduce_lr --scheduler-patience 10 --patience 25 --oversample --fold -1 --n-folds 10
python 3_train.py --exp 3  --method coral --dataset limuc --backbone inception_v3 --optimizer adam --lr 2e-4 --lr-head 2e-4 --scheduler reduce_lr --scheduler-patience 10 --patience 25 --oversample --fold -1 --n-folds 10
python 3_train.py --exp 4  --method corn  --dataset limuc --backbone inception_v3 --optimizer adam --lr 2e-4 --lr-head 2e-4 --scheduler reduce_lr --scheduler-patience 10 --patience 25 --oversample --fold -1 --n-folds 10
python 3_train.py --exp 5  --method mse   --dataset limuc --backbone inception_v3 --optimizer adam --lr 2e-4 --lr-head 2e-4 --scheduler reduce_lr --scheduler-patience 10 --patience 25 --oversample --fold -1 --n-folds 10
python 3_train.py --exp 6  --method FOROH --dataset limuc --backbone inception_v3 --optimizer adam --lr 2e-4 --lr-head 2e-4 --scheduler reduce_lr --scheduler-patience 10 --patience 25 --oversample --fold -1 --n-folds 10
python 3_train.py --exp 7  --method FOROH --dataset limuc --backbone inception_v3 --optimizer adam --lr 2e-4 --lr-head 2e-4 --scheduler reduce_lr --scheduler-patience 10 --patience 25 --oversample --fold -1 --n-folds 10 --optimize-boundaries
python 3_train.py --exp 8  --method FOROH --dataset limuc --backbone inception_v3 --optimizer adam --lr 2e-4 --lr-head 2e-4 --scheduler reduce_lr --scheduler-patience 10 --patience 25 --oversample --fold -1 --n-folds 10 --freq-weight

## APTOS — EffB3, Adam 1e-4 (single LR), batch 32, Dixit aug/preproc, 5-fold
## 출처: Dixit 2025 (Med Eng Phys, doi:10.1016/j.medengphy.2025.104350)
# Exp 9: Dixit 2025† 원저 보고값 (Kappa 0.88, ACC 88.44%) — 실행 불필요
# Exp 10~16: 우리 실행
python 3_train.py --exp 10 --method ce    --dataset aptos --backbone efficientnet_b3 --optimizer adam --lr 1e-4 --lr-head 1e-4 --batch-size 32 --augment-preset dixit --img-size 256 --fold -1 --n-folds 5
python 3_train.py --exp 11 --method coral --dataset aptos --backbone efficientnet_b3 --optimizer adam --lr 1e-4 --lr-head 1e-4 --batch-size 32 --augment-preset dixit --img-size 256 --fold -1 --n-folds 5
python 3_train.py --exp 12 --method corn  --dataset aptos --backbone efficientnet_b3 --optimizer adam --lr 1e-4 --lr-head 1e-4 --batch-size 32 --augment-preset dixit --img-size 256 --fold -1 --n-folds 5
python 3_train.py --exp 13 --method mse   --dataset aptos --backbone efficientnet_b3 --optimizer adam --lr 1e-4 --lr-head 1e-4 --batch-size 32 --augment-preset dixit --img-size 256 --fold -1 --n-folds 5
python 3_train.py --exp 14 --method FOROH --dataset aptos --backbone efficientnet_b3 --optimizer adam --lr 1e-4 --lr-head 1e-4 --batch-size 32 --augment-preset dixit --img-size 256 --fold -1 --n-folds 5
python 3_train.py --exp 15 --method FOROH --dataset aptos --backbone efficientnet_b3 --optimizer adam --lr 1e-4 --lr-head 1e-4 --batch-size 32 --augment-preset dixit --img-size 256 --fold -1 --n-folds 5 --optimize-boundaries
python 3_train.py --exp 16 --method FOROH --dataset aptos --backbone efficientnet_b3 --optimizer adam --lr 1e-4 --lr-head 1e-4 --batch-size 32 --augment-preset dixit --img-size 256 --fold -1 --n-folds 5 --freq-weight

## KneeXray — D161, AdamW 1e-4/1e-3, Cosine, default aug, fixed split
## 출처: ORM (MTAP 2022) — 세팅 미명시, backbone만 매칭
# Exp 17: ORM† 원저 보고값 (QWK 0.861) — 실행 불필요
# Exp 18: Chen 2019† 원저 보고값 (ACC 69.7%, MAE 0.344) — 실행 불필요
# Exp 19~24: 우리 실행
python 3_train.py --exp 19 --method ce    --dataset kneexray --backbone densenet161
python 3_train.py --exp 20 --method coral --dataset kneexray --backbone densenet161
python 3_train.py --exp 21 --method mse   --dataset kneexray --backbone densenet161
python 3_train.py --exp 22 --method FOROH --dataset kneexray --backbone densenet161
python 3_train.py --exp 23 --method FOROH --dataset kneexray --backbone densenet161 --optimize-boundaries
python 3_train.py --exp 24 --method FOROH --dataset kneexray --backbone densenet161 --freq-weight