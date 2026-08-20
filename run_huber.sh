#!/bin/bash
# RetinaMNIST: Huber vs Weighted Huber
# G4 recall 개선 여부 확인
set -e
T="python 3_train.py"

echo "=== RetinaMNIST: LOR Huber ==="
$T --method lor --dataset retinamnist --fold 0 --loss-fn huber \
   --output-dir outputs/wh_test/huber

echo ""
echo "=== RetinaMNIST: LOR Weighted Huber ==="
$T --method lor --dataset retinamnist --fold 0 --loss-fn weighted_huber \
   --output-dir outputs/wh_test/weighted

echo ""
echo "=== Comparison ==="
python -c "
import torch, json
results = {}
for name, path in [('Huber', 'outputs/wh_test/huber/lor_retinamnist/fold0.pt'),
                    ('W-Huber', 'outputs/wh_test/weighted/lor_retinamnist/fold0.pt')]:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    results[name] = ckpt['test_metrics']

print(f'')
print(f'{\"\":12s} | {\"Huber\":>8s} | {\"W-Huber\":>8s}')
print(f'{\"-\"*12}-+-{\"-\"*8}-+-{\"-\"*8}')
for k in ['mae','qwk','acc','macro_f1']:
    h = results['Huber'][k]
    w = results['W-Huber'][k]
    best = '*' if (w < h if k=='mae' else w > h) else ' '
    print(f'{k:12s} | {h:8.4f} | {best}{w:7.4f}')
print(f'{\"-\"*12}-+-{\"-\"*8}-+-{\"-\"*8}')
c_max = 4
for i in range(c_max+1):
    k = f'recall_g{i}'
    h = results['Huber'][k]
    w = results['W-Huber'][k]
    best = '*' if w > h else ' '
    print(f'{k:12s} | {h:8.3f} | {best}{w:7.3f}')
"