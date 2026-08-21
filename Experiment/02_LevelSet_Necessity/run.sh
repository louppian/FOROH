#!/bin/bash
# Phase-1 Gate B: Does the level-set representation matter?
# Run point prototype vs FOROH on LIMUC / ResNet50 / fold 0.

set -e
cd "$(dirname "$0")/../.."

echo "=== E02A: Point Prototype ==="
python Experiment/common/train.py \
  --config Experiment/02_LevelSet_Necessity/configs/E02A_point_prototype.yaml

echo "=== E02B: FOROH Level Set ==="
python Experiment/common/train.py \
  --config Experiment/02_LevelSet_Necessity/configs/E02B_foroh_levelset.yaml

echo ""
echo "Gate B complete. Compare results in Result/02_LevelSet_Necessity/"
