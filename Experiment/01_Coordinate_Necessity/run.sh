#!/bin/bash
# Phase-1 Gate A: Is the angular coordinate necessary?
# Run all three methods on LIMUC / ResNet50 / fold 0.

set -e
cd "$(dirname "$0")/../.."

echo "=== E01A: Matched Euclidean Huber ==="
python Experiment/common/train.py \
  --config Experiment/01_Coordinate_Necessity/configs/E01A_euclidean_huber.yaml

echo "=== E01B: Normalized Cosine ==="
python Experiment/common/train.py \
  --config Experiment/01_Coordinate_Necessity/configs/E01B_normalized_cosine.yaml

echo "=== E01C: FOROH ==="
python Experiment/common/train.py \
  --config Experiment/01_Coordinate_Necessity/configs/E01C_foroh.yaml

echo ""
echo "Gate A complete. Compare results in Result/01_Coordinate_Necessity/"
