#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

python Experiment/00_Inventory/inventory.py --root "$ROOT"
python Experiment/00_Inventory/match_paper.py --root "$ROOT"
python Experiment/00_Inventory/verify_checkpoints.py --root "$ROOT"

if [[ "${1:-}" == "--reevaluate" ]]; then
  python Experiment/00_Inventory/verify_checkpoints.py \
    --root "$ROOT" \
    --reevaluate \
    --num-workers "${NUM_WORKERS:-4}"
fi

echo
echo "Reports:"
echo "  Result/00_Inventory/inventory.json"
echo "  Result/00_Inventory/inventory.csv"
echo "  Result/00_Inventory/paper_match.json"
echo "  Result/00_Inventory/checkpoint_verification_structural.json"
if [[ "${1:-}" == "--reevaluate" ]]; then
  echo "  Result/00_Inventory/checkpoint_verification_reeval.json"
fi
