#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python Experiment/02_LevelSet_Necessity/run.py
