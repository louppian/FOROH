#!/usr/bin/env bash
set -euo pipefail

# Reproduce the numerical experiments in paper order.
bash Experiment/01_Main_Comparison/run.sh
bash Experiment/02_Backbone_Scaling/run.sh
bash Experiment/03_Score_Function/run.sh
bash Experiment/04_Ablation/run.sh

printf '\nNumerical reproduction finished.\n'
printf 'Inspect Result/01_Main_Comparison through Result/04_Ablation.\n'
printf 'Run representation/figure analysis only after selecting reproduced checkpoints.\n'
