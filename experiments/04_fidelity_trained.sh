#!/usr/bin/env bash
# How far each router's attention output and Q-values are from full attention, on real states with a trained
# critic. Needs the seed-1 FlashMAAC checkpoints (model.pt) written by 03_train_grid.sh.
set -euo pipefail
cd "$(dirname "$0")/.."; source experiments/config.env
[ -d .venv ] && source .venv/bin/activate
export PYTHONPATH="$PWD"; mkdir -p $RES $FIGS
for env in $ENVS; do for N in 64 256 1024; do
  ck=$OUT_RUNS/${env}_N${N}_dense_flash_s1/model.pt
  [ -f $ck ] || { echo "skip $env N=$N (no $ck)"; continue; }
  cmd="python benchmark.py --fidelity --ckpt $ck --device $DEVICE --batch 16 --modes random knn cluster centroid lsh block hybrid \
       --budget_scales 0.25 0.5 1 2 4 --csv $RES/fidelity_${env}_N${N}.csv"
  if [ "$DRY" = 1 ]; then echo $cmd; else $cmd; python plot_figures.py --fidelity $RES/fidelity_${env}_N${N}.csv --out $FIGS/fidelity_${env}_N${N}; fi
done; done
