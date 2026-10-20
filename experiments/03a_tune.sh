#!/usr/bin/env bash
# Reward scale and learning rate, tuned on FlashMAAC at 64 agents only and then used for every critic (this favours
# the reference). Copy the best setting into config.env (REWARD_SCALE, LR_SPREAD, LR_TAG).
set -euo pipefail
cd "$(dirname "$0")/.."; source experiments/config.env
[ -d .venv ] && source .venv/bin/activate
export PYTHONPATH="$PWD"; mkdir -p runs_tune
TUNE_RS=${TUNE_RS:-"1 3 10 30 100"}; TUNE_LR=${TUNE_LR:-"3e-4 1e-3"}; TUNE_SEEDS=${TUNE_SEEDS:-"1 2"}
for env in $ENVS; do for rs in $TUNE_RS; do for lr in $TUNE_LR; do for seed in $TUNE_SEEDS; do
  name=${env}_rs${rs}_lr${lr}_s${seed}
  [ -f runs_tune/$name/final_eval.json ] && continue
  cmd="python train.py --env $env --n_agents 64 --attn dense_flash --device $DEVICE --n_envs 8 --n_episodes 6000 \
       --batch_size 256 --buffer_size 100000 --reward_scale $rs --pi_lr $lr --q_lr $lr --seed $seed \
       --out_dir runs_tune --run_name $name --log_interval 20 --save_interval 0"
  if [ "$DRY" = 1 ]; then echo "$cmd"; else $cmd > runs_tune/$name.log 2>&1 || echo "failed $name"; fi
done; done; done; done
[ "$DRY" = 1 ] && exit 0
python tune_summary.py runs_tune
