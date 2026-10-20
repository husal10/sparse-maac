#!/usr/bin/env bash
# Training runs behind the learning results: both tasks, every critic in TRAIN_SMALL / TRAIN_LARGE, the same
# hyperparameters for all of them. Finished runs (final_eval.json present) are skipped, so the script can be
# restarted. JOBS=2 runs two trainings at once; TIERS="C" ENVS=tag runs one cell.
set -euo pipefail
cd "$(dirname "$0")/.."; source experiments/config.env
[ -d .venv ] && source .venv/bin/activate
export PYTHONPATH="$PWD"; mkdir -p $OUT_RUNS; JOBFILE=$(mktemp)

COMMON="--device $DEVICE --reward_scale $REWARD_SCALE --critic_hidden $HIDDEN --pol_hidden $HIDDEN \
--heads $HEADS --episode_length 25 --steps_per_update 100 --num_updates 4 --save_interval 0 \
--success_coverage $SUCCESS_COVERAGE"

add() {  # add <env> <N> <attn> <seed> <flags...>; SFX adds a suffix to the run name
  local env=$1 N=$2 attn=$3 seed=$4; shift 4
  local name=${env}_N${N}_${attn}${SFX:-}_s${seed}
  if [ -n "$ONLY_METHODS" ] && [[ " $ONLY_METHODS " != *" ${attn}${SFX:-} "* ]]; then return; fi
  [ -f $OUT_RUNS/$name/final_eval.json ] && return
  local lr=$LR_SPREAD; [ "$env" = tag ] && lr=$LR_TAG
  echo "python train.py --env $env --n_agents $N --attn $attn --seed $seed --out_dir $OUT_RUNS --run_name $name $COMMON --pi_lr $lr --q_lr $lr $* > $OUT_RUNS/$name.log 2>&1" >> $JOBFILE
}

for env in $ENVS; do
  if [[ " $TIERS " == *" A "* ]]; then   # 64 agents
    T="--n_envs 8 --batch_size 256 --buffer_size 100000 --n_episodes 10000 --log_interval 10 --eval_every 50 --eval_episodes 16"
    for s in $SEEDS; do
      for a in $TRAIN_SMALL; do add $env 64 $a $s $T; done
      if [[ " $TRAIN_SMALL " == *" dense_loop "* ]]; then SFX=_typeshared add $env 64 dense_loop $s $T --param_sharing type; fi
    done
  fi
  if [[ " $TIERS " == *" B "* ]]; then   # 256 agents
    T="--n_envs 4 --batch_size 64 --buffer_size 40000 --n_episodes 4000 --log_interval 5 --eval_every 25 --eval_episodes 8"
    for s in $SEEDS; do for a in $TRAIN_SMALL; do add $env 256 $a $s $T; done; done
  fi
  if [[ " $TIERS " == *" C "* ]]; then   # 1,024 agents
    ep=$EPISODES_C_SPREAD; [ "$env" = tag ] && ep=$EPISODES_C_TAG
    T="--n_envs 2 --batch_size 32 --buffer_size 20000 --n_episodes $ep --log_interval 4 --eval_every 20 --eval_episodes 4"
    for s in $SEEDS_BIG; do for a in $TRAIN_LARGE; do add $env 1024 $a $s $T; done; done
  fi
  if [[ " $TIERS " == *" D "* ]]; then   # 4,096 agents: 7-9 hours per run on an 8 GB laptop GPU
    T="--n_envs 1 --batch_size 8 --buffer_size 5000 --n_episodes 800 --log_interval 4 --eval_every 20 --eval_episodes 2"
    for s in $SEEDS_BIG; do for a in $TRAIN_LARGE; do add $env 4096 $a $s $T; done; done
  fi
done

echo "$(wc -l < $JOBFILE) training jobs queued (JOBS=$JOBS)"
if [ "$DRY" = 1 ]; then cat $JOBFILE; exit 0; fi
cat $JOBFILE | xargs -P $JOBS -I{} bash -c "{}"
echo "done. Logs: $OUT_RUNS/*.log"
