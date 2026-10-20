#!/usr/bin/env bash
# Quick check (5-10 minutes) before the long runs: unit tests, one tiny training run per critic on both tasks,
# with and without bf16 autocast (--amp), and a short benchmark.
set -uo pipefail
cd "$(dirname "$0")/.."; source experiments/config.env
[ -d .venv ] && source .venv/bin/activate
export PYTHONPATH="$PWD"

python -m pytest tests -q -x || { echo "UNIT TESTS FAILED"; exit 1; }
fail=0
for env in $ENVS; do
  for attn in dense_loop dense_flash cluster random centroid knn meanfield lsh hybrid block; do
    for amp in "" "--amp"; do
      [ -n "$amp" ] && [[ "$attn" == dense_loop || "$attn" == cluster || "$attn" == lsh || "$attn" == random ]] && continue
      if python train.py --env $env --n_agents 64 --attn $attn $amp --device $DEVICE --n_envs 4 --n_episodes 24 \
           --batch_size 64 --steps_per_update 20 --num_updates 1 --buffer_size 2000 --eval_episodes 4 --log_interval 2 \
           --save_interval 0 --out_dir /tmp/smoke_runs --run_name smoke_${env}_${attn}${amp// /} > /tmp/smoke_${env}_${attn}${amp// /}.log 2>&1
      then echo "ok    $env $attn $amp"; else echo "FAIL  $env $attn $amp  (see /tmp/smoke_${env}_${attn}${amp// /}.log)"; fail=1; fi
    done
  done
done
# quick GPU benchmark sanity (also prints peak memory)
python benchmark.py --modes dense_flash block centroid --ns 256 1024 --batch 8 --iters 5 --warmup 2
[ $fail -eq 0 ] && echo "ALL SMOKE CHECKS PASSED" || { echo "SOME CHECKS FAILED (an --amp failure only means: train without --amp)"; }
