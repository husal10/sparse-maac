#!/usr/bin/env bash
# Tables and figures from the benchmark and the training runs.
#   bash experiments/05_analyze.sh                                   # your own runs/ and results/bench_gpu.csv
#   OUT_RUNS=results/training/main BENCH=results/2026-10-15_gpu_sweep/bench_gpu.csv bash experiments/05_analyze.sh
set -euo pipefail
cd "$(dirname "$0")/.."; source experiments/config.env
[ -d .venv ] && source .venv/bin/activate
export PYTHONPATH="$PWD"; mkdir -p $RES $FIGS
BENCH=${BENCH:-$RES/bench_gpu.csv}

if [ -f "$BENCH" ]; then
  python plot_figures.py --bench $BENCH --out $FIGS ${VRAM_MB:+--vram_mb $VRAM_MB} \
    --modes dense_loop dense_loop_shared dense_flash centroid block hybrid knn random cluster lsh meanfield
  python make_maac_variants_figure.py --bench $BENCH --out $FIGS/maac_variants --column
fi
python summary_table.py --runs $OUT_RUNS --out $RES/summary          # learning table (results/summary.md)
python plot_paper_results.py --runs $OUT_RUNS --out $FIGS            # learning curves and quality/cost figure
for env in $ENVS; do for N in 64 256 1024 4096; do                   # one learning-curve figure per task and size
  specs=()
  for m in dense_loop dense_loop_typeshared dense_flash none block centroid knn meanfield; do
    ls -d $OUT_RUNS/${env}_N${N}_${m}_s* >/dev/null 2>&1 && specs+=("$m=$OUT_RUNS/${env}_N${N}_${m}_s*")
  done
  [ ${#specs[@]} -gt 0 ] && python plot_figures.py --curves "${specs[@]}" --out $FIGS/curves_${env}_N${N} --tag "$env, N=$N"
done; done
echo "figures in $FIGS/, tables in $RES/"
