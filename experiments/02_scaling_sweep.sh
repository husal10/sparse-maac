#!/usr/bin/env bash
# Critic update time and peak memory against N for every critic, including both MAAC loops. Each (critic, N)
# runs in a fresh process. Rows get a status (ok, spilled, oom, failed, skipped); "spilled" means the run went past
# the card's dedicated memory and Windows paged it into system RAM, so its time is meaningless and it is left out
# of the crossover table and the figures. To get clean out-of-memory errors instead, set "CUDA - Sysmem Fallback
# Policy" to "Prefer No Sysmem Fallback" in the NVIDIA Control Panel. VRAM_MB overrides the detected capacity
# (MB = 1e6 bytes; an 8 GiB card is 8589). Output: results/bench_gpu.csv and results/crossover*.md.
set -euo pipefail
cd "$(dirname "$0")/.."; source experiments/config.env
[ -d .venv ] && source .venv/bin/activate
export PYTHONPATH="$PWD"; mkdir -p $RES
VR=${VRAM_MB:+--vram_mb $VRAM_MB}
C="--device $DEVICE --batches $BENCH_BATCHES --iters $BENCH_ITERS --warmup $BENCH_WARMUP --hidden $HIDDEN --heads $HEADS $VR"
run() { if [ "$DRY" = 1 ]; then echo python benchmark.py "$@"; else python benchmark.py "$@"; fi; }
LOOP_NS=$(for n in $BENCH_NS; do if [ $n -le $LOOP_MAX_N ]; then echo -n "$n "; fi; done)   # same grid as everyone else

# ONLY_LOOP=1 re-measures only the loop critics (3), ONLY_FUSED=1 only the fused backends (1b); both merge with
# the CSVs already there.
if [ "${ONLY_LOOP:-0}" != 1 ] && [ "${ONLY_FUSED:-0}" != 1 ]; then
# (1) FlashMAAC and every router on the common grid BENCH_NS
run --modes dense_flash block centroid knn meanfield hybrid random cluster lsh --ns $BENCH_NS $C --csv $RES/bench_main.csv
# (2) 8,192 agents, only for the critics that fit
[ -n "$BENCH_NS_BIG" ] && run --modes dense_flash block centroid --ns $BENCH_NS_BIG $C --csv $RES/bench_8k.csv
fi
# (1b) the same routers with fused kernels: +ebag sums the attended values with one F.embedding_bag call,
#      +flex runs Z-MAAC as one FlexAttention call with a static block-sparse mask
if [ "${ONLY_LOOP:-0}" != 1 ]; then
run --modes $BENCH_FUSED --ns $BENCH_NS $C --csv $RES/bench_fused.csv
[ -n "$BENCH_NS_BIG" ] && run --modes block+flex --ns $BENCH_NS_BIG $C --csv $RES/bench_fused_8k.csv
fi
[ "${ONLY_FUSED:-0}" = 1 ] && SKIP_LOOP=1
if [ "${SKIP_LOOP:-0}" != 1 ]; then
# (3) MAAC's loop critic with per-agent networks and with shared networks, same grid up to LOOP_MAX_N
run --modes dense_loop --ns $LOOP_NS --loop_max_n $LOOP_MAX_N --loop_sharing agent --device $DEVICE --batches $BENCH_BATCHES --iters 5 --warmup 2 $VR --csv $RES/bench_loop_agent.csv
run --modes dense_loop --ns $LOOP_NS --loop_max_n $LOOP_MAX_N --loop_sharing type  --device $DEVICE --batches $BENCH_BATCHES --iters 5 --warmup 2 $VR --csv $RES/bench_loop_shared.csv
fi

[ "$DRY" = 1 ] && exit 0
python - <<PY
import pandas as pd, glob
fs = ["$RES/bench_main.csv", "$RES/bench_8k.csv", "$RES/bench_fused.csv", "$RES/bench_fused_8k.csv",
      "$RES/bench_loop_agent.csv", "$RES/bench_loop_shared.csv"]
parts = []
for f in fs:
    if not __import__("os").path.exists(f):
        continue
    d = pd.read_csv(f)
    parts.append(d)
df = pd.concat(parts, ignore_index=True)
df.to_csv("$RES/bench_gpu.csv", index=False)
for b, d in df.groupby("batch"):
    print(f"--- status, batch {b} ---")
    print(d.pivot_table(index="mode", columns="N", values="status", aggfunc="first").fillna(""))
PY
python fit_crossover.py $RES/bench_gpu.csv --ref dense_flash --fit_min_n 512 --out $RES/crossover $VR
