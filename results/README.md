# Results

Everything here was measured on one laptop: RTX PRO 1000 Blackwell laptop GPU (8 GB), Intel Core Ultra 7 265H,
32 GB RAM, WSL2, PyTorch 2.14.1 with CUDA 13.0.

## Benchmark sweeps

`2026-10-15_gpu_sweep/` is the sweep reported in the paper. `bench_gpu.csv` has one row per critic, N and minibatch
(8, 16, 32): median and minimum update time in ms, peak allocated and reserved memory in MB, and a status. Only rows
with status `ok` are used; `spilled` rows went past the card's memory and were paged into system RAM by the Windows
driver. `crossover.md` and `crossover_by_batch.md` are the output of `fit_crossover.py`, `figs/` the output of
`plot_figures.py` and `make_maac_variants_figure.py`.

`2026-10-08_gpu_sweep/` is an earlier sweep on the same laptop with an older driver (minibatch 8 and 16). Z-MAAC's
crossover agrees with the later sweep within 8%, C-MAAC's within 44%. Its `dense_loop` rows were measured before the
loop critic copied MAAC's per-agent extractor loops and are therefore too fast for the original MAAC.

## Training runs

One folder per run, named `<task>_N<agents>_<critic>_s<seed>`, with the files written by `train.py`
(`config.json`, `metrics.csv`, `eval.csv`, `final_eval.json`, `random_eval.json`). Checkpoints and TensorBoard
logs are left out.

- `training/main/`: the runs behind the learning results, 2 tasks × 64, 256 and 1,024 agents × 5 seeds. At 64 and
  256 agents six critics, at 1,024 agents four. Tag at 1,024 agents ran for 4,000 episodes (100k environment steps).
- `training/tag1024_50k_steps/`: tag at 1,024 agents with 2,000 episodes, where FlashMAAC had barely left random
  play. The paper mentions the update speed-ups measured in these runs.
- `training/spread4096_seed1/`: one seed of FlashMAAC and of the no-interaction critic on spread with 4,096 agents.

Keep the folders apart when analysing: `collect_results.py` groups runs by their configuration, not by folder, and
would merge the two tag sets.
