# Sparse attention critics for MAAC

Code for *Sparse Attention Critics for Thousands of Agents Without Losing the Counterfactual Baseline*.

MAAC (Iqbal & Sha, 2019) gives every agent an attention critic over all other agents. That is quadratic in the
number of agents N, and the released code also loops over agents and heads in Python. This repository has three
versions of the MAAC critic and two routed ones that scale to thousands of agents:

| `--attn` | name | what each agent attends to |
|---|---|---|
| `dense_loop` | MAAC | everyone, with MAAC's original loops (one network per agent) |
| `dense_flash` | FlashMAAC | everyone, in one fused attention call (one network per agent type) |
| `centroid` | **C-MAAC** | one summary per k-means cluster of positions, c = ⌈√N⌉ clusters |
| `block` | Z-MAAC | agents sorted along a Z-order curve and cut into blocks of b = √(N/3): its own and the two neighbouring blocks agent by agent, every other block as one summary |
| `none` | no interaction | nobody: Q_i sees only o_i |
| `meanfield`, `knn`, `random`, `cluster`, `lsh`, `hybrid` | baselines | see `sparse_maac/routing.py` |
| `anneal` | annealed critic | FlashMAAC below a size n0, Z-MAAC above 4·n0, block size shrinking in between (not evaluated in the paper) |

A summary is the mean key and mean value of a group, with log n_g added to its score, so it carries as much weight
as its n_g members would if their keys were equal.

## The counterfactual baseline stays exact

MAAC's critic outputs Q_i(o, ·) for every action of agent i in one pass, and the policy gradient uses the expected
value under π_i as its baseline. That only works if the output does not depend on the action a_i that went into the
critic. All critics here keep three conditions:

1. **Routing ignores actions.** Routers see positions only, detached and without gradients.
2. **No summary contains agent i.** Agent i's own cluster or block is summarized without it.
3. **No batch statistics.** BatchNorm in train mode mixes all agents' inputs, a_i included, into everyone's
   embedding when encoders are shared across agents. Observations are normalized with running statistics instead
   (`--obs_norm running`, the default).

`tests/test_all.py` checks this directly: changing a_i leaves Q_i(o, ·) bit-for-bit unchanged for every critic.

## Install

Python 3.10 or newer and PyTorch 2.4 or newer. A CUDA GPU is needed for the timings; everything else runs on CPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # pick the wheel that matches your driver
pip install -r requirements.txt
pytest -q tests
```

On Windows, use WSL2 and follow `experiments/README_WSL.md` (`experiments/00_setup_wsl.sh` does the setup).

## Quick start

```bash
# train C-MAAC on spread with 256 agents
python train.py --env spread --n_agents 256 --attn centroid --n_envs 4 --batch_size 64 \
    --n_episodes 4000 --eval_every 25 --out_dir runs

# time one critic update for several critics and sizes
python benchmark.py --modes dense_flash centroid block --ns 256 1024 4096 --batches 8 32 --csv results/bench.csv
python fit_crossover.py results/bench.csv --ref dense_flash --out results/crossover
```

`python train.py --help` lists all options. Each run writes `config.json`, `metrics.csv` (training), `eval.csv`
(greedy evaluations), `final_eval.json`, `random_eval.json` and `model.pt` into `runs/<run_name>/`.

## Tasks

Both tasks are batched particle worlds in plain PyTorch (`sparse_maac/envs/particle.py`). The arena grows with N
so the density stays constant, and observations have a fixed size.

- **spread** (cooperative): N agents and N landmarks. Each agent is scored on how well its k nearest landmarks are
  covered, by anyone, minus collisions. Task success is the share of landmarks covered at the end of an episode.
- **tag** (competitive): two thirds of the agents are predators, one third faster prey. Returns and the capture rate
  are measured for trained predators against random prey, because in self-play trained prey can make trained
  predators look worse than random ones.

## Reproducing the paper

All numbers come from one laptop with an 8 GB RTX PRO 1000 GPU under WSL2. The scripts in `experiments/` run in
order; `experiments/README_WSL.md` gives the times.

```bash
bash experiments/02_scaling_sweep.sh     # cost of every critic against N, minibatch 8, 16 and 32
bash experiments/03a_tune.sh             # reward scale and learning rate, on FlashMAAC at 64 agents
bash experiments/03_train_grid.sh        # 2 tasks x 64/256/1,024 agents x critics x 5 seeds
bash experiments/05_analyze.sh           # tables and figures
```

The measured results are in `results/`, so the tables and figures can be rebuilt without training:

```bash
OUT_RUNS=results/training/main BENCH=results/2026-10-15_gpu_sweep/bench_gpu.csv bash experiments/05_analyze.sh
```

Main numbers (critic update, FlashMAAC divided by the routed critic):

| minibatch | C-MAAC crossover | Z-MAAC crossover | C-MAAC speed-up at 4,096 / 8,192 agents | Z-MAAC speed-up at 4,096 / 8,192 |
|---|---|---|---|---|
| 8 | 1,065 | 895 | 9.5× / 13.0× | 6.4× / 9.2× |
| 32 | 559 | 1,041 | 8.0× / 12.2× | 4.7× / 9.4× |

FlashMAAC alone is 697× faster than MAAC's loops at 512 agents (minibatch 8). In training, C-MAAC and Z-MAAC reach
0.92 to 1.04 of FlashMAAC's normalized return across both tasks and all sizes, and their task success is within
0.01 of it.

## Layout

```
sparse_maac/          the package: environments, routers, attention backends, networks, MAAC update
train.py              training
evaluate.py           roll out a checkpoint, render a GIF, play two checkpoints against each other in tag
benchmark.py          time and memory per critic update, and approximation error against full attention
fit_crossover.py      crossover sizes and scaling exponents from a benchmark CSV
collect_results.py    load finished runs and normalize their returns
summary_table.py      learning table (results/summary.md)
plot_*.py, make_*.py  figures
tune_summary.py, tag_check.py   tuning helpers
experiments/          the scripts behind the paper, with their settings in config.env
results/              measured benchmark sweeps and the logs of every training run (see results/README.md)
tests/                unit tests
```

## Citation

The citation will be added after review.
