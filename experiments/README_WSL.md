# Running the experiments on WSL2 (Ubuntu, NVIDIA GPU)

All results in the paper come from one laptop: NVIDIA RTX PRO 1000 Blackwell laptop GPU (8 GB, driver 597.13),
Intel Core Ultra 7 265H, 32 GB RAM, Windows with WSL2, PyTorch 2.14.1, CUDA 13.0.

## Windows side (once)

1. WSL2 with an Ubuntu distribution: `wsl --install -d Ubuntu`, then `wsl --update` in PowerShell.
2. Install the NVIDIA driver on Windows only. Do not install a Linux driver inside WSL; WSL uses the Windows driver,
   and the PyTorch wheels bring their own CUDA libraries.
3. `nvidia-smi` inside Ubuntu must list the GPU. If it does not, update the Windows driver and run `wsl --shutdown`.
4. Optional: give WSL more memory in `C:\Users\<you>\.wslconfig`, then `wsl --shutdown`:
   ```
   [wsl2]
   memory=24GB
   swap=16GB
   ```
5. For timing runs, keep the laptop plugged in, use the "Best performance" power mode and close other GPU programs.
   Clocks cannot be locked from WSL, so the benchmark reports medians over 30 updates.

## Setup

Keep the project on the Linux file system (`~`), not under `/mnt/c`: every Python start and every log write is
several times slower through the Windows file bridge. The scripts print a warning when run from a Windows drive.

```bash
cd ~/sparse-maac
bash experiments/00_setup_wsl.sh      # venv, PyTorch with CUDA, dependencies, GPU check
# if the default wheel does not match the driver, choose an index on pytorch.org, e.g.
#   TORCH_INDEX=https://download.pytorch.org/whl/cu126 bash experiments/00_setup_wsl.sh
source .venv/bin/activate
bash experiments/01_gpu_smoke.sh      # tests and a tiny training run per critic, about 10 minutes
```

PyTorch uses one CPU thread per core by default. Under WSL2 that made the small host-side operations up to 170 times
slower, so the package and the scripts use one thread (`CPU_THREADS` in `config.env`, or `OMP_NUM_THREADS`).
Training runs on the GPU, so this costs nothing.

## Run order

Use `tmux` so runs survive a closed terminal (`tmux new -s exp`, detach with `Ctrl-b d`, back with `tmux attach -t exp`).

| step | command | output | time on the laptop |
|---|---|---|---|
| 1 | `bash experiments/02_scaling_sweep.sh` | `results/bench_gpu.csv`, crossover tables | 1-2 h |
| 2 | `bash experiments/03a_tune.sh` | tuning runs in `runs_tune/`; copy the best setting into `config.env` | 2-4 h |
| 3 | `bash experiments/03_train_grid.sh` | training runs in `runs/` | about 4 days |
| 4 | `bash experiments/04_fidelity_trained.sh` | approximation error of each router on real states | 10-20 min |
| 5 | `bash experiments/05_analyze.sh` | `results/summary.md`, figures in `figs/` | minutes |

The training grid is restartable: finished runs are skipped. To start smaller, e.g. `SEEDS="1 2 3" TIERS="A B"`.
At 1,024 agents a spread run takes about 70 minutes and a tag run about 50 minutes; one run at 4,096 agents
(`TIERS="D"`) takes 7 to 9 hours, mostly in the environment.

The benchmark times every critic at each minibatch size in `BENCH_BATCHES` (default `8 16 32`). Use
`BENCH_BATCHES="8 32"` to shorten it.

## Runs that do not fit in GPU memory

On Windows the driver does not raise an out-of-memory error by default. It pages GPU memory into system RAM, the run
finishes, and its time means nothing. `benchmark.py` marks such rows `spilled` (peak reserved memory above the
card's dedicated memory), and `fit_crossover.py` and `plot_figures.py` leave them out. For clean errors instead, set
NVIDIA Control Panel, Manage 3D settings, "CUDA - Sysmem Fallback Policy" to "Prefer No Sysmem Fallback".
Pass `VRAM_MB` (MB = 1e6 bytes; 8 GiB = 8589) if the detected capacity is wrong.
