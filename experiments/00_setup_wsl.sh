#!/usr/bin/env bash
# One-time setup inside WSL2 Ubuntu. Run from the project root:  bash experiments/00_setup_wsl.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== GPU visible inside WSL? =="
nvidia-smi || { echo "nvidia-smi failed: install or update the NVIDIA driver on the Windows side (not inside WSL) and run 'wsl --update' in PowerShell."; exit 1; }

echo "== python venv =="
sudo apt-get update -y && sudo apt-get install -y python3-venv python3-pip build-essential tmux
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# PyTorch with CUDA. Pick the index that matches the CUDA version shown by nvidia-smi (top-right) and
# https://pytorch.org/get-started/locally  -- examples:
#   CUDA 12.6 / 12.8 driver :  pip install torch --index-url https://download.pytorch.org/whl/cu126
#   newer drivers           :  pip install torch   (default PyPI wheel)
TORCH_INDEX=${TORCH_INDEX:-""}
if [ -n "$TORCH_INDEX" ]; then pip install torch --index-url "$TORCH_INDEX"; else pip install torch; fi
pip install -r requirements.txt scipy tabulate

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available: check driver / torch wheel / WSL GPU support"
print("device:", torch.cuda.get_device_name(0), "| total mem GB:", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))
x = torch.randn(1024, 1024, device="cuda"); print("matmul ok:", (x @ x).sum().item() != 0)
PY
echo "setup OK. Activate later with:  source .venv/bin/activate"
