"""Routed attention critics for MAAC (C-MAAC, Z-MAAC) that scale to thousands of agents."""
import os as _os

__version__ = "0.1.0"


def _limit_cpu_threads():
    """Use one CPU thread for PyTorch unless OMP_NUM_THREADS is set.

    Environments, replay buffer and networks all live on the GPU, so the CPU only runs small host-side ops. With
    PyTorch's default of one thread per core these ran up to 170x slower on a busy WSL2 laptop."""
    if "OMP_NUM_THREADS" in _os.environ:
        return
    import torch
    torch.set_num_threads(1)


_limit_cpu_threads()
