"""Test settings.

Compile in-process: a compile worker pool starts several fresh Python processes, which is slow under WSL and can
hang at exit. One CPU thread per process, since many threads made small CPU ops far slower on WSL2. Two harmless
warnings are silenced: flex_attention run eagerly on CPU (the CPU tests use the reference path on purpose) and a
Triton DeprecationWarning under Python 3.14.
"""
import os

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def pytest_configure(config):
    config.addinivalue_line("filterwarnings", "ignore:flex_attention called without torch.compile:UserWarning")
    config.addinivalue_line("filterwarnings", "ignore:AnnAssign.__init__ missing:DeprecationWarning")
