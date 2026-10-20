"""Time, memory and approximation error of each critic.

    python benchmark.py --modes dense_loop dense_flash centroid block --ns 64 256 1024 --batches 8 16 32 \
                        --csv results/bench.csv
    python benchmark.py --fidelity --modes knn centroid block --ns 512 --budget_scales 0.5 1 2 4 \
                        --ckpt runs/x/model.pt --csv results/fidelity.csv

Every (critic, N, batch) runs in its own subprocess so that peak memory is measured cleanly. The timed unit is one
critic update: target forward, forward and backward, Adam step. The policy update is the same for every critic and
is not timed. Each row gets a status: ok, spilled, oom, failed or skipped.

On Windows the driver does not fail when GPU memory runs out; it pages into system RAM and the run just gets slow.
Such rows (peak reserved memory above the card's memory, override with --vram_mb) are marked `spilled` and stopped
after one step, and larger N for the same critic and batch are skipped.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import subprocess
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sparse_maac.agent import SparseMAAC
from sparse_maac.config import ATTN_MODES, parse
from sparse_maac.networks import AttentionCritic
from sparse_maac.routing import make_router
from sparse_maac.runner import make_env

BACKENDS = {"ebag": ("sparse_agg", "ebag"), "flex": ("block_impl", "flex")}


def split_mode(mode):
    """'knn+ebag' -> ('knn', {'sparse_agg': 'ebag'}); 'block+flex' -> ('block', {'block_impl': 'flex'})."""
    base, *opts = mode.split("+")
    if base not in ATTN_MODES or any(o not in BACKENDS for o in opts):
        raise ValueError(f"unknown mode {mode!r}: use one of {ATTN_MODES}, optionally +ebag (index routers) or +flex (block)")
    return base, dict(BACKENDS[o] for o in opts)


def mode_type(s):
    split_mode(s)
    return s


BUDGET_KEYS = {"random": "rand_k", "knn": "knn_k", "meanfield": "knn_k", "cluster": "n_clusters", "centroid": "n_clusters",
               "lsh": "lsh_bucket", "block": "block_size"}


def make_cfg(mode, N, a, scale=1.0):
    mode, backend = split_mode(mode)
    c = parse([f"--attn={mode}", f"--n_agents={N}"])
    for k, val in backend.items():
        setattr(c, k, val)
    for k in ("knn_k", "rand_k", "n_clusters", "lsh_bucket", "block_size", "lsh_rounds", "knn_backend",
              "anneal_n0", "anneal_n1"):
        if getattr(a, k, None) is not None:
            setattr(c, k, getattr(a, k))
    if scale != 1.0:
        s = math.sqrt(N) if mode != "block" else math.sqrt(N / 3)
        if mode == "hybrid":
            for k in ("knn_k", "rand_k", "n_clusters"):
                setattr(c, k, max(1, round(s * scale)))
        elif mode in BUDGET_KEYS:
            setattr(c, BUDGET_KEYS[mode], max(1, round(s * scale)))
    return c


def build_critic(mode, N, a, dev, scale=1.0, D=19, A=5, ranges=None, use_bn=True):
    c = make_cfg(mode, N, a, scale)
    mode = c.attn
    if ranges is None:
        per_agent = mode == "dense_loop" and a.loop_sharing == "agent"
        ranges = [(i, i + 1) for i in range(N)] if per_agent else [(0, N)]
    router = None if mode in ("dense_loop", "dense_flash", "none") else make_router(mode, N, c)
    return AttentionCritic(D, A, ranges, mode, a.hidden, a.heads, router, use_bn=use_bn,
                           sparse_agg=c.sparse_agg, block_impl=c.block_impl).to(dev)


def synth(B, N, dev, D=19, A=5, seed=0, smooth=False):
    """Synthetic states. smooth=True makes features a smooth function of position (+noise), i.e. spatially
    correlated like a real particle world; smooth=False is i.i.d. (timing only)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    obs = torch.randn(B, N, D, generator=g)
    pos = torch.rand(B, N, 2, generator=g) * 2 - 1
    if smooth:
        W = torch.randn(2, D, generator=g) * 2.0
        obs = torch.tanh(pos @ W) + 0.05 * obs
    obs[..., 2:4] = pos
    acts = F.one_hot(torch.randint(0, A, (B, N), generator=g), A).float()
    return obs.to(dev), acts.to(dev)


def _rss_kb(field):
    for line in open("/proc/self/status"):
        if line.startswith(field):
            return int(line.split()[1])
    return 0


def run_single(mode, N, a, dev):
    crit = build_critic(mode, N, a, dev).train()
    tgt = copy.deepcopy(crit)
    opt = torch.optim.Adam(crit.parameters(), 1e-3)
    obs, acts = synth(a.batch, N, dev)

    def step():
        with torch.no_grad():
            nq, _, _ = tgt(obs, acts, tgt.route(obs))
        q, _, _ = crit(obs, acts, crit.route(obs))
        opt.zero_grad(); F.mse_loss(q, nq).backward(); opt.step()

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    else:
        try:
            open("/proc/self/clear_refs", "w").write("5")
        except OSError:
            pass
    base = _rss_kb("VmRSS")
    cap = (a.vram_mb or torch.cuda.get_device_properties(dev).total_memory / 1e6) if dev.type == "cuda" else None
    for w in range(max(a.warmup, 1)):
        step()
        if w == 0 and cap is not None:          # already spilled: the time would be meaningless and slow
            torch.cuda.synchronize()
            if torch.cuda.max_memory_reserved() / 1e6 > cap:
                nan = float("nan")
                return {"ms": nan, "ms_min": nan, "mem_mb": torch.cuda.max_memory_allocated() / 1e6,
                        "reserved_mb": torch.cuda.max_memory_reserved() / 1e6, "status": "spilled"}
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = []
    for _ in range(a.iters):
        t0 = time.perf_counter(); step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1000)
    status, reserved = "ok", float("nan")
    if dev.type == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1e6
        reserved = torch.cuda.max_memory_reserved() / 1e6
        if reserved > cap:
            status = "spilled"
    else:
        mem = (_rss_kb("VmHWM") - base) / 1e3
    t = sorted(t)
    return {"ms": t[len(t) // 2], "ms_min": t[0], "mem_mb": mem, "reserved_mb": reserved, "status": status}


@torch.no_grad()
def mean_tokens(r, N):
    n = 0.0
    if r.idx is not None:
        n += r.mask.float().sum(-1).mean().item()
    if r.assign is not None:
        n += r.n_clusters
    if r.perm is not None:
        nb = math.ceil(N / r.block)
        n += min(3 * r.block, N) - 1 + (max(nb - 3, 0) if r.use_global else 0)
    return n


@torch.no_grad()
def real_states(ckpt, dev, B):
    """Real (obs, one-hot acts) from the checkpoint's own env + policy, mid-episode. Use a type-shared
    (dense_flash / sparse) checkpoint: per-agent-parameter critics are not loadable into the dense reference."""
    d = torch.load(ckpt, map_location=dev)
    cfg = SimpleNamespace(**d["cfg"])
    env = make_env(cfg, dev, n_envs=B, seed=777)
    agent = SparseMAAC(env.spec, cfg, dev)
    agent.load(ckpt)
    agent.prep_rollouts()
    obs, sel = env.reset(), None
    for t in range(cfg.episode_length):
        ac = agent.act(obs, True)
        if t == cfg.episode_length // 2:
            sel = (obs.clone(), F.one_hot(ac, env.n_actions).float())
        obs = env.step(ac)[0]
    return dict(obs=agent.norm(sel[0]), obs_raw=sel[0], use_bn=cfg.obs_norm == "bn", acts=sel[1], N=env.N, D=env.obs_dim, A=env.n_actions, ranges=env.ranges,
                hidden=cfg.critic_hidden, heads=cfg.heads, state=d["critic"])


@torch.no_grad()
def run_fidelity(mode, N, a, dev, scale, ctx=None):
    if ctx:
        a.hidden, a.heads = ctx["hidden"], ctx["heads"]
        kw = dict(D=ctx["D"], A=ctx["A"], ranges=ctx["ranges"], use_bn=ctx["use_bn"])
        obs, acts = ctx["obs"], ctx["acts"]
    else:
        kw, (obs, acts) = {}, synth(a.batch, N, dev, smooth=True)
    dense = build_critic("dense_flash", N, a, dev, **kw).eval()
    if ctx:
        dense.load_state_dict(ctx["state"])
    elif a.sharpen != 1.0:   # stress test: emulate trained, peaked attention with random weights
        with torch.no_grad():
            dense.W_q.weight.mul_(a.sharpen)
    sp = build_critic(mode, N, a, dev, scale, **kw).eval()
    sp.load_state_dict(dense.state_dict())
    _, xd, _ = dense.attend(obs, acts)
    r = sp.route(ctx["obs_raw"] if ctx else obs)   # route on raw positions, attend on normalized observations
    _, xs, _ = sp.attend(obs, acts, r)
    qd, qs = dense(obs, acts)[1], sp(obs, acts, r)[1]
    rel = lambda x, y: ((x - y).norm() / y.norm().clamp(min=1e-9)).item()
    return {"tokens": mean_tokens(r, N), "rel_err_x": rel(xs, xd), "rel_err_q": rel(qs, qd)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modes", nargs="+", default=["dense_flash", "block", "hybrid"], type=mode_type,
                   help=f"{ATTN_MODES}; append +ebag (index routers) or +flex (block) for the fused backends, e.g. knn+ebag")
    p.add_argument("--ns", nargs="+", type=int, default=[64, 256, 1024])
    p.add_argument("--batch", type=int, default=8, help="batch for --fidelity (and for timing if --batches is not given)")
    p.add_argument("--batches", nargs="+", type=int, default=None,
                   help="timing: sweep these batch sizes (rows carry a `batch` column), e.g. --batches 8 16 32")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--loop_max_n", type=int, default=128)
    for k in ("knn_k", "rand_k", "n_clusters", "lsh_bucket", "block_size", "anneal_n0", "anneal_n1"):
        p.add_argument(f"--{k}", type=int, default=None)
    p.add_argument("--lsh_rounds", type=int, default=None)
    p.add_argument("--knn_backend", default=None)
    p.add_argument("--fidelity", action="store_true")
    p.add_argument("--budget_scales", nargs="+", type=float, default=[1.0])
    p.add_argument("--ckpt", default=None, help="fidelity on REAL env states from this (type-shared) checkpoint")
    p.add_argument("--loop_sharing", default="agent", choices=["agent", "type"],
                   help="dense_loop parameter sharing: agent = original MAAC, type = matched ablation")
    p.add_argument("--sharpen", type=float, default=1.0, help="fidelity w/o ckpt: scale W_q to peak the attention")
    p.add_argument("--vram_mb", type=float, default=None,
                   help="dedicated GPU memory in MB for the spill check (default: detected total_memory)")
    p.add_argument("--device", default="auto")
    p.add_argument("--csv", default=None)
    p.add_argument("--single", nargs=2, metavar=("MODE", "N"), help=argparse.SUPPRESS)
    a = p.parse_args()
    dev = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else
                       ("cpu" if a.device == "auto" else a.device))
    if a.single:
        mode, N = a.single[0], int(a.single[1])
        try:
            print("RESULT " + json.dumps(run_single(mode, N, a, dev)))
        except torch.cuda.OutOfMemoryError:
            print("RESULT " + json.dumps({"ms": float("nan"), "ms_min": float("nan"), "mem_mb": float("nan"),
                                          "reserved_mb": float("nan"), "status": "oom"}))
        return
    rows, ctx = [], None
    if a.fidelity and a.ckpt:
        ctx = real_states(a.ckpt, dev, a.batch)
        a.ns = [ctx["N"]]
    batches = [a.batch] if (a.fidelity or not a.batches) else a.batches
    for B, mode in ((B, m) for B in batches for m in a.modes):
        dead = None                                  # first N at which this (mode, batch) did not fit
        for N in a.ns:
            if mode == "dense_loop" and N > a.loop_max_n:
                continue
            if a.fidelity:
                if mode in ("dense_loop", "dense_flash"):
                    continue
                for sc in (a.budget_scales if split_mode(mode)[0] in BUDGET_KEYS or split_mode(mode)[0] == "hybrid" else [1.0]):
                    r = run_fidelity(mode, N, a, dev, sc, ctx)
                    rows.append({"mode": mode, "N": N, "scale": sc, **r})
                    print(f"{mode:11s} N={N:5d} x{sc:<4} tokens={r['tokens']:7.1f} err_x={r['rel_err_x']:.4f} err_q={r['rel_err_q']:.4f}")
            else:
                label = "dense_loop_shared" if (mode == "dense_loop" and a.loop_sharing == "type") else mode
                nan = float("nan")
                if dead is not None:                 # memory grows with N: a larger N cannot fit either
                    rows.append({"mode": label, "N": N, "batch": B, "ms": nan, "ms_min": nan, "mem_mb": nan,
                                 "reserved_mb": nan, "status": "skipped"})
                    print(f"{mode:11s} N={N:5d} B={B:<3d} skipped (did not fit at N={dead})"); continue
                cmd = [sys.executable, __file__, "--single", mode, str(N), "--device", str(dev), "--batch", str(B),
                       "--heads", str(a.heads), "--hidden", str(a.hidden), "--iters", str(a.iters), "--warmup", str(a.warmup),
                       "--loop_sharing", a.loop_sharing]
                if a.vram_mb:
                    cmd += ["--vram_mb", str(a.vram_mb)]
                for k in ("knn_k", "rand_k", "n_clusters", "lsh_bucket", "block_size", "lsh_rounds", "knn_backend"):
                    if getattr(a, k) is not None:
                        cmd += [f"--{k}", str(getattr(a, k))]
                out = subprocess.run(cmd, capture_output=True, text=True)
                line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
                if not line:   # crashed subprocess (e.g. host OOM-kill): keep the row so the table shows it
                    rows.append({"mode": label, "N": N, "batch": B, "ms": nan, "ms_min": nan, "mem_mb": nan,
                                 "reserved_mb": nan, "status": "failed"})
                    dead = N
                    print(f"{mode:11s} N={N:5d} B={B:<3d} FAILED {out.stderr[-200:]}"); continue
                r = json.loads(line[0][7:]); rows.append({"mode": label, "N": N, "batch": B, **r})
                if r["status"] != "ok":
                    dead = N
                flag = "" if r["status"] == "ok" else f"  [{r['status'].upper()}: does not fit, timing invalid]"
                print(f"{mode:11s} N={N:5d} B={B:<3d} {r['ms']:9.1f} ms/update  {r['mem_mb']:9.1f} MB{flag}")
    if a.csv and rows:
        import os
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        keys = list(rows[0].keys())
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
