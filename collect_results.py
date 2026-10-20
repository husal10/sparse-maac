"""Load finished training runs and score them against FlashMAAC.

    python collect_results.py --runs results/training/main --out results/runs.csv

Each run folder holds config.json, metrics.csv, eval.csv, final_eval.json and random_eval.json (written by
train.py). Returns are normalized per task and number of agents so that the uniformly random policy scores 0 and
the mean of the reference critic (dense_flash) scores 1. In tag the reported return is that of the trained
predators against random prey: in self-play both teams improve together, which hides how good either one is.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def ci95(x):
    """Half-width of the 95% t-interval of the mean."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def welch_ci(a, b, level=95):
    """Difference of means a - b with a Welch t interval (unequal variances)."""
    a, b = (np.asarray(v, float)[np.isfinite(v)] for v in (a, b))
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    diff = float(a.mean() - b.mean())
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    if se < 1e-12:
        return diff, diff, diff
    df = (va + vb) ** 2 / ((va ** 2 / (len(a) - 1) if va > 0 else 0) + (vb ** 2 / (len(b) - 1) if vb > 0 else 0))
    h = float(stats.t.ppf(0.5 + level / 200, df)) * se
    return diff, diff - h, diff + h


def ret_key(env):
    return "vr_return_team0" if env == "tag" else "return_mean"


def load_runs(root):
    rows = []
    for fe in sorted(Path(root).glob("*/final_eval.json")):
        run = fe.parent
        cfg = json.load(open(run / "config.json"))
        sharing = cfg.get("param_sharing", "auto")
        if sharing == "auto":
            sharing = "agent" if cfg["attn"] == "dense_loop" else "type"
        m = pd.read_csv(run / "metrics.csv")
        upd = m["update_ms"][m["update_ms"] > 0]
        ev = pd.read_csv(run / "eval.csv") if (run / "eval.csv").exists() else None
        rnd = json.load(open(run / "random_eval.json")) if (run / "random_eval.json").exists() else None
        rows.append({"run": run.name, "env": cfg["env"], "N": cfg["n_agents"], "attn": cfg["attn"], "sharing": sharing,
                     "seed": cfg["seed"], "final": json.load(open(fe)), "random": rnd, "curve": ev,
                     # second half of the run: the first updates include compilation and warm-up
                     "update_ms": upd.iloc[len(upd) // 2:].mean() if len(upd) else float("nan"),
                     "peak_mem_mb": m["peak_mem_mb"].max(), "wall_s": m["elapsed_s"].iloc[-1]})
    return rows


def method_name(r):
    return r["attn"] + ("_typeshared" if r["attn"] == "dense_loop" and r["sharing"] == "type" else "")


def convergence_point(x, y, t=None, frac=0.9, window=3):
    """First evaluation at which the curve, smoothed over `window` evaluations, reaches `frac` of the run's own
    final level (counted from the random policy at 0). Returns (env steps, seconds); NaN if the run never improves."""
    y = np.asarray(y, float)
    if len(y) < window + 1:
        return float("nan"), float("nan")
    final = float(np.mean(y[-window:]))
    if not np.isfinite(final) or final <= 0:
        return float("nan"), float("nan")
    sm = np.convolve(y, np.ones(window) / window, mode="full")[:len(y)]
    sm[:window - 1] = [y[:i + 1].mean() for i in range(window - 1)]
    idx = np.nonzero(sm >= frac * final)[0]
    if not len(idx):
        return float("nan"), float("nan")
    i = int(idx[0])
    return float(x[i]), (float(t[i]) if t is not None else float("nan"))


def score_runs(rows, ref="dense_flash"):
    """One row per run with raw and normalized final return, task success, steps to converge and cost."""
    out = []
    groups = {}
    for r in rows:
        groups.setdefault((r["env"], r["N"]), []).append(r)
    for (env, N), rs in groups.items():
        k = ret_key(env)
        R_ref = float(np.mean([r["final"][k] for r in rs if method_name(r) == ref] or [np.nan]))
        R_rnd = float(np.mean([r["random"][k] for r in rs if r["random"]] or [np.nan]))
        span = R_ref - R_rnd
        norm = (lambda v: (v - R_rnd) / span) if np.isfinite(span) and abs(span) > 1e-9 else (lambda v: float("nan"))
        for r in rs:
            f = r["final"]
            row = {"env": env, "N": N, "method": method_name(r), "seed": r["seed"], "run": r["run"],
                   "return": f[k], "return_norm": norm(f[k]),
                   # spread: share of landmarks covered at the last step; tag: share of prey caught at least once
                   "task_success": f.get("final_coverage" if env == "spread" else "vr_capture_rate", float("nan")),
                   "update_ms": r["update_ms"], "peak_mem_mb": r["peak_mem_mb"], "wall_s": r["wall_s"],
                   "R_ref": R_ref, "R_random": R_rnd}
            c = r["curve"]
            if c is not None and len(c) and k in c:
                x = np.concatenate([[0.0], c["env_steps"].to_numpy(float)])
                t = np.concatenate([[0.0], c["elapsed_s"].to_numpy(float)]) if "elapsed_s" in c else None
                y = np.array([0.0] + [norm(v) for v in c[k]])
                row["steps_to_conv"], row["time_to_conv"] = convergence_point(x, y, t)
            out.append(row)
    return pd.DataFrame(out)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results/runs.csv")
    a = p.parse_args(argv)
    rows = load_runs(a.runs)
    if not rows:
        print("no finished runs in", a.runs)
        return None
    df = score_runs(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} runs -> {a.out}")
    return df


if __name__ == "__main__":
    main()
