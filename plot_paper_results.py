"""Learning figures: curves at each number of agents, then what routing costs in quality and saves in time.

    python plot_paper_results.py --runs results/training/main --out figs

fig_learning      greedy evaluation return against environment steps, one panel per task and number of agents, mean
                  and 95% band over seeds; the dashed line is the random policy (tag: trained predators against
                  random prey).
fig_scaling_cost  against the number of agents: (a) normalized return (random 0, FlashMAAC 1), (b) task success
                  (spread: landmarks covered; tag: prey caught), (c) critic-update speed-up over FlashMAAC measured
                  inside the same training runs. Error bars are 95% t-intervals over seeds.
Steps to converge are written to scaling_cost.csv but not plotted, since several runs end before converging.
Only finished runs (final_eval.json present) are used. PDF and 300-dpi PNG.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collect_results import load_runs, method_name, ret_key, score_runs
from plot_figures import DBL, LABEL, MARK, OKABE, save, smooth, style
from matplotlib.ticker import FuncFormatter, NullFormatter

ORDER = ["dense_flash", "centroid", "block", "knn", "meanfield", "none"]
ENVS = {"spread": "Spread (cooperative)", "tag": "Tag (competitive)"}
TSTAR = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (float(v.mean()) if len(v) else float("nan")), float("nan")
    return float(v.mean()), TSTAR.get(len(v), 1.96) * float(v.std(ddof=1)) / np.sqrt(len(v))


def _order(methods):
    return [m for m in ORDER if m in methods] + sorted(set(methods) - set(ORDER))


def fig_learning(rows, out, tag=None, smooth_k=3):
    envs = [e for e in ENVS if any(r["env"] == e for r in rows)]
    Ns = sorted({r["N"] for r in rows})
    if not envs or not Ns:
        return
    fig, axes = plt.subplots(len(envs), len(Ns), figsize=(DBL, 1.75 * len(envs) + 0.35), squeeze=False)
    for i, env in enumerate(envs):
        k = ret_key(env)
        for j, N in enumerate(Ns):
            ax = axes[i, j]
            rs = [r for r in rows if r["env"] == env and r["N"] == N and r["curve"] is not None and k in r["curve"]]
            for m in _order({method_name(r) for r in rs}):
                cs = [r["curve"] for r in rs if method_name(r) == m]
                L = min(len(c) for c in cs)
                if L < 2:
                    continue
                x = cs[0]["env_steps"].to_numpy(float)[:L] / 1e3
                Y = np.stack([smooth(c[k].to_numpy(float)[:L], smooth_k) for c in cs])
                mu = Y.mean(0)
                h = (TSTAR.get(len(Y), 1.96) * Y.std(0, ddof=1) / np.sqrt(len(Y))) if len(Y) > 1 else np.zeros_like(mu)
                ax.plot(x, mu, color=OKABE.get(m), lw=2.2 if m == "dense_flash" else 1.1, label=LABEL.get(m, m),
                        ls=":" if m == "none" else "-", zorder=1.5 if m == "dense_flash" else 2)
                ax.fill_between(x, mu - h, mu + h, color=OKABE.get(m), alpha=0.18, lw=0)
            rnd = [r["random"][k] for r in rs if r.get("random") and k in r["random"]]
            if rnd:
                ax.axhline(float(np.mean(rnd)), color="0.45", lw=0.7, ls="--", label="Random policy")
            if i == 0:
                ax.set_title(("$N = " + f"{N:,}".replace(",", "{,}") + "$"))
            if j == 0:
                ax.set_ylabel(f"{ENVS[env].split()[0]}\nreturn")
            ax.tick_params(labelsize=6.5)
            if not rs:
                ax.text(0.5, 0.5, "not run", ha="center", va="center", transform=ax.transAxes, color="0.5", fontsize=7)
    seen = {}
    for a in axes.ravel():
        for hnd, lab in zip(*a.get_legend_handles_labels()):
            seen.setdefault(lab, hnd)
    fig.supxlabel("Environment steps (thousands)", fontsize=8, y=0.02)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    fig.legend(seen.values(), seen.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=7, fontsize=6.5)
    save(fig, out, "fig_learning", tag)


def scaling_table(rows):
    """Per task, N and critic: mean and 95% interval of normalized return, task success, steps to converge (also
    relative to FlashMAAC) and the critic-update speed-up over FlashMAAC, all from the same training runs."""
    df = score_runs(rows)
    out = []
    for (env, N), g in df.groupby(["env", "N"]):
        ref = g[g.method == "dense_flash"]
        ref_steps = ref["steps_to_conv"].mean() if "steps_to_conv" in ref else np.nan
        ref_ms = ref["update_ms"].mean()
        for m, d in g.groupby("method"):
            r = {"env": env, "N": N, "method": m, "seeds": len(d)}
            for col in ("return_norm", "task_success", "steps_to_conv", "update_ms"):
                r[col], r[col + "_ci"] = ci(d[col]) if col in d else (np.nan, np.nan)
            r["steps_rel"], r["steps_rel_ci"] = ci(d["steps_to_conv"] / ref_steps) if "steps_to_conv" in d else (np.nan, np.nan)
            r["speedup"], r["speedup_ci"] = ci(ref_ms / d["update_ms"])
            out.append(r)
    return pd.DataFrame(out)


def random_success(rows):
    """Task success of the uniformly random policy per (env, N), from the runs' random evaluations
    (spread: final landmark coverage; tag: capture rate of random predators against random prey)."""
    out = {}
    for r in rows:
        if r.get("random"):
            k = "final_coverage" if r["env"] == "spread" else "vr_capture_rate"
            if k in r["random"]:
                out.setdefault((r["env"], r["N"]), []).append(r["random"][k])
    return {key: float(np.mean(v)) for key, v in out.items()}


def fig_scaling_cost(t, out, tag=None, rnd=None, ylim_return=None):
    """(a) final return, random 0 and FlashMAAC 1 (optionally clipped with ylim_return);
    (b) task success, random policy as grey markers; (c) critic-update speed-up over FlashMAAC in the same runs."""
    if t.empty:
        return
    panels = [("return_norm", "(a) Final return\n(random 0, FlashMAAC 1)", 1.0),
              ("task_success", "(b) Task success", None),
              ("speedup", "(c) Critic-update speed-up\nover FlashMAAC (same runs)", 1.0)]
    fig, axes = plt.subplots(1, 3, figsize=(DBL, 2.2))
    ls = {"spread": "-", "tag": "--"}
    ns = sorted(set(t.N))
    for ax, (col, title, ref) in zip(axes, panels):
        for env in [e for e in ENVS if e in set(t.env)]:
            for m in _order(set(t.method)):
                if col == "speedup" and m in ("dense_flash", "none"):
                    continue
                d = t[(t.env == env) & (t.method == m)].sort_values("N")
                if d.empty or not np.isfinite(d[col]).any():
                    continue
                ax.errorbar(d.N, d[col], yerr=d[col + "_ci"].fillna(0), color=OKABE.get(m), marker=MARK.get(m, "o"),
                            ms=3.2, lw=1.0, ls=ls[env], capsize=1.5, elinewidth=0.6,
                            label=f"{LABEL.get(m, m)}" if env == "spread" else None)
            if col == "task_success" and rnd:
                xs = [n for n in ns if (env, n) in rnd]
                ax.plot(xs, [rnd[(env, n)] for n in xs], color="0.45", lw=0.8, ls=ls[env], marker="_", ms=7,
                        label="Random policy" if env == "spread" else None)
        if ref is not None:
            ax.axhline(ref, color="0.4", lw=0.6, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns); ax.set_xticklabels([f"{n:,}" for n in ns], fontsize=6)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Number of agents $N$", fontsize=7)
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=6.5)
    if ylim_return:
        axes[0].set_ylim(*ylim_return)
    axes[2].set_yscale("log")
    axes[2].set_yticks([0.125, 0.25, 0.5, 1, 2])
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}\u00d7"))
    axes[2].yaxis.set_minor_formatter(NullFormatter())
    seen = {}
    for a in axes:
        for hnd, lab in zip(*a.get_legend_handles_labels()):
            seen.setdefault(lab, hnd)
    seen["spread (solid)"] = plt.Line2D([], [], color="0.3", ls="-")
    seen["tag (dashed)"] = plt.Line2D([], [], color="0.3", ls="--")
    fig.tight_layout(w_pad=1.0)
    fig.legend(seen.values(), seen.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=6.5)
    save(fig, out, "fig_scaling_cost", tag)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results/figs")
    p.add_argument("--tag", default=None)
    p.add_argument("--ns", nargs="+", type=int, default=[64, 256, 1024],
                   help="numbers of agents to show")
    a = p.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    style()
    rows = [r for r in load_runs(a.runs) if r["N"] in set(a.ns)]
    if not rows:
        print("no finished runs in", a.runs); return
    fig_learning(rows, a.out, a.tag)
    t = scaling_table(rows)
    t.to_csv(Path(a.out) / "scaling_cost.csv", index=False)
    fig_scaling_cost(t, a.out, a.tag, random_success(rows))


if __name__ == "__main__":
    main()
