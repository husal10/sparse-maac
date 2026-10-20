"""Cost and learning-curve figures (PDF and 300-dpi PNG, Okabe-Ito palette, 3.4 in / 7.0 in wide).

    python plot_figures.py --bench results/2026-10-15_gpu_sweep/bench_gpu.csv --out figs
    python plot_figures.py --curves "dense_flash=runs/spread_N64_dense_flash_s*" "block=runs/spread_N64_block_s*"

fig_scaling: update time and memory against N with fitted exponents. fig_speedup: speed-up over the MAAC loop
and over FlashMAAC. fig_batch: speed-up per benchmark batch size. fig_pareto: approximation error against the
number of entries each agent attends to. fig_curves: return against environment steps and wall-clock time.
Inputs that are not given are skipped."""
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OKABE = {"dense_loop": "#000000", "dense_loop_shared": "#7f7f7f", "dense_flash": "#E69F00", "block": "#009E73", "hybrid": "#0072B2",
         "centroid": "#56B4E9", "knn": "#D55E00", "random": "#CC79A7", "cluster": "#F0E442", "lsh": "#999999",
         "none": "#bbbbbb", "meanfield": "#8c564b", "anneal": "#117733"}
MARK = {"dense_loop": "s", "dense_loop_shared": "s", "dense_flash": "D", "block": "o", "hybrid": "^", "centroid": "v", "knn": "P",
        "random": "X", "cluster": "*", "lsh": "h", "none": "x", "meanfield": "<", "anneal": ">"}
LABEL = {"dense_loop": "MAAC loop (orig. structure)", "dense_loop_shared": "MAAC loop (shared params)",
         "dense_loop_typeshared": "MAAC loop (shared params)", "dense_flash": "FlashMAAC", "block": "Z-MAAC (ours)",
         "hybrid": "Hybrid", "centroid": "C-MAAC (ours)", "knn": "kNN", "random": "Random-K",
         "cluster": "Cluster", "lsh": "LSH", "none": "No interaction",
         "meanfield": "Mean field", "anneal": "Annealed (FlashMAAC to Z-MAAC)"}
# fused backends ("knn+ebag", "block+flex") keep the colour of their router and get their own marker
for _m in ("cluster", "random", "knn", "lsh", "hybrid"):
    OKABE[_m + "+ebag"], MARK[_m + "+ebag"], LABEL[_m + "+ebag"] = OKABE[_m], "d", LABEL[_m] + " (fused sum)"
OKABE["block+flex"], MARK["block+flex"], LABEL["block+flex"] = OKABE["block"], "p", "Z-MAAC (ours, FlexAttention)"
COL, DBL = 3.4, 7.0


def plain_log_y(a):
    """Log y axis labelled 0.2, 0.5, 1, 2, 5, 10 ... instead of 2x10^-1 style."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    a.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.yaxis.set_minor_formatter(NullFormatter())


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"], "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 4, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.4, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.spines.top": False,
        "axes.spines.right": False})


def save(fig, out, name, tag=None):
    if tag:
        fig.text(0.995, 0.005, tag, ha="right", va="bottom", fontsize=5, color="0.5")
    for ext in ("pdf", "png"):
        fig.savefig(Path(out) / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", Path(out) / f"{name}.pdf")


def exponent(n, y, nmin=0):
    """Tail scaling exponent: log-log slope over N >= nmin (same rule as fit_crossover.py)."""
    n, y = np.asarray(n, float), np.asarray(y, float)
    m = n >= nmin
    return np.polyfit(np.log(n[m]), np.log(y[m]), 1)[0] if m.sum() >= 2 else float("nan")


def legend_below(fig, axes, ncol=5):
    """One de-duplicated legend under the panels, so it never covers data."""
    seen = {}
    for a in axes:
        for h, l in zip(*a.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=ncol)


def order(modes):
    ref = list(OKABE)
    return sorted(modes, key=lambda m: ref.index(m) if m in ref else 99)


def fig_scaling(df, out, tag, fit_min_n=512, sfx=""):
    fig, ax = plt.subplots(1, 2, figsize=(DBL, 2.5))
    for m in order(df["mode"].unique()):
        d = df[df["mode"] == m].dropna(subset=["ms"]).sort_values("N")
        for a, col, ylab in ((ax[0], "ms", "Critic update time (ms)"), (ax[1], "mem_mb", "Peak memory (MB)")):
            dd = d.dropna(subset=[col])
            if dd.empty:
                continue
            e = exponent(dd["N"].values, dd[col].clip(lower=1e-3).values, fit_min_n)
            et = f" ($N^{{{e:.2f}}}$)" if a is ax[0] and np.isfinite(e) else ""
            a.plot(dd["N"], dd[col], marker=MARK.get(m, "o"), color=OKABE.get(m), label=LABEL.get(m, m) + et)
            a.set_xscale("log", base=2); a.set_yscale("log"); a.set_xlabel("Number of agents $N$"); a.set_ylabel(ylab)
    ax[0].set_title(f"(a) Compute (exponent fitted on $N \\geq {fit_min_n}$)"); ax[1].set_title("(b) Memory")
    fig.tight_layout(); legend_below(fig, ax[:1]); save(fig, out, "fig_scaling" + sfx, tag)


def fig_speedup(df, out, tag, sfx=""):
    fig, ax = plt.subplots(1, 2, figsize=(DBL, 2.4), sharey=False)
    for a, ref, ttl in ((ax[0], "dense_loop", "(a) vs MAAC loop (original structure)"), (ax[1], "dense_flash", "(b) vs FlashMAAC")):
        r = df[df["mode"] == ref].set_index("N")
        if r.empty:
            a.set_visible(False); continue
        for m in order(df["mode"].unique()):
            if m == ref or m.startswith("dense_loop"):
                continue
            d = df[df["mode"] == m].set_index("N").join(r[["ms", "mem_mb"]], rsuffix="_ref", how="inner")
            if d.empty:
                continue
            a.plot(d.index, d["ms_ref"] / d["ms"], marker=MARK.get(m, "o"), color=OKABE.get(m), label=LABEL.get(m, m))
        a.axhline(1, color="k", lw=0.6, ls="--"); a.set_xscale("log", base=2); a.set_yscale("log"); plain_log_y(a)
        a.set_xlabel("Number of agents $N$"); a.set_ylabel("Speed-up (x)"); a.set_title(ttl)
    fig.tight_layout(); legend_below(fig, ax, ncol=4); save(fig, out, "fig_speedup" + sfx, tag)


def fig_batch(df, out, tag, modes=("centroid", "block"), ref="dense_flash"):
    """Effect of the benchmark batch size: speed-up over batched dense vs N, one line per batch, one panel per mode.
    Shows how the break-even point N* (where a line crosses 1) moves with batch size."""
    modes = [m for m in modes if m in set(df["mode"])]
    if not modes:
        return
    batches = sorted(df["batch"].unique())
    cmap = plt.get_cmap("viridis")
    col = {b: cmap(0.15 + 0.7 * i / max(len(batches) - 1, 1)) for i, b in enumerate(batches)}
    fig, ax = plt.subplots(1, len(modes), figsize=(DBL if len(modes) > 1 else COL, 2.3), squeeze=False)
    for a, m in zip(ax[0], modes):
        for b in batches:
            d = df[df["batch"] == b]
            r = d[d["mode"] == ref].set_index("N")
            j = d[d["mode"] == m].set_index("N").join(r[["ms"]], rsuffix="_ref", how="inner").sort_index()
            if len(j):
                a.plot(j.index, j["ms_ref"] / j["ms"], marker=MARK.get(m, "o"), color=col[b], label=f"batch {b}")
        a.axhline(1, color="k", lw=0.6, ls="--"); a.set_xscale("log", base=2); a.set_yscale("log"); plain_log_y(a)
        a.set_xlabel("Number of agents $N$"); a.set_ylabel(f"Speed-up vs {LABEL.get(ref, ref)}")
        a.set_title(LABEL.get(m, m))
    from matplotlib.lines import Line2D
    fig.tight_layout()
    fig.legend([Line2D([], [], color=col[b]) for b in batches], [f"benchmark batch {b}" for b in batches],
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(batches))
    save(fig, out, "fig_batch", tag)


def fig_pareto(df, out, tag):
    fig, ax = plt.subplots(1, 2, figsize=(DBL, 2.5))
    for m in order(df["mode"].unique()):
        d = df[df["mode"] == m].sort_values("tokens")
        for a, col in ((ax[0], "rel_err_x"), (ax[1], "rel_err_q")):
            a.plot(d["tokens"], d[col], marker=MARK.get(m, "o"), color=OKABE.get(m), label=LABEL.get(m, m))
    for a, yl in ((ax[0], "Rel. error of attended features"), (ax[1], "Rel. error of $Q$-values")):
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("Tokens attended per agent"); a.set_ylabel(yl)
    ax[0].set_title("(a) Attention fidelity"); ax[1].set_title("(b) Critic fidelity")
    fig.tight_layout(); legend_below(fig, ax); save(fig, out, "fig_pareto", tag)


def load_runs(pattern):
    """Learning curves of the runs matching `pattern`: the greedy evaluations (eval.csv) when present, with the
    task's return in column `return_mean` (tag: predators against random prey); otherwise the training returns."""
    runs = []
    for d in sorted(glob.glob(pattern)):
        e, f = Path(d) / "eval.csv", Path(d) / "metrics.csv"
        if e.exists():
            c = pd.read_csv(e)
            if "vr_return_team0" in c:
                c["return_mean"] = c["vr_return_team0"]
            runs.append(c)
        elif f.exists():
            runs.append(pd.read_csv(f))
    return runs


def ci95(a):
    n = a.shape[0]
    if n < 2:
        return np.zeros(a.shape[1])
    t = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}.get(n, 1.96)
    return t * a.std(0, ddof=1) / math.sqrt(n)


def smooth(y, k):
    if k <= 1:
        return y
    ker = np.ones(k) / k
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, ker, mode="valid")


def fig_curves(specs, out, tag, smooth_k=3, name="fig_curves"):
    fig, ax = plt.subplots(1, 2, figsize=(DBL, 2.5))
    pal = list(OKABE.values())[1:]
    for i, s in enumerate(specs):
        key, pat = s.split("=", 1)
        label = LABEL.get(key, key)                  # mode keys get the same name/colour as in the other figures
        col = OKABE.get(key.replace("_typeshared", "_shared"), pal[i % len(pal)])
        runs = load_runs(pat)
        if not runs:
            continue
        n = min(len(r) for r in runs)
        for a, xcol in ((ax[0], "env_steps"), (ax[1], "elapsed_s")):
            x = np.mean([r[xcol].values[:n] for r in runs], 0)
            Y = np.stack([smooth(r["return_mean"].values[:n], smooth_k) for r in runs])
            m, c = Y.mean(0), ci95(Y)
            a.plot(x, m, color=col, label=f"{label} (n={len(runs)})")
            a.fill_between(x, m - c, m + c, color=col, alpha=0.2, lw=0)
    ax[0].set_xlabel("Environment steps"); ax[1].set_xlabel("Wall-clock time (s)")
    for a in ax:
        a.set_ylabel("Mean episode return")
    ax[0].legend(); ax[0].set_title("(a) Sample efficiency"); ax[1].set_title("(b) Wall-clock efficiency")
    fig.tight_layout(); save(fig, out, name, tag)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench"); p.add_argument("--fidelity"); p.add_argument("--curves", nargs="*", default=[])
    p.add_argument("--out", default="figs"); p.add_argument("--tag", default=None)
    p.add_argument("--curves_name", default="fig_curves", help="file name of the learning-curve figure")
    p.add_argument("--vram_mb", type=float, default=None, help="see fit_crossover.py (CSVs without a status column)")
    p.add_argument("--fit_min_n", type=int, default=512, help="scaling exponents use N >= this (as fit_crossover.py)")
    p.add_argument("--batch", type=int, default=None, help="batch size of a legacy CSV without a batch column")
    p.add_argument("--batch_modes", nargs="+", default=["centroid", "block"], help="modes shown in fig_batch")
    p.add_argument("--modes", nargs="+", default=None,
                   help="benchmark modes to plot (default: all); e.g. leave out the fused '+ebag'/'+flex' paths")
    p.add_argument("--bench_name", default="", help="suffix added to the scaling/speed-up figure names")
    p.add_argument("--n_min", default="common",
                   help="first N plotted: 'common' = the same start for every mode (largest per-mode minimum N), "
                        "'all' = everything measured, or an integer")
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    style()
    if a.bench and Path(a.bench).exists():
        from fit_crossover import load               # same status rules as the crossover table
        df = load([a.bench], a.vram_mb, a.batch)
        bad = df[df["status"] != "ok"]               # rows that did not fit in GPU memory have no valid timing
        if len(bad):
            print("not plotted (does not fit):", ", ".join(f"{m}@{n}/B{b}" for m, n, b in bad[["mode", "N", "batch"]].values))
        df = df[df["status"] == "ok"]
        if a.modes:
            df = df[df["mode"].isin(a.modes)]
        batches = sorted(df["batch"].unique())
        for b in batches:                             # one set of figures per benchmark batch size
            d = df[df["batch"] == b]
            if a.n_min == "common":                  # every curve starts at the same N (fair visual comparison)
                n0 = int(d.groupby("mode")["N"].min().max())
            else:
                n0 = 0 if a.n_min == "all" else int(a.n_min)
            cut = d[d["N"] < n0]
            if len(cut):
                print(f"[B={b}] curves start at N={n0} for every mode; not plotted:",
                      ", ".join(f"{m}@{n}" for m, n in cut[["mode", "N"]].values))
            d = d[d["N"] >= n0]
            sfx = (f"_b{b}" if len(batches) > 1 else "") + a.bench_name
            btag = f"{a.tag}, B={b}" if (a.tag and b >= 0 and len(batches) > 1) else a.tag
            fig_scaling(d, a.out, btag, a.fit_min_n, sfx); fig_speedup(d, a.out, btag, sfx)
        if len(batches) > 1:
            fig_batch(df, a.out, a.tag, a.batch_modes)
    if a.fidelity and Path(a.fidelity).exists():
        fig_pareto(pd.read_csv(a.fidelity), a.out, a.tag)
    if a.curves:
        fig_curves(a.curves, a.out, a.tag, name=a.curves_name)


if __name__ == "__main__":
    main()
