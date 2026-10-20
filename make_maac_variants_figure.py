"""Figure and table: three implementations of the same MAAC critic.

    python make_maac_variants_figure.py --bench results/2026-10-15_gpu_sweep/bench_gpu.csv --out figs/maac_variants

MAAC loops               MAAC's released code: one network per agent, Python loops over agents and heads.
MAAC loops, shared       the same loops with one network per agent type.
FlashMAAC                one network per agent type, all agents and heads in one fused attention call.
All three compute the same function (the tests compare values to 1e-5 and gradients to 1e-4). Time is one critic
update (target forward, forward and backward, optimizer step), median of 30.
Writes fig_maac_variants.{pdf,png} and table_maac_variants.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SERIES = [  # (mode, label, colour, marker) -- validated categorical slots 2, 3, 1 (dataviz palette, light mode)
    ("dense_loop", "MAAC loops, original structure", "#eb6834", "s"),
    ("dense_loop_shared", "MAAC loops, shared parameters", "#1baf7a", "^"),
    ("dense_flash", "FlashMAAC (batched)", "#2a78d6", "o"),
]
PARAMS = {"dense_loop": {64: 2_564_544, 512: 20_171_392}, "dense_loop_shared": 88_581, "dense_flash": 88_581}


def slope(d, nmin=64):
    s = d[d.N >= nmin]
    return float(np.polyfit(np.log(s.N), np.log(s.ms), 1)[0]) if len(s) >= 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=None,
                    help="default: results/bench_gpu.csv if you ran the sweep, else the published one")
    ap.add_argument("--out", default="figs/maac_variants")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--column", action="store_true", help="single-column layout (3.4 in wide), file fig_maac_variants_col")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
                         "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
    if a.bench is None:
        a.bench = "results/bench_gpu.csv" if Path("results/bench_gpu.csv").exists() else "results/2026-10-15_gpu_sweep/bench_gpu.csv"
    df = pd.read_csv(a.bench)
    if "status" in df:
        df = df[df.status == "ok"]                  # rows that did not fit carry no valid time
    print(f"bench: {a.bench}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    b = df[df.batch == a.batch]
    colw = a.column
    fs = 5.6 if colw else 7                                # end-label font size
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.5) if colw else (6.8, 2.45))
    for ax, col, ylab in ((axes[0], "ms", "Critic update time (ms)"), (axes[1], "mem_mb", "Peak GPU memory (MB)")):
        for mode, lab, c, mk in SERIES:
            d = b[b["mode"] == mode].sort_values("N")
            ax.plot(d.N, d[col], color=c, lw=1.6, marker=mk, ms=4.2, mec="white", mew=0.6, zorder=3)
            x, y = d.N.iloc[-1], d[col].iloc[-1]
            short = {"dense_loop": "loops, original", "dense_loop_shared": "loops, shared",
                     "dense_flash": "FlashMAAC"}[mode]
            if colw:                                      # the legend names the curves; no end labels
                continue
            dy = {"dense_loop": 3, "dense_loop_shared": -3}.get(mode, 0)   # the two loop curves end close together
            ax.annotate(short, (x, y), xytext=(4, dy), textcoords="offset points",
                        va={3: "bottom", -3: "top"}.get(dy, "center"), fontsize=fs, color="#333")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi * 2.2)   # headroom for the end labels below the panel title
        ax.set_xlabel("Number of agents N")
        ax.set_ylabel(ylab)
        if colw:
            ax.set_xticks([16, 64, 256, 1024, 8192]); ax.set_xticklabels(["16", "64", "256", "1k", "8k"])
        else:
            ax.set_xticks([16, 64, 256, 1024, 4096, 8192]); ax.set_xticklabels(["16", "64", "256", "1k", "4k", "8k"])
        ax.grid(True, which="major", color="#e6e6e3", lw=0.5); ax.set_axisbelow(True)
        ax.set_xlim(12, 8192 * 3.4)
    # speed-up annotation at N = 512 on the time panel
    t = b.set_index(["mode", "N"])["ms"]
    sp = t[("dense_loop", 512)] / t[("dense_flash", 512)]
    axes[0].annotate("", xy=(512, t[("dense_flash", 512)]), xytext=(512, t[("dense_loop", 512)]),
                     arrowprops=dict(arrowstyle="<->", color="#555", lw=0.7))
    axes[0].text(560, np.sqrt(t[("dense_loop", 512)] * t[("dense_flash", 512)]), f"{sp:.0f}×", fontsize=6.5 if colw else 8,
                 va="center", color="#222")
    axes[0].set_title("(a) Time per critic update", fontsize=8, loc="left")
    axes[1].set_title("(b) Peak memory", fontsize=8, loc="left")
    h = [plt.Line2D([], [], color=c, marker=mk, lw=1.6, ms=4.2, mec="white", mew=0.6) for _, _, c, mk in SERIES]
    if colw:
        for ax in axes:
            ax.tick_params(labelsize=6); ax.xaxis.label.set_size(6.5); ax.yaxis.label.set_size(6.5)
            ax.title.set_size(7)
    fig.legend(h, [s[1] for s in SERIES], loc="lower center", ncol=1 if colw else 3, frameon=False,
               fontsize=6.3 if colw else 7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.17 if colw else 0.08, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_maac_variants{'_col' if colw else ''}.{ext}", dpi=300, bbox_inches="tight")

    rows = []
    for mode, lab, _, _ in SERIES:
        r = {"Critic": lab}
        for bb in (8, 16):
            d = df[(df["mode"] == mode) & (df.batch == bb)].set_index("N")
            for n in (64, 512):
                r[f"ms N={n} B={bb}"] = d.ms.get(n, float("nan"))
            r[f"slope B={bb}"] = slope(df[(df["mode"] == mode) & (df.batch == bb)],
                                       512 if mode == "dense_flash" else 64)
            r[f"MB N=512 B={bb}"] = d.mem_mb.get(512, float("nan"))
        p = PARAMS[mode]
        r["params N=64"] = p[64] if isinstance(p, dict) else p
        r["params N=512"] = p[512] if isinstance(p, dict) else p
        r["largest N run"] = int(df[df["mode"] == mode].N.max())
        rows.append(r)
    tab = pd.DataFrame(rows)
    f = tab.set_index("Critic")
    flash = f.loc["FlashMAAC (batched)"]
    lines = ["| Critic | Parameters (N=64 / 512) | Update ms, N=64 (B=8/16) | Update ms, N=512 (B=8/16) | "
             "Slow-down vs ours at N=512 | Memory MB, N=512 (B=8/16) | Time exponent (B=8/16) | Largest N |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r in f.iterrows():
        sd = (f"{r['ms N=512 B=8'] / flash['ms N=512 B=8']:.0f}x / {r['ms N=512 B=16'] / flash['ms N=512 B=16']:.0f}x"
              if name != "FlashMAAC (batched)" else "1x")
        lines.append(f"| {name} | {int(r['params N=64']):,} / {int(r['params N=512']):,} | "
                     f"{r['ms N=64 B=8']:.1f} / {r['ms N=64 B=16']:.1f} | {r['ms N=512 B=8']:.1f} / {r['ms N=512 B=16']:.1f} | "
                     f"{sd} | {r['MB N=512 B=8']:,.0f} / {r['MB N=512 B=16']:,.0f} | "
                     f"{r['slope B=8']:.2f} / {r['slope B=16']:.2f} | {int(r['largest N run']):,} |")
    lines.append("")
    lines.append("*Fitted exponent of update time in N: loops over N = 64-512, ours over N = 512-8,192 (below 512 the "
                 "fused call is dominated by fixed overheads).")
    (out / "table_maac_variants.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
