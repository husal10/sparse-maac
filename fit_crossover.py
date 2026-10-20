"""Where does each routed critic overtake FlashMAAC?

    python fit_crossover.py results/2026-10-15_gpu_sweep/bench_gpu.csv --ref dense_flash --fit_min_n 512 \
                            --out results/crossover

Per benchmark batch size: the largest N that fits in GPU memory (rows that do not fit are left out of everything
else), the crossover N* in time and in memory (smallest N from which the critic stays ahead, interpolated in log-log
between the measured sizes), the scaling exponents fitted for N >= fit_min_n, and speed-ups over the MAAC loops.
Writes <out>.csv, <out>.md and, with several batch sizes, <out>_by_batch.md.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd


def crossover(ns, ratio):
    """ratio = ref / mode (>1 means mode is better). First N after which ratio stays >1."""
    ns, ratio = np.asarray(ns, float), np.asarray(ratio, float)
    ok = ratio > 1
    if ok.all():
        return f"<={int(ns[0])}"
    if not ok[-1]:
        return f">{int(ns[-1])}"
    i = len(ok) - 1
    while i > 0 and ok[i - 1]:
        i -= 1                                   # ok from i onward, not ok at i-1
    x0, x1 = math.log(ns[i - 1]), math.log(ns[i])
    y0, y1 = math.log(ratio[i - 1]), math.log(ratio[i])
    return f"{math.exp(x0 + (0 - y0) * (x1 - x0) / (y1 - y0)):.0f}"


def exponent(ns, y, nmin):
    ns, y = np.asarray(ns, float), np.asarray(y, float)
    m = (ns >= nmin) & np.isfinite(y) & (y > 0)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(ns[m]), np.log(y[m]), 1)[0])


def load(paths, vram_mb=None, batch=None):
    """Read benchmark CSVs; normalise `status` (ok | spilled | oom | failed | skipped) and `batch`
    (legacy CSVs have no batch column: pass `batch`, else -1 = unknown)."""
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if "batch" not in df.columns:
        df["batch"] = batch if batch is not None else -1
    df["batch"] = df["batch"].fillna(batch if batch is not None else -1).astype(int)
    if "status" not in df.columns:
        df["status"] = "ok"
    df["status"] = df["status"].fillna("ok")
    df.loc[df["ms"].isna() & (df["status"] == "ok"), "status"] = "oom"
    if vram_mb:                                   # retro-classify (allocated peak is a lower bound of reserved)
        mem = df["reserved_mb"].fillna(df["mem_mb"]) if "reserved_mb" in df.columns else df["mem_mb"]
        df.loc[(df["status"] == "ok") & (mem > vram_mb), "status"] = "spilled"
    return df


def analyse(df, ref_mode="dense_flash", fit_min_n=512):
    """Per-batch analysis -> one row per (batch, mode)."""
    if "batch" not in df.columns:
        df = df.assign(batch=-1)
    parts = [analyse_one(d, ref_mode, fit_min_n).assign(batch=b) for b, d in df.groupby("batch", sort=True)]
    out = pd.concat(parts, ignore_index=True)
    return out[["batch"] + [c for c in out.columns if c != "batch"]]


def by_batch(out, ref_mode="dense_flash"):
    """Compact mode x batch table: time break-even N* and memory feasibility.
    'fits all' = never failed up to the largest N measured (it may simply not have been run larger, e.g. the loop
    is capped); 'fits <= n' = the next larger N did not fit."""
    col = "N*_time_vs_" + ref_mode

    def cell(r):
        if pd.isna(r["max_fit_N"]):
            return "does not fit"
        nstar = r[col] if col in r and isinstance(r[col], str) else "ref"
        fit = f"fits all <= {r['max_fit_N']:.0f}" if r["first_nofit_N"] == "-" else f"fits <= {r['max_fit_N']:.0f}"
        return f"N* {nstar}; {fit}"
    return out.assign(cell=out.apply(cell, axis=1)).pivot(index="mode", columns="batch", values="cell") \
              .rename(columns=lambda b: f"B={b}")


def analyse_one(df, ref_mode="dense_flash", fit_min_n=512):
    fits = df[df["status"] == "ok"]
    ref = fits[fits["mode"] == ref_mode].set_index("N").sort_index()
    rows = []
    for m in sorted(df["mode"].unique()):
        allm = df[df["mode"] == m]
        d = fits[fits["mode"] == m].set_index("N").sort_index()
        bad = allm[allm["status"] != "ok"]["N"]
        r = {"mode": m, "max_fit_N": int(d.index.max()) if len(d) else float("nan"),
             "first_nofit_N": f"{int(bad.min())} ({allm.loc[bad.idxmin(), 'status']})" if len(bad) else "-",
             "exp_time": exponent(d.index, d["ms"].values, fit_min_n),
             "exp_mem": exponent(d.index, d["mem_mb"].values, fit_min_n)}
        j = d.join(ref[["ms", "mem_mb"]], rsuffix="_ref", how="inner")
        if m != ref_mode and len(j) >= 2:
            r["N*_time_vs_" + ref_mode] = crossover(j.index, j["ms_ref"] / j["ms"])
            r["N*_mem_vs_" + ref_mode] = crossover(j.index, j["mem_mb_ref"] / j["mem_mb"])
        rows.append(r)
    out = pd.DataFrame(rows)
    for lp in [m for m in fits["mode"].unique() if m.startswith("dense_loop")]:
        loop = fits[fits["mode"] == lp].set_index("N")
        for m in fits["mode"].unique():
            if m.startswith("dense_loop"):
                continue
            j = fits[fits["mode"] == m].set_index("N").join(loop[["ms", "mem_mb"]], rsuffix="_loop", how="inner")
            if len(j):
                n = j.index.max()
                out.loc[out["mode"] == m, f"speedup_vs_{lp}@N={n}"] = float(j.loc[n, "ms_loop"] / j.loc[n, "ms"])
                out.loc[out["mode"] == m, f"memratio_vs_{lp}@N={n}"] = float(j.loc[n, "mem_mb_loop"] / j.loc[n, "mem_mb"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="+"); p.add_argument("--ref", default="dense_flash")
    p.add_argument("--fit_min_n", type=int, default=512); p.add_argument("--out", default="results/crossover")
    p.add_argument("--vram_mb", type=float, default=None,
                   help="mark rows whose peak memory exceeds this as 'spilled' (for CSVs without a status column)")
    p.add_argument("--batch", type=int, default=None, help="batch size to assume for CSVs without a batch column")
    a = p.parse_args()
    df = load(a.csv, a.vram_mb, a.batch)
    nofit = df[df["status"] != "ok"]
    if len(nofit):
        print("excluded (does not fit):", ", ".join(f"{m}@{n}/B{b}:{s}" for m, n, b, s in
                                                    nofit[["mode", "N", "batch", "status"]].values))
    out = analyse(df, a.ref, a.fit_min_n)
    out.to_csv(a.out + ".csv", index=False)

    def md(t, index=False):
        try:
            return t.round(2).to_markdown(index=index)
        except ImportError:                          # tabulate not installed
            return t.round(2).to_string(index=index)
    blocks = []
    for b, d in out.groupby("batch", sort=True):
        d = d.drop(columns="batch").dropna(axis=1, how="all")
        blocks.append(f"### benchmark batch {b if b >= 0 else 'unknown'}\n\n" + md(d))
    open(a.out + ".md", "w").write("\n\n".join(blocks) + "\n")
    print("\n\n".join(blocks))
    if out["batch"].nunique() > 1:
        bb = by_batch(out, a.ref)
        txt = ("Time break-even N* vs " + a.ref + " | largest N that fits in GPU memory, per benchmark batch size\n\n"
               + md(bb, index=True) + "\n")
        open(a.out + "_by_batch.md", "w").write(txt)
        print("\n" + txt)


if __name__ == "__main__":
    main()
