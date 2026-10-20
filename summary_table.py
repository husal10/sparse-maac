"""Learning results per task, number of agents and critic: mean and 95% t-interval over seeds.

    python summary_table.py --runs results/training/main --out results/summary

Columns: final greedy return (tag: trained predators against random prey), the same normalized so that random play
is 0 and FlashMAAC is 1, the difference to FlashMAAC with a 95% Welch interval, environment steps (and minutes)
until a run reaches 90% of its own final level, task success (spread: share of landmarks covered at the end; tag:
share of prey caught at least once), critic update time and wall-clock time. Writes <out>.csv and <out>.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from collect_results import ci95, load_runs, score_runs, welch_ci

ORDER = ["dense_flash", "centroid", "block", "knn", "meanfield", "none"]
LABEL = {"dense_flash": "FlashMAAC", "block": "Z-MAAC", "centroid": "C-MAAC", "knn": "kNN",
         "meanfield": "Mean field", "none": "No interaction"}


def fmt(m, h, digits=3):
    if not np.isfinite(m):
        return "n/a"
    return f"{m:.{digits}f} ± {h:.{digits}f}" if np.isfinite(h) else f"{m:.{digits}f}"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs"); p.add_argument("--out", default="results/summary")
    a = p.parse_args(argv)
    rows = load_runs(a.runs)
    if not rows:
        print("no finished runs"); return
    df = score_runs(rows)
    out = []
    for (env, N), g in df.groupby(["env", "N"]):
        ref = g[g.method == "dense_flash"]["return_norm"].to_numpy()
        meths = [m for m in ORDER if m in set(g.method)] + sorted(set(g.method) - set(ORDER))
        for m in meths:
            d = g[g.method == m]
            r = {"env": env, "N": N, "critic": LABEL.get(m, m), "seeds": len(d)}
            for col in ("return", "return_norm", "steps_to_conv", "time_to_conv", "task_success", "update_ms", "wall_s"):
                v = d[col].to_numpy(float) if col in d else np.array([np.nan])
                v = v[np.isfinite(v)]
                r[col] = float(v.mean()) if len(v) else float("nan")
                r[col + "_ci"] = ci95(pd.Series(v)) if len(v) > 1 else float("nan")
                if col == "steps_to_conv":
                    r["converged"] = f"{len(v)}/{len(d)}"
            if m != "dense_flash" and len(ref) > 1 and len(d) > 1:
                diff, lo, hi = welch_ci(d["return_norm"].to_numpy(), ref, 95)
                r["vs_flashmaac"], r["vs_lo"], r["vs_hi"] = diff, lo, hi
            out.append(r)
    t = pd.DataFrame(out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out + ".csv", index=False)
    lines = ["| Task | N | Critic | Seeds | Final return | Relative (FlashMAAC = 1) | Difference to FlashMAAC [95% CI] | "
             "Steps to converge | Converged | Success | Update (ms) | Wall time (min) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        v = r.get("vs_flashmaac", np.nan)
        vs = f"{v:+.3f} [{r['vs_lo']:+.3f}, {r['vs_hi']:+.3f}]" if np.isfinite(v) else "reference"
        sc, sh = r["steps_to_conv"], r["steps_to_conv_ci"]
        conv = (f"{sc / 1e3:.0f}k ± {sh / 1e3:.0f}k" if np.isfinite(sh) else f"{sc / 1e3:.0f}k") if np.isfinite(sc) else "n/a"
        lines.append(f"| {r['env']} | {r['N']} | {r['critic']} | {r['seeds']} | {fmt(r['return'], r['return_ci'])} | "
                     f"{fmt(r['return_norm'], r['return_norm_ci'])} | {vs} | {conv} | {r['converged']} | "
                     f"{fmt(r['task_success'], r['task_success_ci'])} | {r['update_ms']:.1f} | {r['wall_s'] / 60:.0f} |")
    Path(a.out + ".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return t


if __name__ == "__main__":
    main()
