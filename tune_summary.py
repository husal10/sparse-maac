"""Summarize the tuning runs of experiments/03a_tune.sh (FlashMAAC at 64 agents).

    python tune_summary.py runs_tune

For each task, reward scale and learning rate: the final return (spread: all agents; tag: the predator team only,
since the all-agent mean mixes predator gains with prey losses), its gain over the random policy, task success, and
every seed's value. It also says when the best setting lies on the edge of the grid and whether it beats the
runner-up by more than the seed-to-seed spread. Runs with different spread rewards must sit in different folders,
since run names do not record the reward.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

NAME = re.compile(r"^(?P<env>[a-z]+)_rs(?P<rs>[0-9.e+-]+)_lr(?P<lr>[0-9.e+-]+)_s(?P<seed>\d+)$")
PRIMARY = {"spread": "return_mean", "tag": "vr_return_team0"}     # tag: trained predators against random prey
SUCCESS = {"spread": "success_rate", "tag": "vr_capture_rate"}
FALLBACK = {"vr_return_team0": "return_team0", "vr_capture_rate": "capture_rate"}   # older runs without the vs-random evaluation


def load(root):
    rows = []
    for f in sorted(Path(root).glob("*/final_eval.json")):
        m = NAME.match(f.parent.name)
        if not m:
            continue
        env = m["env"]
        fin = json.load(open(f))
        rnd_f = f.parent / "random_eval.json"
        rnd = json.load(open(rnd_f)) if rnd_f.exists() else {}
        key = PRIMARY.get(env, "return_mean")
        skey = SUCCESS.get(env, "")
        old = key not in fin
        if old:
            key, skey = FALLBACK.get(key, key), FALLBACK.get(skey, skey)
        r = fin.get(key, fin.get("return_mean"))
        r0 = rnd.get(key)
        rows.append(dict(old=old, env=env, rs=float(m["rs"]), lr=float(m["lr"]), seed=int(m["seed"]), R=r, R_random=r0,
                         gain=None if r0 is None else r - r0, success=fin.get(skey, float("nan")),
                         success_random=rnd.get(skey, float("nan")),
                         coverage=fin.get("final_coverage", float("nan")), coverage_random=rnd.get("final_coverage", float("nan"))))
    return rows


def mean_sd(v):
    v = [x for x in v if x is not None and x == x]
    if not v:
        return float("nan"), float("nan")
    return st.mean(v), (st.stdev(v) if len(v) > 1 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="runs_tune")
    a = ap.parse_args()
    rows = load(a.root)
    if not rows:
        print(f"no finished tuning runs in {a.root}")
        return
    old = sorted({r["env"] + f" rs{r['rs']:g} lr{r['lr']:g}" for r in rows if r["old"]})
    if old:
        print("skipped (no vs-random evaluation, so their return is not comparable; see tag_check.py):\n  "
              + "\n  ".join(old))
    rows = [r for r in rows if not r["old"]]
    best_cfg = {}
    for env in sorted({r["env"] for r in rows}):
        g = defaultdict(list)
        for r in rows:
            if r["env"] == env:
                g[(r["rs"], r["lr"])].append(r)
        er = [r for r in rows if r["env"] == env]
        rnd = mean_sd([r["R_random"] for r in er])[0]
        srnd = mean_sd([r["success_random"] for r in er])[0]
        print(f"\n== {env}: primary = {PRIMARY.get(env)} (random policy: {rnd:.3f}); success = {SUCCESS.get(env)} "
              f"(random policy: {srnd:.3f})")
        if env == "spread":
            print(f"   final coverage: trained {mean_sd([r['coverage'] for r in er])[0]:.3f} (best run "
                  f"{max((r['coverage'] for r in er if r['coverage'] == r['coverage']), default=float('nan')):.3f}), random {mean_sd([r['coverage_random'] for r in er])[0]:.3f}")
        print(f"{'reward_scale':>12} {'lr':>8} {'return':>9} {'sd':>7} {'gain vs random':>15} {'success':>8}  per seed")
        table = []
        for (rs, lr), rr in g.items():
            m, sd = mean_sd([r["R"] for r in rr])
            gm, _ = mean_sd([r["gain"] for r in rr])
            sm, _ = mean_sd([r["success"] for r in rr])
            table.append((m, sd, gm, sm, rs, lr, rr))
        table.sort(key=lambda t: -t[0])
        for m, sd, gm, sm, rs, lr, rr in table:
            seeds = ", ".join(f"{r['R']:.3f}" for r in sorted(rr, key=lambda r: r["seed"]))
            print(f"{rs:>12g} {lr:>8g} {m:>9.3f} {sd:>7.3f} {gm:>15.3f} {sm:>8.3f}  [{seeds}]  n={len(rr)}")
        b, nxt = table[0], (table[1] if len(table) > 1 else None)
        rs_grid, lr_grid = sorted({k[0] for k in g}), sorted({k[1] for k in g})
        edge = [f"reward scale {b[4]:g} is the {'smallest' if b[4] == rs_grid[0] else 'largest'} tried"
                for _ in [0] if b[4] in (rs_grid[0], rs_grid[-1])]
        edge += [f"lr {b[5]:g} is the {'smallest' if b[5] == lr_grid[0] else 'largest'} tried"
                 for _ in [0] if b[5] in (lr_grid[0], lr_grid[-1]) and len(lr_grid) > 1]
        pooled_sd = mean_sd([t[1] for t in table if t[1] == t[1]])[0]
        print(f"best: reward_scale={b[4]:g}, lr={b[5]:g}" + (f"; on the grid edge: {'; '.join(edge)}" if edge else ""))
        if nxt is not None:
            gap = b[0] - nxt[0]
            print(f"gap to runner-up (rs={nxt[4]:g}, lr={nxt[5]:g}): {gap:.3f}; typical seed SD {pooled_sd:.3f}"
                  + (" -> within seed noise" if gap < pooled_sd else ""))
        best_cfg[env] = {(t[4], t[5]): k for k, t in enumerate(table)}
    if len(best_cfg) > 1:
        common = set.intersection(*(set(v) for v in best_cfg.values()))
        ranked = sorted(common, key=lambda c: sum(best_cfg[e][c] for e in best_cfg))
        print("\n== one setting for both tasks (sum of ranks, lower is better)")
        for c in ranked[:5]:
            print(f"  reward_scale={c[0]:g}, lr={c[1]:g}: ranks " +
                  ", ".join(f"{e} {best_cfg[e][c] + 1}" for e in best_cfg))


if __name__ == "__main__":
    main()
