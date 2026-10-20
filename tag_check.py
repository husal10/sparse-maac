"""Did the tag predators learn? Separate the predators' skill from their own prey's skill.

    python tag_check.py runs_tune            # every finished tag run in the folder
    python tag_check.py runs_tune/tag_rs3_lr3e-4_s1

In self-play both teams improve together, so the predator return of a trained run can fall below random-vs-random
simply because the trained prey escape. This plays each trained model in four match-ups on the same evaluation seed:
  trained predators vs trained prey   (self-play, what final_eval.json reports)
  trained predators vs random prey    (predator skill against a fixed opponent)
  random predators  vs trained prey   (prey skill against a fixed opponent)
  random predators  vs random prey    (the random anchor)
and prints the predator-team return and the capture rate for each.
"""
from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path
from types import SimpleNamespace

import torch

from sparse_maac.agent import SparseMAAC
from sparse_maac.runner import CrossPlayAgent, RandomAgent, evaluate, make_env


def check(run, dev, episodes, n_envs, seed):
    ck = Path(run) / "model.pt"
    d = torch.load(ck, map_location=dev)
    cfg = SimpleNamespace(**d["cfg"])
    env = make_env(cfg, dev, n_envs=n_envs, seed=seed)
    trained = SparseMAAC(env.spec, cfg, dev)
    trained.load(ck)
    rnd = RandomAgent(env.n_actions, seed)
    out = {}
    for name, (pred, prey) in {"trained vs trained": (trained, trained), "trained vs random": (trained, rnd),
                               "random vs trained": (rnd, trained), "random vs random": (rnd, rnd)}.items():
        rnd.gen = None                                       # same random stream in every match-up
        m = evaluate(env, CrossPlayAgent([pred, prey], env.ranges), episodes)
        out[name] = (m["return_team0"], m["capture_rate"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--episodes", type=int, default=8, help="batches of n_envs episodes")
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    p = Path(a.path)
    runs = [p] if (p / "model.pt").exists() else sorted(q.parent for q in p.glob("tag_*/model.pt"))
    if not runs:
        print("no tag runs with model.pt found"); return
    agg = {}
    print(f"{'run':32s} {'match-up (predators vs prey)':28s} {'pred. return':>12s} {'capture':>8s}")
    for r in runs:
        res = check(r, torch.device(a.device), a.episodes, a.n_envs, a.seed)
        for k, (ret, cap) in res.items():
            print(f"{r.name:32s} {k:28s} {ret:12.3f} {cap:8.3f}")
            agg.setdefault(k, []).append((ret, cap))
        print()
    if len(runs) > 1:
        print("mean over runs:")
        for k, v in agg.items():
            print(f"  {k:28s} return {st.mean(x[0] for x in v):8.3f}   capture {st.mean(x[1] for x in v):6.3f}")


if __name__ == "__main__":
    main()
