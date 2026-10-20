"""Roll out a trained checkpoint, optionally as a GIF.

    python evaluate.py runs/<name>/model.pt --render out.gif --frame_every 1
    python evaluate.py runs/tag_..._block_s1/model.pt --vs runs/tag_..._dense_flash_s1/model.pt --json xp.json

With --vs, the predators come from the first checkpoint and the prey from the second.
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import imageio.v2 as imageio
import torch

from sparse_maac.agent import SparseMAAC
from sparse_maac.runner import CrossPlayAgent, evaluate, make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--render", default=None, help="output .gif path")
    p.add_argument("--frame_every", type=int, default=1)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--vs", default=None, help="cross-play: team 1 plays with this checkpoint's policy")
    p.add_argument("--n_envs", type=int, default=1)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--json", default=None, help="write the metrics here")
    a = p.parse_args()
    d = torch.load(a.ckpt, map_location=a.device)
    cfg = SimpleNamespace(**d["cfg"])
    env = make_env(cfg, a.device, n_envs=a.n_envs, seed=a.seed)
    agent = SparseMAAC(env.spec, cfg, a.device)
    agent.load(a.ckpt)
    if a.vs:
        d2 = torch.load(a.vs, map_location=a.device)
        cfg2 = SimpleNamespace(**d2["cfg"])
        assert (cfg2.env, cfg2.n_agents) == (cfg.env, cfg.n_agents), "cross-play needs the same env and N"
        other = SparseMAAC(env.spec, cfg2, a.device)
        other.load(a.vs)
        agent = CrossPlayAgent([agent, other], env.ranges)
    out = evaluate(env, agent, a.episodes, explore=a.stochastic,
                   frame_every=a.frame_every if a.render else 0)
    metrics, frames = out if a.render else (out, [])
    print({k: round(v, 3) for k, v in metrics.items()})
    if a.json:
        import json
        json.dump({"ckpt": a.ckpt, "vs": a.vs, **metrics}, open(a.json, "w"), indent=1)
    if a.render and frames:
        imageio.mimsave(a.render, frames, duration=0.15, loop=0)
        print(f"wrote {a.render} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
