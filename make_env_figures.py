"""Environment and routing figures.

    python make_env_figures.py --out figs/env

env_snapshots            spread and tag at the start of an episode with 64 and 1,024 agents (same density).
spread_N256.gif,         25-step rollouts of a hand-written greedy policy (agents move to the nearest landmark;
tag_N256.gif             predators chase, prey flee). They show the tasks, not a learned policy.
fig_routing_selection    one spread state with 256 agents: for one agent, whom each router lets it attend to one by
                         one and which groups it only sees as a summary.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sparse_maac.envs import ParticleEnv
from sparse_maac.routing import BlockRouter, CentroidRouter, KNNRouter, LSHRouter

BLUE, VERM, GREEN, ORANGE, GREY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#9a9a9a"
DIRS = torch.tensor([[0., 0.], [-1., 0.], [1., 0.], [0., -1.], [0., 1.]])


def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8, "axes.titlesize": 8,
                         "legend.fontsize": 7, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})


def greedy(env):
    """Scripted illustration policy -> (E, N) actions."""
    pos = env.pos
    if env.scenario == "spread":
        d = torch.cdist(pos, env.lm)
        tgt = env.lm.gather(1, d.argmin(-1, keepdim=True).expand(-1, -1, 2)) - pos
    else:
        P = env.n_pred
        pred, prey = pos[:, :P], pos[:, P:]
        to_prey = prey.gather(1, torch.cdist(pred, prey).argmin(-1, keepdim=True).expand(-1, -1, 2)) - pred
        from_pred = prey - pred.gather(1, torch.cdist(prey, pred).argmin(-1, keepdim=True).expand(-1, -1, 2))
        tgt = torch.cat([to_prey, from_pred], 1)
    score = tgt @ DIRS.T                                                   # (E, N, 5)
    score[..., 0] = (tgt.norm(dim=-1) < 0.05).float() * 1e3               # stay once on the target
    return score.argmax(-1)


def draw(ax, env, e=0, title=None, caught=None):
    L = env.L
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5); s.set_color("0.6")
    p = env.pos[e].numpy()
    ms = min(14.0, max(1.0, 900.0 / env.N))
    if env.M:
        lm = env.lm[e].numpy()
        ax.scatter(lm[:, 0], lm[:, 1], marker="*", s=ms * 2.2, c="0.15", lw=0, zorder=1, label="landmark")
    if env.scenario == "spread":
        ax.scatter(p[:, 0], p[:, 1], s=ms, c=BLUE, lw=0, zorder=2, label="agent")
    else:
        P = env.n_pred
        ax.scatter(p[:P, 0], p[:P, 1], s=ms, c=VERM, lw=0, zorder=2, label="predator")
        prey = p[P:]
        if caught is None:
            ax.scatter(prey[:, 0], prey[:, 1], s=ms * 1.3, c=GREEN, lw=0, zorder=3, label="prey")
        else:
            c = caught[e].numpy()
            ax.scatter(prey[~c, 0], prey[~c, 1], s=ms * 1.3, c=GREEN, lw=0, zorder=3, label="prey")
            ax.scatter(prey[c, 0], prey[c, 1], s=ms * 1.3, facecolors="white", edgecolors=GREEN, lw=0.6, zorder=3,
                       label="prey caught at least once")
    if title:
        ax.set_title(title)


def snapshots(out):
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.35))
    for ax, (scn, N) in zip(axs, [("spread", 64), ("spread", 1024), ("tag", 64), ("tag", 1024)]):
        env = ParticleEnv(scn, N, 1, 25, seed=3)
        env.reset()
        who = f"{env.n_pred} predators, {env.n_prey} prey" if scn == "tag" else f"{env.N} agents, {env.M} landmarks"
        draw(ax, env, title=f"{scn.capitalize()}, N = {N:,}\n{who}\nworld {2 * env.L:.1f} units wide")
    h, l = [], []
    for ax in axs:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=len(l), frameon=False, markerscale=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(out) / f"env_snapshots.{ext}")
    plt.close(fig)


def gif(out, scn, N=256, T=25, seed=7):
    import imageio.v2 as imageio
    env = ParticleEnv(scn, N, 1, T, seed=seed)
    env.reset()
    frames = []
    for t in range(T + 1):
        fig, ax = plt.subplots(figsize=(3.6, 3.75), dpi=110)
        caught = (env._first_catch <= env.t) if scn == "tag" else None
        if scn == "spread":
            d = torch.cdist(env.lm, env.pos).min(-1).values[0]
            note = f"landmarks covered: {(d < env.cover_radius).float().mean().item():.0%}"
        else:
            note = f"prey caught at least once: {caught.float().mean().item():.0%}"
        draw(ax, env, title=f"{scn}, N = {N}, step {env.t}/{T}\n{note}", caught=caught)
        ax.text(0.01, -0.03, "scripted greedy policy (illustration)", transform=ax.transAxes, fontsize=6, color="0.4",
                va="top")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
        if t < T:
            env.step(greedy(env))
    imageio.mimsave(Path(out) / f"{scn}_N{N}.gif", frames, duration=0.25, loop=0)
    imageio.imwrite(Path(out) / f"{scn}_N{N}_last.png", frames[-1])


def selection(out, N=256, seed=11):
    env = ParticleEnv("spread", N, 1, 25, seed=seed)
    env.reset()
    pos = env.pos                                                          # (1, N, 2)
    p = pos[0].numpy()
    i = int(((pos[0]) ** 2).sum(-1).argmin())                              # the agent closest to the centre
    s = math.ceil(math.sqrt(N))
    panels = []
    r = KNNRouter(s)(pos)
    panels.append(("kNN", set(r.idx[0, i][r.mask[0, i]].tolist()), []))
    torch.manual_seed(0)
    r = LSHRouter(s, rounds=2)(pos)
    panels.append(("LSH (2 rounds)", set(r.idx[0, i][r.mask[0, i]].tolist()), []))
    torch.manual_seed(0)
    r = CentroidRouter(s)(pos)
    a = r.assign[0]
    groups = []
    for c in range(r.n_clusters):
        mem = [j for j in torch.nonzero(a == c).flatten().tolist() if j != i]
        if mem:
            groups.append(mem)
    panels.append(("C-MAAC (centroid)", set(), groups))
    b = max(1, round(math.sqrt(N / 3)))
    r = BlockRouter(b)(pos)
    order = r.perm[0].tolist()
    blocks = [order[k:k + b] for k in range(0, N, b)]
    bi = next(k for k, blk in enumerate(blocks) if i in blk)
    near = set(j for k in (bi - 1, bi, bi + 1) if 0 <= k < len(blocks) for j in blocks[k] if j != i)
    far = [blk for k, blk in enumerate(blocks) if abs(k - bi) > 1]
    panels.append(("Z-MAAC (block near-far)", near, far))

    fig, axs = plt.subplots(2, 2, figsize=(7.0, 7.6))
    L = env.L
    for ax, (name, indiv, grps) in zip(axs.flat, panels):
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        rest = [j for j in range(N) if j != i and j not in indiv and not any(j in g for g in grps)]
        ax.scatter(p[rest, 0], p[rest, 1], s=7, c="0.85", lw=0, zorder=1, label="not seen by agent i")
        for g in grps:
            gp = p[g]
            ax.scatter(gp[:, 0], gp[:, 1], s=7, c=GREY, lw=0, zorder=2)
            m = gp.mean(0)
            for q in gp:
                ax.plot([q[0], m[0]], [q[1], m[1]], color="0.75", lw=0.3, zorder=1)
            ax.scatter([m[0]], [m[1]], s=12 + 2.2 * len(g), marker="D", facecolors="white", edgecolors="0.2", lw=0.8,
                       zorder=4)
        if indiv:
            ip = p[sorted(indiv)]
            ax.scatter(ip[:, 0], ip[:, 1], s=10, c=BLUE, lw=0, zorder=3, label="seen individually")
        ax.scatter([p[i, 0]], [p[i, 1]], s=70, facecolors="none", edgecolors="black", lw=1.4, zorder=5, label="agent i")
        parts = ([f"{len(indiv)} agents individually"] if indiv else []) + ([f"{len(grps)} group summaries"] if grps else [])
        seen = len(indiv) + sum(len(g) for g in grps)
        ax.set_title(f"{name}\n{' + '.join(parts)}\ncovering {seen} of the {N - 1} other agents")
    handles = [plt.Line2D([], [], ls="", marker="o", mfc="none", mec="black", mew=1.4, ms=8, label="agent i"),
               plt.Line2D([], [], ls="", marker="o", color=BLUE, ms=4, label="seen individually"),
               plt.Line2D([], [], ls="", marker="o", color=GREY, ms=4, label="member of a group"),
               plt.Line2D([], [], ls="", marker="D", mfc="white", mec="0.2", ms=6, label="group summary (size = members)"),
               plt.Line2D([], [], ls="", marker="o", color="0.85", ms=4, label="not seen by agent i")]
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=2.0)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(Path(out) / f"fig_routing_selection.{ext}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="figs/env")
    ap.add_argument("--skip_gif", action="store_true")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    style()
    snapshots(a.out)
    selection(a.out)
    if not a.skip_gif:
        gif(a.out, "spread"); gif(a.out, "tag")
    print("wrote", sorted(p.name for p in Path(a.out).iterdir()))


if __name__ == "__main__":
    main()
