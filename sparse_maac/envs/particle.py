"""Particle worlds in plain PyTorch, batched over E environments and N agents, on CPU or GPU.

spread (cooperative): N agents and N landmarks.
  local reward (default): agent i loses, for each of its k nearest landmarks, the distance from that landmark to
      whichever agent is closest to it, and 1 per agent it touches. A landmark covered by a neighbour still
      counts, so agents gain by spreading out, and the signal does not fade as N grows.
  global reward (MPE simple_spread): one term over all landmarks, shared by everyone. With 64 agents each agent
      moves it by about 1/64, and in our tuning runs agents learned to avoid collisions but not to cover landmarks.
tag (competitive): N - N//3 predators (team 0) chase N//3 faster prey (team 1), as in MPE simple_tag. With
  tag_share_radius > 0 a predator is also paid for catches made by other predators within that radius.

The arena grows with N (L = sqrt(N/8)) so that density stays constant. Observations have a fixed size: own
velocity, own position / L, offsets to the k nearest landmarks and to the k nearest agents with their team, laid out
as [vel(2), pos(2), landmarks(2*kl), agents(3*ka)], so positions sit at pos_slice = (2, 4).

Episode metrics (`episode_metrics()`): spread reports final_coverage (share of landmarks with an agent within
cover_radius at the last step), success_rate and collision_rate; tag reports capture_rate (share of prey caught at
least once) and first_capture (mean step of a prey's first capture, T + 1 if never)."""
from __future__ import annotations

import math

import torch

from ..utils import batch_gather, nearest_k


class ParticleEnv:
    n_actions = 5
    pos_slice = (2, 4)

    def __init__(self, scenario="spread", n_agents=8, n_envs=8, episode_length=25, device="cpu", seed=0,
                 n_landmarks=None, world_size=None, k_agents=3, k_landmarks=3, dt=0.1, damping=0.25,
                 cover_radius=0.1, success_coverage=0.9, spread_reward="local", tag_share_radius=0.0):
        assert scenario in ("spread", "tag")
        assert spread_reward in ("local", "global")
        self.spread_reward = spread_reward
        self.tag_share_radius = float(tag_share_radius)
        self.scenario, self.N, self.E, self.T = scenario, n_agents, n_envs, episode_length
        self.dev, self.dt, self.damping = torch.device(device), dt, damping
        self.cover_radius, self.success_coverage = cover_radius, success_coverage
        self.L = float(world_size or max(1.0, math.sqrt(n_agents / 8)))
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(seed)
        N = n_agents
        if scenario == "spread":
            self.n_pred, self.n_prey = N, 0
            self.M = n_landmarks if n_landmarks is not None else N
            self.ranges = [(0, N)]
            self.team = torch.zeros(N)
            self.accel = torch.full((N,), 3.0)
            self.max_speed = torch.full((N,), 1.0)
            self.size = 0.05
        else:
            self.n_prey = max(1, N // 3)
            self.n_pred = N - self.n_prey
            self.M = n_landmarks or 0
            self.ranges = [(0, self.n_pred), (self.n_pred, N)]
            self.team = torch.cat([torch.zeros(self.n_pred), torch.ones(self.n_prey)])
            self.accel = torch.cat([torch.full((self.n_pred,), 3.0), torch.full((self.n_prey,), 4.0)])
            self.max_speed = torch.cat([torch.full((self.n_pred,), 1.0), torch.full((self.n_prey,), 1.3)])
            self.size = 0.05
        self.team, self.accel, self.max_speed = (t.to(self.dev) for t in (self.team, self.accel, self.max_speed))
        self.ka = min(k_agents, N - 1)
        self.kl = min(k_landmarks, self.M)
        self.obs_dim = 4 + 2 * self.kl + 3 * self.ka
        self.dirs = torch.tensor([[0., 0.], [-1., 0.], [1., 0.], [0., -1.], [0., 1.]], device=self.dev)
        self.reset()

    @property
    def spec(self):
        return dict(n_agents=self.N, obs_dim=self.obs_dim, n_actions=self.n_actions,
                    ranges=self.ranges, pos_slice=self.pos_slice)

    def _u(self, *shape):
        return torch.rand(*shape, device=self.dev, generator=self.gen)

    @torch.no_grad()
    def reset(self):
        L = self.L
        self.pos = (self._u(self.E, self.N, 2) * 2 - 1) * L
        self.vel = torch.zeros(self.E, self.N, 2, device=self.dev)
        self.lm = (self._u(self.E, self.M, 2) * 2 - 1) * 0.95 * L
        self.t = 0
        self._coll_sum = torch.zeros(self.E, device=self.dev)
        self._first_catch = torch.full((self.E, self.n_prey), float(self.T + 1), device=self.dev)
        return self._obs()

    @torch.no_grad()
    def _obs(self):
        parts = [self.vel, self.pos / self.L]
        if self.kl > 0:
            _, li = nearest_k(self.pos, self.lm, self.kl)
            parts.append((batch_gather(self.lm, li) - self.pos[:, :, None]).flatten(2))
        if self.ka > 0:
            _, ai = nearest_k(self.pos, self.pos, self.ka, exclude_self=True)
            rel = batch_gather(self.pos, ai) - self.pos[:, :, None]
            parts.append(torch.cat([rel, self.team[ai][..., None]], -1).flatten(2))
        return torch.cat(parts, -1)

    @torch.no_grad()
    def step(self, actions):
        """actions: (E,N) long -> obs (E,N,D), rewards (E,N), done (bool), info (dict of floats)."""
        force = self.dirs[actions]
        self.vel = self.vel * (1 - self.damping) + force * self.accel[None, :, None] * self.dt
        speed = self.vel.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.vel = self.vel * (self.max_speed[None, :, None] / speed).clamp(max=1.0)
        self.pos = self.pos + self.vel * self.dt
        hit = self.pos.abs() > self.L
        self.pos = self.pos.clamp(-self.L, self.L)
        self.vel = torch.where(hit, torch.zeros_like(self.vel), self.vel)
        self.t += 1
        rew, info = self._reward()
        return self._obs(), rew, self.t >= self.T, info

    def _reward(self):
        s, pos = self.size, self.pos
        if self.scenario == "spread":
            dl, _ = nearest_k(self.lm, pos, 1)                     # (E,M,1) landmark -> nearest agent
            cover = -dl.mean((1, 2))
            own, _ = nearest_k(pos, self.lm, 1)                    # (E,N,1) agent -> nearest landmark
            da, _ = nearest_k(pos, pos, max(self.ka, 1), exclude_self=True)
            coll = (da < 2 * s).float().sum(-1)
            if self.spread_reward == "local":
                _, li = nearest_k(pos, self.lm, self.kl)                        # (E,N,kl) i's nearest landmarks
                rew = -batch_gather(dl[..., 0], li).mean(-1) - 1.0 * coll      # distance to whichever agent is closest
            else:
                rew = cover[:, None] - 0.1 * own[..., 0] / self.L - 1.0 * coll
            self._coll_sum += coll.mean(1)
            self._last_cover = (dl[..., 0] < self.cover_radius).float().mean(1)          # (E,)
            return rew, {"covered": (dl < 2 * s).float().mean().item(), "collisions": coll.mean().item()}
        P, Q = self.n_pred, self.n_prey
        pred, prey = pos[:, :P], pos[:, P:]
        contact = 2.5 * s                                          # predator radius 1.5s + prey radius s
        dp, _ = nearest_k(pred, prey, min(3, Q))                   # predator -> nearest prey
        dq, _ = nearest_k(prey, pred, min(3, P))                   # prey -> nearest predator
        c_pred = (dp < contact).float().sum(-1)
        c_prey = (dq < contact).float().sum(-1)
        caught_now = c_prey > 0                                                         # (E,Q)
        self._first_catch = torch.where(caught_now & (self._first_catch > self.T),
                                        torch.full_like(self._first_catch, float(self.t)), self._first_catch)
        if self.tag_share_radius > 0:     # catches by predators within the radius, own included
            near = (torch.cdist(pred, pred) <= self.tag_share_radius).float()              # (E,P,P)
            c_pred = torch.einsum("epq,eq->ep", near, c_pred)
        r_pred = c_pred - 0.1 * dp[..., 0] / self.L
        r_prey = -c_prey + 0.1 * dq[..., 0].clamp(max=1.0) / self.L
        rew = torch.cat([r_pred, r_prey], 1)
        return rew, {"catches": c_prey.sum(-1).mean().item(),
                     "pred_rew": r_pred.mean().item(), "prey_rew": r_prey.mean().item()}

    @torch.no_grad()
    def episode_metrics(self):
        """Per-env task metrics of the episode that just ended -> dict of (E,) tensors."""
        if self.scenario == "spread":
            cov = self._last_cover
            return {"final_coverage": cov, "success": (cov >= self.success_coverage).float(),
                    "collision_rate": self._coll_sum / max(self.t, 1)}
        caught = self._first_catch <= self.T
        return {"capture_rate": caught.float().mean(1), "success": caught.float().mean(1),
                "first_capture": self._first_catch.mean(1)}

    def render(self, env_index=0, px=512):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(px / 100, px / 100), dpi=100)
        ax.set_xlim(-self.L, self.L); ax.set_ylim(-self.L, self.L); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        p = self.pos[env_index].cpu().numpy()
        ms = max(2.0, 120.0 / math.sqrt(self.N))
        if self.M:
            lm = self.lm[env_index].cpu().numpy()
            ax.scatter(lm[:, 0], lm[:, 1], marker="*", c="k", s=ms * 2.5, zorder=1)
        team = self.team.cpu().numpy()
        colors = np.where(team == 0, "#3b6fd4" if self.scenario == "spread" else "#d94141", "#2e9e4f")
        ax.scatter(p[:, 0], p[:, 1], c=colors, s=ms * 3, zorder=2, edgecolors="none")
        ax.set_title(f"{self.scenario} N={self.N} t={self.t}", fontsize=8)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return img
