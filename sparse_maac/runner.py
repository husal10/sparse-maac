from __future__ import annotations

import torch

from .envs import ParticleEnv


def make_env(cfg, device, n_envs=None, seed=None):
    return ParticleEnv(cfg.env, cfg.n_agents, n_envs or cfg.n_envs, cfg.episode_length, device,
                       cfg.seed if seed is None else seed, cfg.n_landmarks, cfg.world_size,
                       cfg.k_obs_agents, cfg.k_obs_landmarks,
                       cover_radius=getattr(cfg, "cover_radius", 0.1),
                       success_coverage=getattr(cfg, "success_coverage", 0.9),
                       spread_reward=getattr(cfg, "spread_reward", "global"),
                       tag_share_radius=getattr(cfg, "tag_share_radius", 0.0))


class RandomAgent:
    """Uniform random policy; normalized returns put it at 0."""

    def __init__(self, n_actions, seed=0):
        self.A, self.gen = n_actions, None
        self.seed = seed

    def prep_rollouts(self):
        pass

    @torch.no_grad()
    def act(self, obs, explore=True):
        if self.gen is None:
            self.gen = torch.Generator(device=obs.device).manual_seed(self.seed)
        return torch.randint(0, self.A, obs.shape[:2], device=obs.device, generator=self.gen)


class CrossPlayAgent:
    """Team k acts with agents[k], each normalizing observations with its own statistics."""

    def __init__(self, agents, ranges):
        assert len(agents) == len(ranges)
        self.agents, self.ranges = agents, ranges

    def prep_rollouts(self):
        for a in self.agents:
            a.prep_rollouts()

    @torch.no_grad()
    def act(self, obs, explore=True):
        out = None
        for a, (s, e) in zip(self.agents, self.ranges):
            act = a.act(obs, explore)
            out = act.clone() if out is None else out
            out[:, s:e] = act[:, s:e]
        return out


@torch.no_grad()
def evaluate(env, agent, episodes=1, explore=False, frame_every=0, env_index=0, episode_values=False):
    """Roll out `episodes` batches of env.E episodes and average returns (return_mean, return_team<k>), per-step
    info (avg_*) and the episode metrics of env.episode_metrics(). episode_values=True also keeps each episode's
    final coverage (spread); frame_every > 0 also returns rendered frames of one environment."""
    agent.prep_rollouts()
    sums, frames, ep_ret, ep_task = {}, [], [], {}
    for _ in range(episodes):
        obs, ret, done, t = env.reset(), torch.zeros(env.E, env.N, device=env.dev), False, 0
        while not done:
            obs, rew, done, info = env.step(agent.act(obs, explore))
            ret += rew
            for k, v in info.items():
                sums[k] = sums.get(k, 0.0) + v
            if frame_every and t % frame_every == 0:
                frames.append(env.render(env_index))
            t += 1
        ep_ret.append(ret)
        for k, v in env.episode_metrics().items():
            ep_task.setdefault(k, []).append(v)
    ret = torch.stack(ep_ret).mean(0)                       # (E,N)
    out = {f"avg_{k}": v / (episodes * env.T) for k, v in sums.items()}
    out["return_mean"] = ret.mean().item()
    for ti, (s, e) in enumerate(env.ranges):
        out[f"return_team{ti}"] = ret[:, s:e].mean().item()
    for k, v in ep_task.items():
        out["success_rate" if k == "success" else k] = torch.cat(v).float().mean().item()
    if episode_values and "final_coverage" in ep_task:
        out["final_coverage_episodes"] = [round(x, 5) for x in torch.cat(ep_task["final_coverage"]).float().tolist()]
    out["n_episodes"] = episodes * env.E
    return (out, frames) if frame_every else out


VS_RANDOM_SEED = 4242   # the random opponent makes the same moves in every run


def evaluate_vs_random(env, agent, episodes=1, seed=VS_RANDOM_SEED):
    """Tag only: play each trained team against a random opponent team.

    In self-play both teams improve together, and trained predators can score below random ones simply because the
    trained prey escape. Against a fixed random opponent the two skills separate:
      vr_return_team0, vr_capture_rate   trained predators vs random prey (the tag return we report)
      vr_return_team1, vr_escape_rate    random predators vs trained prey
    """
    if len(env.ranges) < 2:
        return {}
    rnd = RandomAgent(env.n_actions, seed)
    m = evaluate(env, CrossPlayAgent([agent, rnd], env.ranges), episodes)
    rnd.gen = None
    q = evaluate(env, CrossPlayAgent([rnd, agent], env.ranges), episodes)
    return {"vr_return_team0": m["return_team0"], "vr_capture_rate": m["capture_rate"],
            "vr_return_team1": q["return_team1"], "vr_escape_rate": 1.0 - q["capture_rate"]}


def random_vs_random_anchor(m):
    """The same keys for the random policy itself, read from its ordinary evaluation."""
    if "return_team1" not in m:
        return {}
    return {"vr_return_team0": m["return_team0"], "vr_capture_rate": m["capture_rate"],
            "vr_return_team1": m["return_team1"], "vr_escape_rate": 1.0 - m["capture_rate"]}
