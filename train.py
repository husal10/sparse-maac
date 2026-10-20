"""Train MAAC with a full or a routed attention critic.

    python train.py --env spread --n_agents 256 --attn centroid --n_envs 4 --batch_size 64
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from sparse_maac.agent import SparseMAAC
from sparse_maac.buffer import ReplayBuffer
from sparse_maac.config import parse
from sparse_maac.runner import RandomAgent, evaluate, evaluate_vs_random, make_env, random_vs_random_anchor


def main(argv=None):
    cfg = parse(argv)
    dev = torch.device("cuda" if (cfg.device == "auto" and torch.cuda.is_available()) else
                       ("cpu" if cfg.device == "auto" else cfg.device))
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    name = cfg.run_name or f"{cfg.env}_N{cfg.n_agents}_{cfg.attn}_s{cfg.seed}"
    run_dir = Path(cfg.out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(cfg), open(run_dir / "config.json", "w"), indent=1)
    env = make_env(cfg, dev)
    agent = SparseMAAC(env.spec, cfg, dev)
    buf = ReplayBuffer(cfg.buffer_size, env.N, env.obs_dim, dev, cfg.buffer_fp16)
    n_par = sum(p.numel() for p in agent.critic.parameters())
    print(f"[{name}] device={dev} critic_params={n_par:,} buffer={buf.mem_gb:.2f}GB obs_dim={env.obs_dim}")
    tb = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(run_dir / "tb"))
    except Exception:
        pass
    f = open(run_dir / "metrics.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(["episode", "env_steps", "return_mean", "q_loss", "pol_loss", "entropy", "update_ms", "elapsed_s", "peak_mem_mb"])
    n_eval = max(1, cfg.eval_episodes // cfg.n_envs)
    rnd = evaluate(make_env(cfg, dev, seed=cfg.seed + 10_000), RandomAgent(env.n_actions, cfg.seed), n_eval, episode_values=True)
    rnd.update(random_vs_random_anchor(rnd))
    json.dump(rnd, open(run_dir / "random_eval.json", "w"), indent=1)   # the 0 of the normalized return
    ef = open(run_dir / "eval.csv", "w", newline="")
    ew = None

    def log_eval(m, ep_done):
        nonlocal ew
        if ew is None:
            ew = csv.writer(ef); ew.writerow(["episode", "env_steps", "elapsed_s"] + list(m.keys()))
        ew.writerow([ep_done, steps, round(time.time() - t0, 2)] + list(m.values())); ef.flush()
    steps, t0, upd_t, upd_n, last = 0, time.time(), 0.0, 0, {}
    for ep in range(0, cfg.n_episodes, cfg.n_envs):
        obs = env.reset()
        agent.prep_rollouts()
        ep_ret = torch.zeros(env.E, env.N, device=dev)
        for _ in range(cfg.episode_length):
            agent.observe(obs)
            acts = agent.act(obs, explore=True)
            nobs, rew, done, info = env.step(acts)
            buf.push(obs, acts, rew, nobs)
            ep_ret += rew
            obs, steps = nobs, steps + env.E
            if len(buf) >= cfg.batch_size and (steps % cfg.steps_per_update) < env.E:
                agent.prep_training()
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                s = time.time()
                for _ in range(cfg.num_updates):
                    last = agent.update(buf.sample(cfg.batch_size))
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                upd_t, upd_n = upd_t + (time.time() - s) * 1000 / cfg.num_updates, upd_n + 1
                agent.prep_rollouts()
        if (ep // cfg.n_envs) % cfg.log_interval == 0:
            r = ep_ret.mean().item()
            vals = {k: float(v) for k, v in last.items()}
            row = [ep + cfg.n_envs, steps, r, vals.get("q_loss", float("nan")), vals.get("pol_loss", float("nan")),
                   vals.get("entropy", float("nan")), upd_t / max(upd_n, 1), time.time() - t0,
                   torch.cuda.max_memory_allocated() / 1e6 if dev.type == "cuda" else float("nan")]
            w.writerow(row); f.flush()
            print(f"ep {ep + cfg.n_envs:6d} | return {r:9.3f} | q_loss {row[3]:.4f} | ent {row[5]:.3f} | "
                  f"upd {row[6]:.1f} ms | {row[7]:.0f}s")
            if tb:
                tb.add_scalar("train/return", r, ep); tb.add_scalar("train/update_ms", row[6], ep)
        if cfg.eval_every and (ep // cfg.n_envs) % cfg.eval_every == 0 and ep > 0:
            m = evaluate(env, agent, n_eval)
            m.update(evaluate_vs_random(env, agent, n_eval))
            log_eval(m, ep + cfg.n_envs)
            print("  eval:", {k: round(v, 3) for k, v in m.items()})
        if cfg.save_interval and (ep // cfg.n_envs) % cfg.save_interval == 0 and ep > 0:
            agent.save(run_dir / "model.pt")
    agent.save(run_dir / "model.pt")
    final = evaluate(env, agent, n_eval, episode_values=True)
    eps = final.pop("final_coverage_episodes", None)  # kept in the JSON, not in eval.csv
    final.update(evaluate_vs_random(env, agent, n_eval))
    log_eval(final, cfg.n_episodes)
    f.close(); ef.close()
    print("final eval:", {k: round(v, 3) for k, v in final.items()})
    if eps is not None:
        final["final_coverage_episodes"] = eps
    json.dump(final, open(run_dir / "final_eval.json", "w"), indent=1)
    return final


if __name__ == "__main__":
    main()
