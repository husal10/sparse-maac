"""Discrete multi-agent soft actor-critic with an attention critic, following MAAC's update rules
(Iqbal & Sha, update_critic / update_policies).

  * --loss_reduce sum reproduces the original objective: per-agent losses summed, shared critic gradients scaled
    by 1/N (scale_shared_grads), critic clip 10*N, policy clip 0.5 per agent. One Adam over all parameters equals
    MAAC's per-agent Adams because Adam is element-wise.
  * --loss_reduce mean (default): the same losses averaged over agents, critic clip 10 and one policy clip 0.5, so
    the step size does not grow with N.
  * The critic pass inside the policy loss runs under no_grad. MAAC detaches it as well; this only skips building
    a graph nobody uses.
  * Routing is computed once per state batch and shared by the online critic, the target critic and the policy
    pass. This is valid because routers never look at actions."""
from __future__ import annotations

import copy
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.optim import Adam

from .networks import AttentionCritic, Policy, RunningNorm, sample_actions
from .routing import make_router
from .utils import hard_update, soft_update


class SparseMAAC:
    def __init__(self, spec, cfg, device):
        self.spec, self.cfg, self.dev = spec, cfg, torch.device(device)
        N, D, A = spec["n_agents"], spec["obs_dim"], spec["n_actions"]
        sharing = cfg.param_sharing
        if sharing == "auto":
            sharing = "agent" if cfg.attn == "dense_loop" else "type"
        ranges = [(i, i + 1) for i in range(N)] if sharing == "agent" else spec["ranges"]
        router = None if cfg.attn in ("dense_loop", "dense_flash", "none") else make_router(cfg.attn, N, cfg)
        use_bn = getattr(cfg, "obs_norm", "running") == "bn"
        self.obs_norm = RunningNorm(D).to(self.dev) if not use_bn else None
        self.policy = Policy(D, A, ranges, cfg.pol_hidden, use_bn).to(self.dev)
        self.critic = AttentionCritic(D, A, ranges, cfg.attn, cfg.critic_hidden, cfg.heads, router,
                                      cfg.route_feat, spec["pos_slice"], cfg.attn_reg, use_bn,
                                      getattr(cfg, "sparse_agg", "gather"), getattr(cfg, "block_impl", "sdpa")).to(self.dev)
        self.target_policy, self.target_critic = copy.deepcopy(self.policy), copy.deepcopy(self.critic)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.pi_opt = Adam(self.policy.parameters(), lr=cfg.pi_lr)
        self.q_opt = Adam(self.critic.parameters(), lr=cfg.q_lr, weight_decay=1e-3)
        self.n_actions = A
        self.niter = 0

    def _ac(self):
        if self.cfg.amp:
            return torch.autocast(device_type=self.dev.type, dtype=torch.bfloat16)
        return nullcontext()

    def norm(self, obs):
        return self.obs_norm(obs) if self.obs_norm is not None else obs

    def observe(self, obs):
        """Feed raw rollout observations to the running normaliser (call before act())."""
        if self.obs_norm is not None:
            self.obs_norm.update(obs)

    def prep_rollouts(self):
        self.policy.eval()

    def prep_training(self):
        for m in (self.policy, self.target_policy, self.critic, self.target_critic):
            m.train()

    @torch.no_grad()
    def act(self, obs, explore=True):
        """obs (E,N,D) -> action indices (E,N)."""
        with self._ac():
            logits = self.policy(self.norm(obs))
        onehot, _, _ = sample_actions(logits, explore)
        return onehot.argmax(-1)

    def update(self, batch):
        cfg = self.cfg
        raw_obs, acs_idx, rews, raw_next = batch
        obs, next_obs = self.norm(raw_obs), self.norm(raw_next)
        acs = F.one_hot(acs_idx, self.n_actions).float()
        # critic: soft TD target from the target policy and target critic
        with torch.no_grad(), self._ac():
            n_ac, n_logp, _ = sample_actions(self.target_policy(next_obs))
            nq, _, _ = self.target_critic(next_obs, n_ac, self.target_critic.route(raw_next))
            target = rews + cfg.gamma * nq.float() - n_logp / cfg.reward_scale
        route = self.critic.route(raw_obs)                      # state-only, shared by both critic passes
        with self._ac():
            q, _, reg = self.critic(obs, acs, route)
        q_loss = F.mse_loss(q.float(), target) + cfg.attn_reg * reg.float()   # mean over agents
        summed = getattr(cfg, "loss_reduce", "mean") == "sum"
        N = q.shape[1]
        self.q_opt.zero_grad(set_to_none=True)
        (q_loss * N if summed else q_loss).backward()
        if summed:                                              # MAAC scale_shared_grads
            for p in self.critic.shared_parameters():
                if p.grad is not None:
                    p.grad.mul_(1.0 / N)
        q_gn = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0 * N if summed else 10.0)
        self.q_opt.step()
        # policy: soft policy gradient with the counterfactual baseline
        with self._ac():
            logits = self.policy(obs)
        ac, logp, probs = sample_actions(logits)
        with torch.no_grad(), self._ac():
            qs, q_all, _ = self.critic(obs, ac, route)
            v = (q_all.float() * probs).sum(-1, keepdim=True)
            adv = qs.float() - v
        pol_loss = (logp * (logp / cfg.reward_scale - adv).detach()).mean() + 1e-3 * (logits.float() ** 2).mean()
        self.pi_opt.zero_grad(set_to_none=True)
        (pol_loss * N if summed else pol_loss).backward()
        if summed:                                              # MAAC: one clip_grad_norm(0.5) per agent policy
            pi_gn = torch.stack([torch.nn.utils.clip_grad_norm_(m.parameters(), 0.5)
                                 for m in self.policy.net.mods]).norm()
        else:
            pi_gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.pi_opt.step()
        soft_update(self.target_critic, self.critic, cfg.tau)
        soft_update(self.target_policy, self.policy, cfg.tau)
        self.niter += 1
        ent = -(probs * probs.clamp(min=1e-8).log()).sum(-1).mean()
        return {"q_loss": q_loss.detach(), "pol_loss": pol_loss.detach(), "q_grad_norm": q_gn.detach(),
                "pi_grad_norm": pi_gn.detach(), "entropy": ent.detach()}

    def save(self, path):
        torch.save({"cfg": vars(self.cfg), "spec": self.spec, "policy": self.policy.state_dict(),
                    "obs_norm": self.obs_norm.state_dict() if self.obs_norm is not None else None,
                    "critic": self.critic.state_dict(), "target_critic": self.target_critic.state_dict(),
                    "target_policy": self.target_policy.state_dict()}, path)

    def load(self, path):
        d = torch.load(path, map_location=self.dev)
        for k in ("policy", "critic", "target_critic", "target_policy"):
            getattr(self, k).load_state_dict(d[k])
        if self.obs_norm is not None and d.get("obs_norm") is not None:
            self.obs_norm.load_state_dict(d["obs_norm"])
