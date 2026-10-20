"""Policy and attention critic.

Parameter sharing is given by `ranges`, a list of (start, end) agent slices with one module per slice: one slice
per agent type shares networks within the type, N single-agent slices give every agent its own networks as in the
original MAAC."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import flash_attention, loop_attention, mean_field_attention, sparse_attention


class TypedModule(nn.Module):
    def __init__(self, ranges, factory):
        super().__init__()
        self.ranges = list(ranges)
        self.mods = nn.ModuleList([factory() for _ in self.ranges])

    def forward(self, x):  # x: (B,N,d)
        outs = []
        for (s, e), m in zip(self.ranges, self.mods):
            xs = x[:, s:e]
            B, n, d = xs.shape
            outs.append(m(xs.reshape(B * n, d)).reshape(B, n, -1))
        return torch.cat(outs, 1)


def enc_block(din, dout, use_bn=True):
    pre = [nn.BatchNorm1d(din, affine=False)] if use_bn else []
    return nn.Sequential(*pre, nn.Linear(din, dout), nn.LeakyReLU())


class RunningNorm(nn.Module):
    """Observation normalizer with running mean and variance, updated from rollouts and held fixed during updates.
    BatchNorm in train mode would mix the statistics of all agents in the batch, agent i's own action included,
    into everyone's embedding, and Q_i would then depend on a_i."""

    def __init__(self, dim, clip=10.0, eps=1e-6):
        super().__init__()
        self.clip, self.eps = clip, eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, self.mean.numel()).float()
        b = x.shape[0]
        bm, bv = x.mean(0), x.var(0, unbiased=False)
        delta, tot = bm - self.mean, self.count + b
        m2 = self.var * self.count + bv * b + delta ** 2 * self.count * b / tot
        self.mean += delta * b / tot
        self.var.copy_(m2 / tot)
        self.count.copy_(tot)

    def forward(self, x):
        return ((x - self.mean) / (self.var + self.eps).sqrt()).clamp(-self.clip, self.clip)


class Policy(nn.Module):
    def __init__(self, obs_dim, n_actions, ranges, hidden=128, use_bn=True):
        super().__init__()
        self.net = TypedModule(ranges, lambda: nn.Sequential(
            *([nn.BatchNorm1d(obs_dim, affine=False)] if use_bn else []),
            nn.Linear(obs_dim, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, n_actions)))

    def forward(self, obs):
        return self.net(obs)


def sample_actions(logits, explore=True):
    """-> one-hot actions (B,N,A), log_pi of chosen action (B,N,1), probs (B,N,A)."""
    logp_all = F.log_softmax(logits.float(), -1)
    probs = logp_all.exp()
    if explore:
        a = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).view(probs.shape[:-1])
    else:
        a = probs.argmax(-1)
    onehot = F.one_hot(a, probs.shape[-1]).to(probs.dtype)
    return onehot, logp_all.gather(-1, a[..., None]), probs


class AttentionCritic(nn.Module):
    """MAAC critic. The query comes from the state encoding, keys and values from the state-action encoding, and
    the head returns Q for every own action, so the counterfactual baseline needs one pass. Routing (sparse modes)
    reads only observations and runs under no_grad; actions never influence whom an agent attends to."""

    def __init__(self, obs_dim, n_actions, ranges, attn="dense_flash", hidden=128, heads=4,
                 router=None, route_feat="pos", pos_slice=(2, 4), attn_reg=0.0, use_bn=True,
                 sparse_agg="gather", block_impl="sdpa"):
        super().__init__()
        self.sparse_agg, self.block_impl = sparse_agg, block_impl
        assert hidden % heads == 0
        self.attn, self.H, self.dh = attn, heads, hidden // heads
        self.router, self.route_feat, self.pos_slice, self.attn_reg = router, route_feat, pos_slice, attn_reg
        self.sa_enc = TypedModule(ranges, lambda: enc_block(obs_dim + n_actions, hidden, use_bn))
        self.s_enc = TypedModule(ranges, lambda: enc_block(obs_dim, hidden, use_bn))
        self.head = TypedModule(ranges, lambda: nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.LeakyReLU(), nn.Linear(hidden, n_actions)))
        if attn != "meanfield":                 # the mean-field baseline weighs neighbours equally: no queries, no keys
            self.W_q = nn.Linear(hidden, hidden, bias=False)
            self.W_k = nn.Linear(hidden, hidden, bias=False)
        self.W_v = nn.Sequential(nn.Linear(hidden, hidden), nn.LeakyReLU())
        if attn not in ("dense_loop", "dense_flash", "none") and router is None:
            raise ValueError("sparse attention modes need a router")

    def shared_parameters(self):
        """Parameters every agent's Q_i depends on through the attention (MAAC `shared_modules`: key, selector and
        value extractors + state-action encoders). `--loss_reduce sum` scales their grads by 1/N like
        MAAC's `scale_shared_grads`."""
        for name in ("W_q", "W_k", "W_v", "sa_enc"):
            if hasattr(self, name):
                yield from getattr(self, name).parameters()

    def route(self, obs):
        """Precompute routing from observations only (cacheable across online/target/policy passes)."""
        if self.router is None or self.route_feat != "pos":
            return None
        with torch.no_grad():
            return self.router(obs[..., self.pos_slice[0]:self.pos_slice[1]].float())

    def _extract_loop(self, s, sa):
        """Original MAAC extraction: one small matmul per (head, agent) for selectors, keys and values
        (`[[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]`). W_q/W_k/W_v hold the H
        per-head extractors stacked row-wise, so this is numerically identical to the batched path; only the
        Python/kernel-launch structure differs (3*H*N small ops)."""
        B, N, _ = s.shape
        dh, lin_v = self.dh, self.W_v[0]
        qs, ks, vs = [], [], []
        for h in range(self.H):
            r = slice(h * dh, (h + 1) * dh)
            wq, wk, wv, bv = self.W_q.weight[r], self.W_k.weight[r], lin_v.weight[r], lin_v.bias[r]
            qs.append(torch.stack([F.linear(s[:, j], wq) for j in range(N)], 1))
            ks.append(torch.stack([F.linear(sa[:, j], wk) for j in range(N)], 1))
            vs.append(torch.stack([F.leaky_relu(F.linear(sa[:, j], wv, bv)) for j in range(N)], 1))
        return torch.stack(qs, 2), torch.stack(ks, 2), torch.stack(vs, 2)   # (B,N,H,dh) each

    def attend(self, obs, acts, routing=None):
        B, N, _ = obs.shape
        s = self.s_enc(obs)
        if self.attn == "none":     # critic without attention: Q_i(o_i, .) sees no other agent
            return s, s.new_zeros(s.shape), s.new_zeros(())
        sa = self.sa_enc(torch.cat([obs, acts], -1))
        if self.attn == "meanfield":
            v = self.W_v(sa).view(B, N, self.H, self.dh)
            if routing is None:
                with torch.no_grad():
                    feat = s.detach() if self.route_feat == "emb" else obs[..., self.pos_slice[0]:self.pos_slice[1]]
                    routing = self.router(feat.float())
            x, reg = mean_field_attention(v, routing, self.sparse_agg)
            return s, x, reg
        if self.attn == "dense_loop":
            q, k, v = self._extract_loop(s, sa)
        else:
            q = self.W_q(s).view(B, N, self.H, self.dh)
            k = self.W_k(sa).view(B, N, self.H, self.dh)
            v = self.W_v(sa).view(B, N, self.H, self.dh)
        need_reg = self.attn_reg > 0 and self.training
        if self.attn == "dense_loop":
            x, reg = loop_attention(q, k, v, need_reg)
        elif self.attn == "dense_flash":
            x, reg = flash_attention(q, k, v, need_reg)
        else:
            if routing is None:
                with torch.no_grad():
                    feat = s.detach() if self.route_feat == "emb" else obs[..., self.pos_slice[0]:self.pos_slice[1]]
                    routing = self.router(feat.float())
            x, reg = sparse_attention(q, k, v, routing, need_reg, self.sparse_agg, self.block_impl)
        return s, x, reg

    def forward(self, obs, acts, routing=None):
        """-> q (B,N,1) of the taken action, q_all (B,N,A), attention regulariser."""
        s, x, reg = self.attend(obs, acts, routing)
        q_all = self.head(torch.cat([s, x], -1))
        return (q_all * acts).sum(-1, keepdim=True), q_all, reg
