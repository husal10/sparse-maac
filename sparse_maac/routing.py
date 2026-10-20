"""Routers decide whom each agent attends to.

A router reads detached float features (positions, or detached state embeddings) and never sees actions or
gradients; that is what keeps the one-pass counterfactual baseline exact. It returns a `Routing`:
  * gather part  : idx/mask (B,N,K)   -> attend to the individual agents idx[b,i,:]
  * centroid part: assign (B,N), c    -> attend to c cluster means (own cluster without the agent itself)
  * block part   : perm, block        -> Z-MAAC's near-far blocks along the Z-order curve
Gather and centroid parts can be combined (hybrid); they share one softmax.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .utils import knn_zorder, morton2d, nearest_k


@dataclass
class Routing:
    idx: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    assign: Optional[torch.Tensor] = None
    n_clusters: int = 0
    perm: Optional[torch.Tensor] = None   # block mode: (B,N) agent order along the space-filling curve
    block: int = 0                        # block mode: block size b
    use_global: bool = True               # block mode: add far-field block-centroid tokens


@torch.no_grad()
def batched_kmeans(x, c, iters=5):
    """Lloyd's algorithm on GPU, batched over B. x: (B,N,d) -> assign (B,N), centroids (B,c,d)."""
    B, N, d = x.shape
    c = min(c, N)
    init = torch.rand(B, N, device=x.device).topk(c, dim=1).indices
    cent = x.gather(1, init[..., None].expand(-1, -1, d))
    ones = torch.ones(B, N, device=x.device)
    for _ in range(iters):
        assign = torch.cdist(x, cent).argmin(-1)
        sums = torch.zeros(B, c, d, device=x.device).scatter_add_(1, assign[..., None].expand(-1, -1, d), x)
        cnt = torch.zeros(B, c, device=x.device).scatter_add_(1, assign, ones)
        cent = torch.where(cnt[..., None] > 0, sums / cnt.clamp(min=1)[..., None], cent)
    return torch.cdist(x, cent).argmin(-1), cent


@torch.no_grad()
def members_from_assign(assign, c, balanced=True):
    """Turn a cluster assignment (B,N) into per-agent member lists (B,N,M) + mask (self excluded).
    balanced=True : sort agents by cluster id and cut into equal blocks of M=ceil(N/c)
                    (keeps cost O(N^2/c) even if k-means is unbalanced; Reformer-style chunking).
    balanced=False: exact clusters, padded to the largest cluster (worst case O(N^2))."""
    B, N = assign.shape
    dev = assign.device
    order = torch.argsort(assign, dim=1, stable=True)
    sorted_assign = assign.gather(1, order)
    pos = torch.arange(N, device=dev).expand(B, N)
    if balanced:
        M = math.ceil(N / c)
        gid, rank, G = pos // M, pos % M, math.ceil(N / M)
    else:
        counts = torch.zeros(B, c, dtype=torch.long, device=dev).scatter_add_(1, assign, torch.ones_like(assign))
        starts = counts.cumsum(1) - counts
        rank = pos - starts.gather(1, sorted_assign)
        gid, M, G = sorted_assign, int(counts.max()), c
    groups = torch.full((B, G * M), -1, dtype=torch.long, device=dev)
    groups.scatter_(1, gid * M + rank, order)
    agent_gid = torch.zeros(B, N, dtype=torch.long, device=dev).scatter_(1, order, gid)
    idx = groups.view(B, G, M).gather(1, agent_gid[..., None].expand(-1, -1, M))
    mask = (idx >= 0) & (idx != torch.arange(N, device=dev)[None, :, None])
    return idx.clamp(min=0), mask


@torch.no_grad()
def dedup(idx, mask):
    """Mask repeated agent indices so a token set never contains the same agent twice.
    Needed when token lists from several sources are concatenated (LSH rounds, kNN + random in hybrid):
    a duplicate would silently get double weight in the joint softmax. Order of tokens is irrelevant."""
    s = idx.masked_fill(~mask, -1).sort(-1).values
    keep = s >= 0
    keep[..., 1:] &= s[..., 1:] != s[..., :-1]
    return s.clamp(min=0), keep


class RandomRouter:
    """K other agents drawn uniformly with replacement, fresh for every sample. Repeats are kept on purpose: with
    i.i.d. draws the softmax over the sample is the self-normalised Monte Carlo estimate of full attention."""
    def __init__(self, k):
        self.k = k

    @torch.no_grad()
    def __call__(self, feat):
        B, N, _ = feat.shape
        K = min(self.k, N - 1)
        r = torch.randint(0, N - 1, (B, N, K), device=feat.device)
        idx = r + (r >= torch.arange(N, device=feat.device)[None, :, None]).long()
        return Routing(idx=idx, mask=torch.ones_like(idx, dtype=torch.bool))


class KNNRouter:
    """The k nearest agents. backend: exact (chunked cdist), zorder (approximate, near-linear) or auto."""
    def __init__(self, k, backend="auto", window=None):
        self.k, self.backend, self.window = k, backend, window

    @torch.no_grad()
    def __call__(self, feat):
        B, N, d = feat.shape
        backend = self.backend
        if backend == "auto":
            backend = "zorder" if (d == 2 and N > 512) else "exact"
        if backend == "zorder":
            if d != 2:
                raise ValueError("zorder kNN needs 2-D features; use backend='exact'")
            idx, mask = knn_zorder(feat, self.k, self.window)
        else:
            _, idx = nearest_k(feat, feat, self.k, exclude_self=True)
            mask = torch.ones_like(idx, dtype=torch.bool)
        return Routing(idx=idx, mask=mask)


class ClusterRouter:
    """Only the other members of the agent's own k-means cluster."""
    def __init__(self, n_clusters, iters=5, balanced=True):
        self.c, self.iters, self.balanced = n_clusters, iters, balanced

    @torch.no_grad()
    def __call__(self, feat):
        assign, _ = batched_kmeans(feat, self.c, self.iters)
        idx, mask = members_from_assign(assign, min(self.c, feat.shape[1]), self.balanced)
        return Routing(idx=idx, mask=mask)


class CentroidRouter:
    """C-MAAC: one summary per k-means cluster of positions, the agent's own cluster without the agent."""
    def __init__(self, n_clusters, iters=5):
        self.c, self.iters = n_clusters, iters

    @torch.no_grad()
    def __call__(self, feat):
        c = min(self.c, feat.shape[1])
        assign, _ = batched_kmeans(feat, c, self.iters)
        return Routing(assign=assign, n_clusters=c)


class LSHRouter:
    """Random-projection LSH with `rounds` independent hashes, sorted and chunked as in Reformer.
    Each hash projects features on n_hash random directions, quantises each projection into a grid
    cell, sorts agents by the compound code and cuts the order into blocks of `bucket_size`.
    Tokens from all rounds are concatenated; agents found by several rounds are kept once (dedup)."""
    def __init__(self, bucket_size, rounds=2, n_hash=3):
        self.m, self.rounds, self.n_hash = bucket_size, rounds, n_hash

    @torch.no_grad()
    def __call__(self, feat):
        B, N, d = feat.shape
        nb = max(1, math.ceil(N / self.m))
        g = max(2, math.ceil(nb ** (1.0 / self.n_hash)))
        x = feat - feat.mean(1, keepdim=True)
        idxs, masks = [], []
        for _ in range(self.rounds):
            proj = x @ torch.randn(d, self.n_hash, device=feat.device)             # (B,N,h)
            lo, hi = proj.amin(1, keepdim=True), proj.amax(1, keepdim=True)
            u = (proj - lo) / (hi - lo).clamp(min=1e-6) + torch.rand(1, 1, self.n_hash, device=feat.device) / g
            cell = (u * g).long().clamp(0, g)                                       # (B,N,h)
            code = torch.zeros(B, N, dtype=torch.long, device=feat.device)
            for j in range(self.n_hash):
                code = code * (g + 1) + cell[..., j]
            idx, mask = members_from_assign(code, nb, balanced=True)
            idxs.append(idx)
            masks.append(mask)
        idx, mask = dedup(torch.cat(idxs, -1), torch.cat(masks, -1))
        return Routing(idx=idx, mask=mask)


class HybridRouter:
    """kNN + random + cluster summaries in one softmax. Agents attended individually are de-duplicated but also
    sit inside the summaries, so they are counted twice; `block` partitions the agents and avoids this."""
    def __init__(self, knn: KNNRouter, rand: RandomRouter, cent: CentroidRouter):
        self.knn, self.rand, self.cent = knn, rand, cent

    @torch.no_grad()
    def __call__(self, feat):
        a, b, c = self.knn(feat), self.rand(feat), self.cent(feat)
        idx, mask = dedup(torch.cat([a.idx, b.idx], -1), torch.cat([a.mask, b.mask], -1))
        return Routing(idx=idx, mask=mask, assign=c.assign, n_clusters=c.n_clusters)


class BlockRouter:
    """Z-MAAC's block near-far routing, in the spirit of Barnes-Hut.
    Sort agents along a Z-order curve of their positions (deterministic, per-sample, state-only), cut the
    order into blocks of b. Near field = own block + the two adjacent blocks (exact, individual agents);
    far field = one centroid token per remaining block (logit + log n_b). Static shapes, no k-means,
    no data-dependent worst case: cost O(N (3b + N/b)), minimised at b ~ sqrt(N/3).
    For d != 2 features the curve is replaced by a fixed random-projection lexicographic code."""
    def __init__(self, block_size, use_global=True, n_hash=3, seed=0):
        self.b, self.use_global, self.n_hash, self.seed, self._proj = block_size, use_global, n_hash, seed, None

    @torch.no_grad()
    def __call__(self, feat):
        B, N, d = feat.shape
        lo = feat.amin(1, keepdim=True)
        span = (feat.amax(1, keepdim=True) - lo).clamp(min=1e-6)
        if d == 2:
            code = morton2d(feat, lo, lo + span)
        else:
            if self._proj is None or self._proj.shape[0] != d:
                g = torch.Generator().manual_seed(self.seed)
                self._proj = torch.randn(d, self.n_hash, generator=g)
            u = ((feat - lo) / span - 0.5) @ self._proj.to(feat.device)
            u = (u - u.amin(1, keepdim=True)) / (u.amax(1, keepdim=True) - u.amin(1, keepdim=True)).clamp(min=1e-6)
            cell = (u * 15).long().clamp(0, 15)
            code = torch.zeros(B, N, dtype=torch.long, device=feat.device)
            for j in range(self.n_hash):
                code = code * 16 + cell[..., j]
        return Routing(perm=code.argsort(dim=1, stable=True), block=min(self.b, N), use_global=self.use_global)


def anneal_schedule(n_agents, n0, n1=None):
    """Block size of the annealed critic, which moves from FlashMAAC to Z-MAAC as the number of agents grows.

        t(N) = clip( log(N / n0) / log(n1 / n0), 0, 1 )                       (n1 defaults to 4 n0)
        log b(N) = (1 - t) log N + t log b_sparse(N),   b_sparse(N) = round(sqrt(N / 3))

    Up to n0 there is one block holding every agent, which is FlashMAAC; from n1 on it is Z-MAAC. In between the near
    field (3b agents) shrinks and the far field grows, with no switch. While b >= N/2 the result equals full attention.
    n0 should be the crossover measured on your GPU and minibatch size (fit_crossover.py). Returns (t, b)."""
    n1 = n1 or 4 * n0
    if n1 <= n0:
        raise ValueError("anneal: need n1 > n0")
    b_sparse = max(1, round(math.sqrt(n_agents / 3)))
    t = min(1.0, max(0.0, math.log(n_agents / n0) / math.log(n1 / n0))) if n_agents > 0 else 0.0
    if t <= 0.0:
        return 0.0, n_agents
    if t >= 1.0:
        return 1.0, b_sparse
    b = round(math.exp((1.0 - t) * math.log(n_agents) + t * math.log(b_sparse)))
    return t, int(min(n_agents, max(b_sparse, b)))


def make_router(name, n_agents, cfg):
    s = max(2, math.ceil(math.sqrt(n_agents)))
    k = cfg.knn_k or s
    rk = cfg.rand_k or s
    c = cfg.n_clusters or s
    if name == "random":
        return RandomRouter(rk)
    if name in ("knn", "meanfield"):        # meanfield: same neighbour set as knn, equal weights (attention.py)
        return KNNRouter(k, cfg.knn_backend)
    if name == "cluster":
        return ClusterRouter(c, cfg.kmeans_iters, cfg.balanced)
    if name == "centroid":
        return CentroidRouter(c, cfg.kmeans_iters)
    if name == "lsh":
        return LSHRouter(cfg.lsh_bucket or s, cfg.lsh_rounds)
    if name == "hybrid":
        return HybridRouter(KNNRouter(k, cfg.knn_backend), RandomRouter(rk), CentroidRouter(c, cfg.kmeans_iters))
    if name == "block":
        b = getattr(cfg, "block_size", None) or max(1, round(math.sqrt(n_agents / 3)))
        return BlockRouter(b, getattr(cfg, "block_global", True))
    if name == "anneal":                    # block router with the annealed block size (anneal_schedule)
        _, b = anneal_schedule(n_agents, getattr(cfg, "anneal_n0", 900), getattr(cfg, "anneal_n1", None))
        return BlockRouter(b, getattr(cfg, "block_global", True))
    raise ValueError(f"unknown router {name}")
