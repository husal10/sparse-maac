"""Small numerical helpers: soft updates, batched kNN (exact and Z-order), etc."""
from __future__ import annotations

import torch

NEG = -1e9


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.lerp_(sp, tau)


def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def batch_gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x: (B, M, ...) ; idx: (B, ...) long -> x[b, idx[b, ...]] with shape (B, *idx.shape[1:], ...)."""
    B = x.shape[0]
    view = (B,) + (1,) * (idx.dim() - 1)
    return x[torch.arange(B, device=x.device).view(view), idx]


@torch.no_grad()
def nearest_k(query, keys, k, exclude_self=False, chunk=1024):
    """Exact k nearest keys for each query. query (B,Q,d), keys (B,M,d).
    Chunked over queries, so memory is O(B*chunk*M). Returns (dist, idx), each (B,Q,k).
    With exclude_self, query[i] and keys[i] are the same entity and are excluded."""
    B, Q, _ = query.shape
    M = keys.shape[1]
    k = min(k, M - 1 if exclude_self else M)
    ds, ids = [], []
    for s in range(0, Q, chunk):
        q = query[:, s:s + chunk]
        d = torch.cdist(q, keys, compute_mode="donot_use_mm_for_euclid_dist")
        if exclude_self:
            rows = torch.arange(q.shape[1], device=q.device)
            d[:, rows, rows + s] = float("inf")
        dist, idx = d.topk(k, dim=-1, largest=False)
        ds.append(dist)
        ids.append(idx)
    return torch.cat(ds, 1), torch.cat(ids, 1)


def _part1by1(x):
    x = (x | (x << 8)) & 0x00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F
    x = (x | (x << 2)) & 0x33333333
    x = (x | (x << 1)) & 0x55555555
    return x


def morton2d(pos, lo, hi, bits=16):
    """Z-order (Morton) code for 2-D positions (..., 2) -> (...,) int64."""
    q = ((pos - lo) / (hi - lo)).clamp(0, 1 - 1e-6) * (2 ** bits)
    q = q.long()
    return _part1by1(q[..., 0]) | (_part1by1(q[..., 1]) << 1)


@torch.no_grad()
def knn_zorder(pos, k, window=None, shifts=2):
    """Approximate kNN in 2-D in O(N log N + N*window*shifts).
    Sort agents along `shifts` randomly shifted Z-order curves, take +-window
    curve-neighbours as candidates, and keep the true k nearest of them.
    Returns idx (B,N,k) and a validity mask (B,N,k)."""
    B, N, _ = pos.shape
    k = min(k, N - 1)
    W = window or max(k, 8)
    dev = pos.device
    lo = pos.amin(dim=(0, 1))
    span = (pos.amax(dim=(0, 1)) - lo).clamp(min=1e-6)
    ar = torch.arange(N, device=dev)
    offs = torch.cat([torch.arange(-W, 0), torch.arange(1, W + 1)]).to(dev)
    cands = []
    for s in range(shifts):
        shift = torch.zeros(2, device=dev) if s == 0 else torch.rand(2, device=dev) * 0.5 * span
        code = morton2d(pos + shift, lo, lo + 1.5 * span)
        order = code.argsort(dim=1)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, ar.expand(B, N).contiguous())
        cr = (rank[:, :, None] + offs).clamp(0, N - 1)
        cands.append(order.gather(1, cr.reshape(B, -1)).reshape(B, N, -1))
    cand, _ = torch.cat(cands, -1).sort(-1)
    dup = torch.zeros_like(cand, dtype=torch.bool)
    dup[..., 1:] = cand[..., 1:] == cand[..., :-1]
    selfm = cand == ar[None, :, None]
    cp = batch_gather(pos, cand)  # (B,N,C,2)
    d = (cp - pos[:, :, None]).norm(dim=-1).masked_fill(dup | selfm, float("inf"))
    dist, ti = d.topk(k, dim=-1, largest=False)
    return cand.gather(-1, ti), torch.isfinite(dist)
