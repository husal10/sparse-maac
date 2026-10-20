"""Attention backends. Each takes q, k, v of shape (B,N,H,dh) and returns (out (B,N,H*dh), reg).

  loop_attention        MAAC's original Python loops over heads and agents (`dense_loop`)
  flash_attention       all agents at once in one fused SDPA call, own entry masked (`dense_flash`, FlashMAAC)
  sparse_attention      individually attended agents and/or group summaries under one softmax (routers)
  mean_field_attention  equal weights over the neighbours, no queries or keys (`meanfield`)

Two options change speed and memory but not the result (the tests check this): agg="ebag" sums the attended values
with one F.embedding_bag call instead of gathering a (B,N,K,H,dh) copy, and block_impl="flex" runs Z-MAAC as one
FlexAttention call with a static block-sparse mask instead of padded dense blocks.

`reg` is MAAC's attention-logit penalty in the original units: the mean over agents of the sum over heads of the mean
squared raw dot product (q . k, not divided by sqrt(dh)), so `--attn_reg 1e-3` matches MAAC's
`1e-3 * attend_mag_reg`. The log n_g bias of a summary is not part of the dot product and is left out."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .routing import Routing
from .utils import NEG


def loop_attention(q, k, v, need_reg=False):
    B, N, H, dh = q.shape
    outs, reg = [], q.new_zeros(())
    for h in range(H):
        per_agent = []
        for i in range(N):
            keys = torch.cat([k[:, :i, h], k[:, i + 1:, h]], 1)       # (B,N-1,dh)
            vals = torch.cat([v[:, :i, h], v[:, i + 1:, h]], 1)
            logits = torch.matmul(q[:, i, h].unsqueeze(1), keys.transpose(1, 2)) / math.sqrt(dh)
            w = F.softmax(logits, dim=2)
            per_agent.append((vals.transpose(1, 2) * w).sum(2))      # (B,dh)
            if need_reg:
                reg = reg + ((logits * math.sqrt(dh)) ** 2).mean()   # raw (unscaled) logits, as in MAAC
        outs.append(torch.stack(per_agent, 1))
    return torch.cat(outs, -1), reg / N


def flash_attention(q, k, v, need_reg=False):
    B, N, H, dh = q.shape
    qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))               # (B,H,N,dh)
    eye = torch.eye(N, dtype=torch.bool, device=q.device)
    if need_reg:   # explicit path: SDPA does not expose logits
        logits = qh @ kh.transpose(-1, -2) / math.sqrt(dh)
        w = torch.softmax(logits.masked_fill(eye, NEG).float(), -1).to(vh.dtype)
        out = w @ vh
        reg = ((logits.masked_fill(eye, 0.0).float() * math.sqrt(dh)) ** 2).sum() / (B * N * max(N - 1, 1))
    else:
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=~eye)
        reg = q.new_zeros(())
    return out.transpose(1, 2).reshape(B, N, H * dh), reg


def _centroid_part(q, k, v, assign, c, scale):
    """One summary per cluster. The agent's own cluster is summarized without it, and every summary's logit gets
    + log(n_c), so it weighs as much as its n_c members would if they shared the mean key."""
    B, N, H, dh = q.shape
    a = assign[:, :, None].expand(-1, -1, H * dh)
    Sk = k.new_zeros(B, c, H * dh).scatter_add(1, a, k.reshape(B, N, -1)).view(B, c, H, dh)
    Sv = v.new_zeros(B, c, H * dh).scatter_add(1, a, v.reshape(B, N, -1)).view(B, c, H, dh)
    n = torch.zeros(B, c, device=q.device).scatter_add_(1, assign, torch.ones(B, N, device=q.device))
    own = assign[:, :, None]                                           # (B,N,1)
    n_eff = n[:, None, :].repeat(1, N, 1)
    n_eff.scatter_add_(2, own, -torch.ones(B, N, 1, device=q.device))  # leave self out of own cluster
    L = torch.einsum("bnhd,bchd->bnhc", q, Sk)
    own_h = own[:, :, None, :].expand(B, N, H, 1)
    L = L.scatter_add(3, own_h, -(q * k).sum(-1, keepdim=True))
    cnt = n_eff.clamp(min=1)[:, :, None, :]                            # (B,N,1,c)
    logits = L.float() / cnt * scale + cnt.log()
    return logits, (n_eff > 0)[:, :, None, :], Sv, cnt, own_h


def ebag_weighted_sum(w, v, idx):
    """out[b,n,h] = sum_k w[b,n,h,k] * v[b, idx[b,n,k], h] in a single F.embedding_bag(mode="sum") call, with the
    values as the table, so gradients reach both v and w and no (B,N,K,H,dh) copy is built."""
    B, N, H, K = w.shape
    dh = v.shape[-1]
    table = v.reshape(B * N * H, dh)
    rows = ((torch.arange(B, device=v.device).view(B, 1, 1, 1) * N + idx[:, :, None, :]) * H
            + torch.arange(H, device=v.device).view(1, 1, H, 1))                  # (B,N,H,K) row of (b, j, h)
    out = F.embedding_bag(rows.reshape(-1, K), table, mode="sum", per_sample_weights=w.reshape(-1, K).to(table.dtype))
    return out.view(B, N, H, dh)


_FLEX = {}


def _flex_attention(device):
    """flex_attention, compiled on CUDA; eager (reference implementation) elsewhere."""
    key = device.type
    if key not in _FLEX:
        from torch.nn.attention.flex_attention import flex_attention
        _FLEX[key] = torch.compile(flex_attention, dynamic=False) if device.type == "cuda" else flex_attention
    return _FLEX[key]


_BLOCK_MASKS = {}


def _flex_ok(*t):
    """FlexAttention has no CPU backward, so on CPU it is used only when no gradient is needed."""
    return t[0].is_cuda or not (torch.is_grad_enabled() and any(x.requires_grad for x in t))


def _near_far_block_mask(N, b, nb, use_global, device):
    """Mask over the Z-order-sorted agents (kv < N) and the block summaries (kv >= N): a query sees the agents of
    its own and the two neighbouring blocks, except itself, and the summary of every other block, with + log n_g.
    It depends only on N and b, so it is built once per N; each minibatch only changes the permutation."""
    from torch.nn.attention.flex_attention import create_block_mask
    key = (N, b, use_global, str(device))
    if key not in _BLOCK_MASKS:
        def mask_mod(bb, h, qi, kv):
            near = (kv < N) & ((qi // b - kv // b).abs() <= 1) & (qi != kv)
            if not use_global:
                return near
            return near | ((kv >= N) & ((qi // b - (kv - N)).abs() > 1))
        bm = create_block_mask(mask_mod, B=None, H=None, Q_LEN=N, KV_LEN=N + (nb if use_global else 0), device=device)
        score_mod = None
        if use_global:
            cnt = torch.full((nb,), float(b), device=device)
            cnt[-1] = N - (nb - 1) * b
            logn = cnt.log()

            def score_mod(score, bb, h, qi, kv):
                return torch.where(kv >= N, score + logn[(kv - N).clamp(min=0)], score)
        _BLOCK_MASKS[key] = (bm, score_mod)
    return _BLOCK_MASKS[key]


def block_attention_flex(q, k, v, routing: Routing):
    """Same function as block_attention (SDPA path), computed by FlexAttention without padding."""
    B, N, H, dh = q.shape
    dev, b = q.device, routing.block
    nb = math.ceil(N / b)
    use_global = routing.use_global and nb > 2
    perm = routing.perm
    srt = lambda t: t.reshape(B, N, -1).gather(1, perm[..., None].expand(-1, -1, H * dh)).view(B, N, H, dh)
    qs, ks, vs = srt(q), srt(k), srt(v)
    if use_global:
        Np = nb * b
        cnt = torch.full((nb,), float(b), device=dev)
        cnt[-1] = N - (nb - 1) * b
        pad = lambda t: torch.cat([t, t.new_zeros(B, Np - N, H, dh)], 1).view(B, nb, b, H, dh)
        Kg = pad(ks).sum(2) / cnt.view(1, nb, 1, 1).to(ks.dtype)                     # (B,nb,H,dh) block means
        Vg = pad(vs).sum(2) / cnt.view(1, nb, 1, 1).to(vs.dtype)
        ks, vs = torch.cat([ks, Kg], 1), torch.cat([vs, Vg], 1)
    bm, score_mod = _near_far_block_mask(N, b, nb, use_global, dev)
    out = _flex_attention(dev)(qs.transpose(1, 2), ks.transpose(1, 2), vs.transpose(1, 2),
                               score_mod=score_mod, block_mask=bm)                     # (B,H,N,dh)
    out = out.transpose(1, 2).reshape(B, N, H * dh)
    inv = perm.argsort(1)
    return out.gather(1, inv[..., None].expand(-1, -1, H * dh)), q.new_zeros(())


def block_attention(q, k, v, routing: Routing, need_reg=False):
    """Z-MAAC attention over the Z-order-sorted agents (see BlockRouter).
    Each block of b agents attends to [previous block | own block | next block | summaries of all other blocks],
    with the agent itself masked. Shapes are static, and keys and values are copied once per block, not per agent."""
    B, N, H, dh = q.shape
    dev, b = q.device, routing.block
    nb = math.ceil(N / b)
    Np, pad = nb * b, nb * b - N
    perm = routing.perm
    perm_p = torch.cat([perm, perm[:, :1].expand(B, pad)], 1) if pad else perm
    vmask = (torch.arange(Np, device=dev) < N).view(1, nb, b)

    def blocks(t):
        return t.reshape(B, N, -1).gather(1, perm_p[..., None].expand(-1, -1, H * dh)).view(B, nb, b, H, dh)

    def window(t, fill):
        z = fill.expand(t.shape[0], 1, *t.shape[2:]) if t.dim() > 3 else fill.expand(t.shape[0], 1, b)
        tp = torch.cat([z, t, z], 1)
        return torch.cat([tp[:, :-2], tp[:, 1:-1], tp[:, 2:]], 2)

    qb, kb, vb = blocks(q), blocks(k), blocks(v)
    zb = kb.new_zeros(1, 1, b, H, dh)
    K, V = window(kb, zb), window(vb, zb)                                       # (B,nb,3b,H,dh)
    vloc = window(vmask, torch.zeros(1, 1, b, dtype=torch.bool, device=dev))    # (1,nb,3b)
    mask = vloc[:, :, None, :].repeat(1, 1, b, 1)                               # (1,nb,b,3b)
    mask[..., b:2 * b] &= ~torch.eye(b, dtype=torch.bool, device=dev)           # no self
    bias = torch.zeros(1, nb, b, 3 * b, device=dev)
    if routing.use_global and nb > 2:
        vm = vmask[..., None, None].to(kb.dtype)
        n = vmask.sum(2).float()                                                # (1,nb)
        cnt = n.clamp(min=1)
        Kg = ((kb * vm).sum(2) / cnt[..., None, None].to(kb.dtype))             # (B,nb,H,dh)
        Vg = ((vb * vm).sum(2) / cnt[..., None, None].to(vb.dtype))
        far = (torch.arange(nb, device=dev)[:, None] - torch.arange(nb, device=dev)[None]).abs() > 1
        gmask = (far[None] & (n > 0)[:, None, :])[:, :, None, :].expand(1, nb, b, nb)
        gbias = cnt.log()[:, None, None, :].expand(1, nb, b, nb)
        K = torch.cat([K, Kg[:, None].expand(B, nb, nb, H, dh)], 2)
        V = torch.cat([V, Vg[:, None].expand(B, nb, nb, H, dh)], 2)
        mask = torch.cat([mask, gmask], -1)
        bias = torch.cat([bias, gbias], -1)
    Kt = K.shape[2]
    qh = qb.permute(0, 1, 3, 2, 4).reshape(B * nb, H, b, dh)
    Kh = K.permute(0, 1, 3, 2, 4).reshape(B * nb, H, Kt, dh)
    Vh = V.permute(0, 1, 3, 2, 4).reshape(B * nb, H, Kt, dh)
    fmask = bias.masked_fill(~mask, NEG).expand(B, nb, b, Kt).reshape(B * nb, 1, b, Kt).to(qh.dtype)
    if need_reg:
        logits = (qh @ Kh.transpose(-1, -2)).float() / math.sqrt(dh) + fmask[:, :1].float()
        out = (torch.softmax(logits, -1).to(Vh.dtype) @ Vh)
        mk = mask.expand(B, nb, b, Kt).reshape(B * nb, 1, b, Kt).expand_as(logits)
        raw = (logits - fmask[:, :1].float()) * math.sqrt(dh)                 # strip the log n_b bias
        qmask = vmask.expand(B, nb, b).reshape(B * nb, 1, b, 1)                # padded query rows do not count
        mk = mk & qmask
        reg = H * (raw.masked_fill(~mk, 0.0) ** 2).sum() / mk.sum().clamp(min=1)
    else:
        out = F.scaled_dot_product_attention(qh, Kh, Vh, attn_mask=fmask)
        reg = q.new_zeros(())
    out = out.view(B, nb, H, b, dh).permute(0, 1, 3, 2, 4).reshape(B, Np, H * dh)[:, :N]
    inv = perm.argsort(1)
    return out.gather(1, inv[..., None].expand(-1, -1, H * dh)), reg


def mean_field_attention(v, routing: Routing, agg="gather"):
    """Mean-field critic in the style of Yang et al. (2018): x_i is the plain average of the values of agent i's k
    nearest neighbours. Yang et al. average the neighbours' actions; averaging V g(o_j, a_j) instead keeps x_i the
    same size as in the other critics, so only the weighting differs."""
    B, N, H, dh = v.shape
    idx, mask = routing.idx, routing.mask
    K = idx.shape[-1]
    w = mask.to(v.dtype)
    w = (w / w.sum(-1, keepdim=True).clamp(min=1))[:, :, None, :].expand(B, N, H, K)     # (B,N,H,K)
    if agg == "ebag":
        out = ebag_weighted_sum(w, v, idx)
    else:
        gv = v.reshape(B, N, -1).gather(1, idx.reshape(B, N * K, 1).expand(-1, -1, H * dh)).view(B, N, K, H, dh)
        out = torch.einsum("bnhk,bnkhd->bnhd", w, gv)
    return out.reshape(B, N, H * dh), v.new_zeros(())


def sparse_attention(q, k, v, routing: Routing, need_reg=False, agg="gather", block_impl="sdpa"):
    if routing.perm is not None:
        if routing.block >= q.shape[1]:     # one block holds every agent: this is full attention
            return flash_attention(q, k, v, need_reg)
        if block_impl == "flex" and not need_reg and _flex_ok(q, k, v):   # the penalty needs explicit logits
            return block_attention_flex(q, k, v, routing)
        return block_attention(q, k, v, routing, need_reg)
    B, N, H, dh = q.shape
    scale = dh ** -0.5
    logit_parts, mask_parts, bias_parts, gv, K, cent = [], [], [], None, 0, None
    if routing.idx is not None:
        K = routing.idx.shape[-1]
        gi = routing.idx.reshape(B, N * K, 1).expand(-1, -1, H * dh)
        gk = k.reshape(B, N, -1).gather(1, gi).view(B, N, K, H, dh)
        gv = None if agg == "ebag" else v.reshape(B, N, -1).gather(1, gi).view(B, N, K, H, dh)
        logit_parts.append(torch.einsum("bnhd,bnkhd->bnhk", q, gk).float() * scale)
        mask_parts.append(routing.mask[:, :, None, :].expand(B, N, H, K))
        bias_parts.append(torch.zeros(1, 1, 1, K, device=q.device))
    if routing.assign is not None:
        lc, mc, Sv, cnt, own_h = _centroid_part(q, k, v, routing.assign, routing.n_clusters, scale)
        logit_parts.append(lc)
        mask_parts.append(mc.expand(B, N, H, -1))
        bias_parts.append(cnt.log())
        cent = (Sv, cnt, own_h)
    logits = torch.cat(logit_parts, -1)
    mask = torch.cat(mask_parts, -1)
    w = torch.softmax(logits.masked_fill(~mask, NEG), -1) * mask       # fully-masked rows -> 0
    out = 0.0
    if gv is not None:
        out = out + torch.einsum("bnhk,bnkhd->bnhd", w[..., :K].to(gv.dtype), gv)
    elif K:
        out = out + ebag_weighted_sum(w[..., :K], v, routing.idx)
    if cent is not None:
        Sv, cnt, own_h = cent
        coef = (w[..., K:] / cnt).to(Sv.dtype)
        out = out + torch.einsum("bnhc,bchd->bnhd", coef, Sv) - coef.gather(3, own_h) * v
    if need_reg:
        raw = (logits - torch.cat([bp.expand(B, N, H, -1) for bp in bias_parts], -1)) / scale
        reg = H * (raw.masked_fill(~mask, 0.0) ** 2).sum() / mask.sum().clamp(min=1)
    else:
        reg = q.new_zeros(())
    return out.reshape(B, N, H * dh), reg
