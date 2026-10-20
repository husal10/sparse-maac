import math
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from sparse_maac.attention import flash_attention, loop_attention, sparse_attention
from sparse_maac.config import ATTN_MODES, parse
from sparse_maac.envs import ParticleEnv
from sparse_maac.networks import AttentionCritic
from sparse_maac.routing import Routing, make_router, members_from_assign
from sparse_maac.utils import knn_zorder, nearest_k

torch.manual_seed(0)
NO_ROUTER = ("dense_loop", "dense_flash", "none")
SPARSE = [m for m in ATTN_MODES if m not in NO_ROUTER]


def qkv(B=3, N=12, H=2, dh=8):
    return [torch.randn(B, N, H, dh) for _ in range(3)]


def test_loop_equals_flash():
    q, k, v = qkv()
    a, _ = loop_attention(q, k, v)
    b, _ = flash_attention(q, k, v)
    c, rc = flash_attention(q, k, v, need_reg=True)
    assert torch.allclose(a, b, atol=1e-5) and torch.allclose(b, c, atol=1e-5) and rc > 0
    _, rl = loop_attention(q, k, v, need_reg=True)
    assert torch.allclose(rl, rc, rtol=1e-4)


def test_knn_full_equals_dense():
    q, k, v = qkv()
    B, N = q.shape[:2]
    idx = torch.stack([torch.cat([torch.arange(i), torch.arange(i + 1, N)]) for i in range(N)])
    r = Routing(idx=idx.expand(B, -1, -1), mask=torch.ones(B, N, N - 1, dtype=torch.bool))
    s, _ = sparse_attention(q, k, v, r)
    d, _ = flash_attention(q, k, v)
    assert torch.allclose(s, d, atol=1e-5)


def test_centroid_singletons_equals_dense():
    q, k, v = qkv()
    B, N = q.shape[:2]
    r = Routing(assign=torch.arange(N).expand(B, N).contiguous(), n_clusters=N)
    s, _ = sparse_attention(q, k, v, r)
    d, _ = flash_attention(q, k, v)
    assert torch.allclose(s, d, atol=1e-5)


def test_centroid_single_cluster_is_leave_one_out_mean_field():
    # one cluster: token = mean of the other N-1 keys, weight (N-1): output must be a convex-ish mean of values
    q, k, v = qkv()
    B, N = q.shape[:2]
    r = Routing(assign=torch.zeros(B, N, dtype=torch.long), n_clusters=1)
    s, _ = sparse_attention(q, k, v, r)
    loo = (v.sum(1, keepdim=True) - v) / (N - 1)
    assert torch.allclose(s, loo.reshape(B, N, -1), atol=1e-5)


@pytest.mark.parametrize("mode", SPARSE)
def test_routers_valid(mode):
    N = 50
    cfg = parse([f"--attn={mode}", f"--n_agents={N}"])
    feat = torch.rand(4, N, 2)
    r = make_router(mode, N, cfg)(feat)
    if r.idx is not None:
        assert r.idx.shape[:2] == (4, N) and r.idx.min() >= 0 and r.idx.max() < N
        selfidx = torch.arange(N)[None, :, None]
        assert not ((r.idx == selfidx) & r.mask).any(), "self must be masked out"
        assert r.mask.any(-1).float().mean() > 0.9
    if r.assign is not None:
        assert r.assign.max() < r.n_clusters


def test_members_balanced_vs_exact():
    assign = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 2]])
    idx, m = members_from_assign(assign, 3, balanced=False)
    assert idx.shape[-1] == 6 and m[0, 0].sum() == 5 and m[0, 7].sum() == 0
    idx, m = members_from_assign(assign, 3, balanced=True)
    assert idx.shape[-1] == 3                                  # capped block size


def test_zorder_recall():
    pos = torch.rand(2, 600, 2)
    _, ex = nearest_k(pos, pos, 8, exclude_self=True)
    ap, mask = knn_zorder(pos, 8, window=24, shifts=3)
    hit = (ap[..., :, None] == ex[..., None, :]).any(-1).float().mean()
    assert hit > 0.8, hit


@pytest.mark.parametrize("mode", ATTN_MODES)
def test_critic_forward_backward_and_action_independent_routing(mode):
    N, D, A = 20, 19, 5
    cfg = parse([f"--attn={mode}", f"--n_agents={N}"])
    router = None if mode in NO_ROUTER else make_router(mode, N, cfg)
    crit = AttentionCritic(D, A, [(0, N)], mode, 32, 4, router, attn_reg=1e-3)
    obs = torch.randn(6, N, D)
    a1 = F.one_hot(torch.randint(0, A, (6, N)), A).float()
    q, qa, reg = crit(obs, a1)
    assert q.shape == (6, N, 1) and qa.shape == (6, N, A) and torch.isfinite(qa).all()
    (q.mean() + reg).backward()
    used = [p for n, p in crit.named_parameters() if mode != "none" or n.startswith(("s_enc", "head"))]
    assert mode != "meanfield" or not any(n.startswith(("W_q", "W_k")) for n, _ in crit.named_parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in used)
    if router is not None:                                     # routing depends on observations only
        torch.manual_seed(1); r1 = crit.route(obs)
        torch.manual_seed(1); r2 = crit.route(obs)
        for a, b in ((r1.idx, r2.idx), (r1.assign, r2.assign)):
            assert (a is None) or torch.equal(a, b)
        assert not any(t.requires_grad for t in (r1.idx, r1.assign) if t is not None)


def test_loop_critic_equals_flash_critic_shared_params():
    N, D, A = 10, 19, 5
    c1 = AttentionCritic(D, A, [(0, N)], "dense_loop", 32, 4)
    c2 = AttentionCritic(D, A, [(0, N)], "dense_flash", 32, 4)
    c2.load_state_dict(c1.state_dict())
    c1.eval(); c2.eval()
    obs = torch.randn(5, N, D); ac = F.one_hot(torch.randint(0, A, (5, N)), A).float()
    assert torch.allclose(c1(obs, ac)[1], c2(obs, ac)[1], atol=1e-5)


@pytest.mark.parametrize("scn", ["spread", "tag"])
def test_env(scn):
    env = ParticleEnv(scn, 30, 3, 5, seed=0)
    obs = env.reset()
    assert obs.shape == (3, 30, env.obs_dim)
    for _ in range(5):
        obs, rew, done, info = env.step(torch.randint(0, 5, (3, 30)))
        assert torch.isfinite(obs).all() and rew.shape == (3, 30)
    assert done and env.pos.abs().max() <= env.L + 1e-6
    img = env.render(0, px=200)
    assert img.shape[-1] == 3
    e2 = ParticleEnv(scn, 30, 3, 5, seed=0); o2 = e2.reset()
    assert torch.equal(ParticleEnv(scn, 30, 3, 5, seed=0).reset(), o2)    # deterministic


@pytest.mark.parametrize("mode", ATTN_MODES)
@pytest.mark.parametrize("scn", ["spread", "tag"])
def test_training_smoke(mode, scn, tmp_path):
    import train
    out = train.main(["--env", scn, "--n_agents", "12", "--attn", mode, "--n_envs", "2", "--n_episodes", "8",
                      "--batch_size", "16", "--steps_per_update", "4", "--num_updates", "1", "--buffer_size", "500",
                      "--critic_hidden", "32", "--pol_hidden", "32", "--device", "cpu", "--out_dir", str(tmp_path),
                      "--eval_episodes", "2", "--log_interval", "1", "--save_interval", "0"])
    assert math.isfinite(out["return_mean"])


# block-wise attention
from sparse_maac.attention import block_attention
from sparse_maac.routing import BlockRouter


def _rand_perm(B, N):
    return torch.stack([torch.randperm(N) for _ in range(B)])


@pytest.mark.parametrize("N,b", [(12, 12), (12, 1), (7, 1), (9, 9)])
def test_block_exact_limits(N, b):
    q, k, v = qkv(3, N)
    out, _ = block_attention(q, k, v, Routing(perm=_rand_perm(3, N), block=b))
    d, _ = flash_attention(q, k, v)
    assert torch.allclose(out, d, atol=1e-5)


def test_block_permutation_equivariance_and_padding():
    q, k, v = qkv(2, 13)                     # 13 not divisible by 5 -> padded block
    perm = _rand_perm(2, 13)
    a, _ = block_attention(q, k, v, Routing(perm=perm, block=5))
    assert a.shape == (2, 13, 16) and torch.isfinite(a).all()
    # relabelling agents must relabel outputs the same way when the curve order is relabelled consistently
    sh = torch.randperm(13)
    inv = sh.argsort()
    b, _ = block_attention(q[:, sh], k[:, sh], v[:, sh], Routing(perm=inv[perm], block=5))
    assert torch.allclose(b, a[:, sh], atol=1e-5)


def test_block_spatial_order_beats_random_order():
    torch.manual_seed(3)
    N = 256
    pos = torch.rand(2, N, 2)
    # attention logits depend on position -> nearby agents matter most
    f = lambda p: torch.cat([p * 6, torch.zeros(2, N, 10)], -1).view(2, N, 2, 6)
    q, k = f(pos), f(pos)
    v = torch.randn(2, N, 2, 6)
    d, _ = flash_attention(q, k, v)
    r = BlockRouter(8)(pos)
    sp, _ = block_attention(q, k, v, r)
    rd, _ = block_attention(q, k, v, Routing(perm=_rand_perm(2, N), block=8))
    e = lambda x: ((x - d).norm() / d.norm()).item()
    assert e(sp) < e(rd), (e(sp), e(rd))


def test_block_gradients_and_reg():
    N = 20
    cfg = parse(["--attn=block", f"--n_agents={N}", "--block_size=4"])
    crit = AttentionCritic(19, 5, [(0, N)], "block", 32, 4, make_router("block", N, cfg), attn_reg=1e-3)
    obs = torch.randn(4, N, 19)
    ac = F.one_hot(torch.randint(0, 5, (4, N)), 5).float()
    q, qa, reg = crit(obs, ac)
    (q.mean() + reg).backward()
    assert reg > 0 and all(torch.isfinite(p.grad).all() for p in crit.parameters())
    r1, r2 = crit.route(obs), crit.route(obs)
    assert torch.equal(r1.perm, r2.perm), "block routing must be deterministic given the state"


# exact action-independence of the all-actions head (MAAC Sec. 3.2)
@pytest.mark.parametrize("mode", ATTN_MODES)
@pytest.mark.parametrize("train", [False, True])
def test_q_all_independent_of_own_action(mode, train):
    """Changing only agent i's action must leave Q_i(o, .) (all-actions output) bit-for-bit unchanged, while the
    other agents' Q must change (credit assignment). Requires: routing state-only, own entry removed from every
    token (leave-one-out centroids), and no batch-statistic normalisation (use_bn=False)."""
    N, D, A, i = 24, 19, 5, 3
    cfg = parse([f"--attn={mode}", f"--n_agents={N}"])
    router = None if mode in NO_ROUTER else make_router(mode, N, cfg)
    c = AttentionCritic(D, A, [(0, N)], mode, 32, 4, router, use_bn=False).train(train)
    obs = torch.randn(8, N, D); obs[..., 2:4] = torch.rand(8, N, 2)
    a = F.one_hot(torch.randint(0, A, (8, N)), A).float()
    a2 = a.clone(); a2[:, i] = F.one_hot((a[:, i].argmax(-1) + 1) % A, A).float()
    torch.manual_seed(1); qa = c(obs, a)[1]
    torch.manual_seed(1); qb = c(obs, a2)[1]
    others = [j for j in range(N) if j != i]
    assert (qa[:, i] - qb[:, i]).abs().max() < 1e-6
    if mode == "none":                                         # no-interaction critic: nobody sees a_i
        assert (qa[:, others] - qb[:, others]).abs().max() < 1e-6
    else:
        assert (qa[:, others] - qb[:, others]).abs().max() > 1e-6


def test_batchnorm_leaks_own_action_in_train_mode():
    """Why use_bn=False at scale: with encoders shared across agents, BatchNorm(train) makes Q_i depend on
    a_i through batch statistics over all agents (see the per-agent counterpart below)."""
    N, D, A, i = 24, 19, 5, 3
    c = AttentionCritic(D, A, [(0, N)], "dense_flash", 32, 4, use_bn=True).train()
    obs = torch.randn(8, N, D)
    a = F.one_hot(torch.randint(0, A, (8, N)), A).float()
    a2 = a.clone(); a2[:, i] = F.one_hot((a[:, i].argmax(-1) + 1) % A, A).float()
    assert (c(obs, a)[1][:, i] - c(obs, a2)[1][:, i]).abs().max() > 1e-6


def test_running_norm_is_sample_independent():
    from sparse_maac.networks import RunningNorm
    rn = RunningNorm(4); rn.update(torch.randn(1000, 4) * 3 + 1)
    x = torch.randn(10, 4)
    assert torch.equal(rn(x)[:3], rn(x[:3]))                 # row output does not depend on the rest of the batch
    assert abs(rn.mean.mean().item() - 1) < 0.3 and abs(rn.var.mean().item() - 9) < 1.5


# faithfulness of the dense_loop ablation to the original MAAC code
@pytest.mark.parametrize("ranges", ["agent", "type"])
def test_loop_critic_equals_flash_critic(ranges):
    """dense_loop (per-(head,agent) extractor loops + per-agent attention loops) must compute exactly what the
    batched SDPA critic computes, for both per-agent and shared parameters: the ablation isolates implementation."""
    N, D, A = 9, 19, 5
    rg = [(i, i + 1) for i in range(N)] if ranges == "agent" else [(0, N)]
    c1 = AttentionCritic(D, A, rg, "dense_loop", 32, 4, use_bn=False)
    c2 = AttentionCritic(D, A, rg, "dense_flash", 32, 4, use_bn=False)
    c2.load_state_dict(c1.state_dict())
    obs = torch.randn(5, N, D); ac = F.one_hot(torch.randint(0, A, (5, N)), A).float()
    q1, q2 = c1(obs, ac)[1], c2(obs, ac)[1]
    assert torch.allclose(q1, q2, atol=1e-5)
    q1.sum().backward(); q2.sum().backward()
    for (n, p1), p2 in zip(c1.named_parameters(), c2.parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-4), n


def test_attention_reg_is_maac_units():
    """reg == mean_i sum_h mean_{b,j!=i} (q.k)^2 with raw logits, i.e. MAAC's attend_mag_reg / 1e-3 averaged
    over agents; all backends agree wherever they are exact."""
    q, k, v = qkv(3, 10, 2, 8)
    B, N, H, dh = q.shape
    ref = 0.0
    for i in range(N):
        for h in range(H):
            raw = torch.stack([(q[:, i, h] * k[:, j, h]).sum(-1) for j in range(N) if j != i], 1)
            ref += (raw ** 2).mean()
    ref = ref / N
    _, r_loop = loop_attention(q, k, v, True)
    _, r_flash = flash_attention(q, k, v, True)
    idx = torch.stack([torch.cat([torch.arange(i), torch.arange(i + 1, N)]) for i in range(N)]).expand(B, -1, -1)
    _, r_knn = sparse_attention(q, k, v, Routing(idx=idx, mask=torch.ones(B, N, N - 1, dtype=torch.bool)), True)
    _, r_blk = block_attention(q, k, v, Routing(perm=_rand_perm(B, N), block=N), True)
    for r in (r_loop, r_flash, r_knn, r_blk):
        assert torch.allclose(r, ref, rtol=1e-4), (r, ref)
    # centroid singletons: the log n_c (=0) bias must not enter the regulariser
    _, r_cen = sparse_attention(q, k, v, Routing(assign=torch.arange(N).expand(B, N).contiguous(), n_clusters=N), True)
    assert torch.allclose(r_cen, ref, rtol=1e-4)


def test_shared_parameters_match_maac_shared_modules():
    c = AttentionCritic(19, 5, [(0, 4)], "dense_flash", 32, 4, use_bn=False)
    shared = {id(p) for p in c.shared_parameters()}
    names = {n.split(".")[0] for n, p in c.named_parameters() if id(p) in shared}
    assert names == {"W_q", "W_k", "W_v", "sa_enc"}


@pytest.mark.parametrize("mode", ["lsh", "hybrid"])
def test_concatenated_token_sets_have_no_duplicates(mode):
    N = 64
    cfg = parse([f"--attn={mode}", f"--n_agents={N}", "--lsh_rounds=4"])
    r = make_router(mode, N, cfg)(torch.rand(3, N, 2))
    s = r.idx.masked_fill(~r.mask, -1).sort(-1).values
    dup = (s[..., 1:] == s[..., :-1]) & (s[..., 1:] >= 0)
    assert not dup.any()


@pytest.mark.parametrize("attn", ["dense_loop", "block"])
def test_loss_reduce_sum_smoke(attn, tmp_path):
    import train
    out = train.main(["--env", "spread", "--n_agents", "8", "--attn", attn, "--n_envs", "2", "--n_episodes", "6",
                      "--batch_size", "16", "--steps_per_update", "4", "--num_updates", "1", "--buffer_size", "300",
                      "--critic_hidden", "32", "--pol_hidden", "32", "--device", "cpu", "--out_dir", str(tmp_path),
                      "--eval_episodes", "2", "--save_interval", "0", "--loss_reduce", "sum", "--attn_reg", "1e-3",
                      "--obs_norm", "bn"])
    assert math.isfinite(out["return_mean"])


def test_checkpoint_roundtrip(tmp_path):
    """model.pt must reload with torch>=2.6's weights_only=True default (evaluate.py / benchmark --ckpt path)."""
    import train
    from types import SimpleNamespace
    from sparse_maac.agent import SparseMAAC
    from sparse_maac.runner import evaluate, make_env
    train.main(["--env", "tag", "--n_agents", "9", "--attn", "block", "--n_envs", "2", "--n_episodes", "4",
                "--batch_size", "8", "--steps_per_update", "4", "--num_updates", "1", "--buffer_size", "200",
                "--critic_hidden", "16", "--pol_hidden", "16", "--device", "cpu", "--out_dir", str(tmp_path),
                "--run_name", "ck", "--eval_episodes", "2", "--save_interval", "0"])
    ck = tmp_path / "ck" / "model.pt"
    d = torch.load(ck, map_location="cpu")
    cfg = SimpleNamespace(**d["cfg"])
    env = make_env(cfg, "cpu", n_envs=1, seed=5)
    ag = SparseMAAC(env.spec, cfg, "cpu"); ag.load(ck)
    assert torch.equal(ag.obs_norm.mean, d["obs_norm"]["mean"])
    assert math.isfinite(evaluate(env, ag, 1)["return_mean"])


# benchmark bookkeeping
def test_fit_crossover_excludes_rows_that_do_not_fit(tmp_path):
    import pandas as pd
    import fit_crossover as fc
    ns = [256, 512, 1024, 2048]
    rows = [{"mode": "dense_flash", "N": n, "ms": n * n / 1e4, "mem_mb": n / 2} for n in ns]
    rows += [{"mode": "block", "N": n, "ms": n ** 1.5 / 1e3, "mem_mb": n} for n in ns]
    rows += [{"mode": "knn", "N": n, "ms": 1.0, "mem_mb": 5 * n} for n in ns]        # 'fast' only because it spilled
    f = tmp_path / "b.csv"; pd.DataFrame(rows).to_csv(f, index=False)          # legacy CSV: no status column
    df = fc.load([f], vram_mb=6000)
    assert set(df[df["status"] != "ok"][["mode", "N"]].itertuples(index=False)) == {("knn", 2048)}
    out = fc.analyse(df).set_index("mode")
    assert out.loc["knn", "max_fit_N"] == 1024 and out.loc["knn", "first_nofit_N"].startswith("2048")
    assert out.loc["dense_flash", "first_nofit_N"] == "-"
    assert abs(out.loc["dense_flash", "exp_time"] - 2.0) < 1e-6 and abs(out.loc["block", "exp_time"] - 1.5) < 1e-6
    assert out.loc["block", "N*_time_vs_dense_flash"] == "<=256"


def test_benchmark_single_cpu_row_has_status():
    import benchmark
    from types import SimpleNamespace
    a = SimpleNamespace(batch=2, hidden=32, heads=4, warmup=1, iters=2, loop_sharing="agent", vram_mb=None,
                        knn_k=None, rand_k=None, n_clusters=None, lsh_bucket=None, block_size=None,
                        lsh_rounds=None, knn_backend=None)
    r = benchmark.run_single("block", 32, a, torch.device("cpu"))
    assert r["status"] == "ok" and r["ms"] > 0


# benchmark batch-size sweep
def test_fit_crossover_is_per_batch(tmp_path):
    import pandas as pd
    import fit_crossover as fc
    ns = [256, 512, 1024, 2048]
    rows = []
    for B, k in ((8, 1.0), (32, 4.0)):                     # batch 32: everything 4x slower and 4x more memory
        rows += [{"mode": "dense_flash", "N": n, "batch": B, "ms": k * n * n / 1e4, "mem_mb": k * n / 2, "status": "ok"} for n in ns]
        rows += [{"mode": "block", "N": n, "batch": B, "ms": k * n ** 1.5 / 1e3, "mem_mb": k * n, "status": "ok"} for n in ns]
    rows[-1]["status"] = "spilled"                          # block @2048 does not fit at batch 32
    df = fc.load([_csv(tmp_path, rows)])
    out = fc.analyse(df)
    assert set(out["batch"]) == {8, 32}
    blk = out[out["mode"] == "block"].set_index("batch")
    assert blk.loc[8, "max_fit_N"] == 2048 and blk.loc[32, "max_fit_N"] == 1024
    assert blk.loc[32, "first_nofit_N"].startswith("2048")
    bb = fc.by_batch(out)
    assert list(bb.columns) == ["B=8", "B=32"] and "fits <= 1024" in bb.loc["block", "B=32"]
    legacy = fc.load([_csv(tmp_path, [{k: v for k, v in r.items() if k != "batch"} for r in rows[:4]], "l.csv")], batch=16)
    assert set(legacy["batch"]) == {16}


def _csv(tmp_path, rows, name="b.csv"):
    import pandas as pd
    f = tmp_path / name
    pd.DataFrame(rows).to_csv(f, index=False)
    return f


def test_benchmark_batches_and_skip_after_nofit(tmp_path, monkeypatch):
    """Rows carry their batch; once a (mode, batch) does not fit, larger N are recorded as skipped, not run."""
    import json, subprocess, sys as _sys
    import pandas as pd
    import benchmark
    calls = []

    def fake_run(cmd, capture_output, text):
        N, B = int(cmd[cmd.index("--single") + 2]), int(cmd[cmd.index("--batch") + 1])
        calls.append((N, B))
        st = "spilled" if N * B >= 256 else "ok"
        return subprocess.CompletedProcess(cmd, 0, "RESULT " + json.dumps(
            {"ms": 1.0, "ms_min": 1.0, "mem_mb": float(N * B), "reserved_mb": float(N * B), "status": st}), "")
    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    out = tmp_path / "b.csv"
    monkeypatch.setattr(_sys, "argv", ["benchmark.py", "--modes", "block", "--ns", "16", "32", "64", "128",
                                       "--batches", "4", "8", "--device", "cpu", "--csv", str(out)])
    benchmark.main()
    df = pd.read_csv(out).set_index(["batch", "N"])["status"]
    assert df[(4, 64)] == "spilled" and df[(4, 128)] == "skipped"
    assert df[(8, 32)] == "spilled" and df[(8, 64)] == "skipped" and df[(8, 128)] == "skipped"
    assert (4, 128) not in calls and (8, 64) not in calls


# task metrics
def test_episode_metrics_spread_and_tag():
    env = ParticleEnv("spread", 6, 2, 3, seed=0, world_size=20.0, success_coverage=0.5)
    env.reset()
    env.lm[:] = torch.arange(6.0)[:, None] * torch.tensor([3.0, 0.0]) - 8.0   # landmarks 3 apart on a line
    env.pos[:] = 15.0                                  # every agent far from every landmark ...
    env.pos[0, :3] = env.lm[0, :3]                     # ... except in env 0, where 3 of 6 landmarks get an agent
    env.vel[:] = 0
    env.step(torch.zeros(2, 6, dtype=torch.long))      # action 0 = no force: positions stay
    m = env.episode_metrics()
    assert torch.allclose(m["final_coverage"], torch.tensor([0.5, 0.0]))
    assert torch.equal(m["success"], torch.tensor([1.0, 0.0]))
    t = ParticleEnv("tag", 9, 1, 4, seed=0, world_size=60.0)
    t.reset()
    t.pos[0, 6:] = torch.tensor([[50.0, 0.0], [50.0, 20.0], [50.0, 40.0]]); t.pos[0, :6] = -50.0          # prey far from predators -> nobody caught
    t.pos[0, 0] = t.pos[0, 6]                          # ... except prey 0 (agent 6) touched by predator 0
    t.vel[:] = 0
    t.step(torch.zeros(1, 9, dtype=torch.long))
    m = t.episode_metrics()
    assert abs(m["capture_rate"].item() - 1 / 3) < 1e-6 and torch.equal(m["success"], m["capture_rate"])


def test_collect_results_normalization_and_convergence(tmp_path):
    import json
    import pandas as pd
    import collect_results as cr

    def run(name, attn, seed, final, curve, rnd=0.0, env="spread"):
        d = tmp_path / name; d.mkdir()
        k = "vr_return_team0" if env == "tag" else "return_mean"
        json.dump({"env": env, "n_agents": 8, "attn": attn, "seed": seed, "param_sharing": "auto"}, open(d / "config.json", "w"))
        json.dump({k: final, "final_coverage": 0.5}, open(d / "final_eval.json", "w"))
        json.dump({k: rnd, "final_coverage": 0.0}, open(d / "random_eval.json", "w"))
        steps = [25 * (i + 1) for i in range(len(curve))]
        pd.DataFrame({"episode": steps, "env_steps": steps, "elapsed_s": steps, k: curve}).to_csv(d / "eval.csv", index=False)
        pd.DataFrame({"update_ms": [1.0], "peak_mem_mb": [1.0], "elapsed_s": [2.0]}).to_csv(d / "metrics.csv", index=False)
    for s in (1, 2, 3):
        run(f"d{s}", "dense_flash", s, 10.0, [2.0, 5.0, 8.0, 10.0, 10.0, 10.0])
        run(f"b{s}", "block", s, 5.0, [5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    df = cr.score_runs(cr.load_runs(tmp_path))
    dense, blk = df[df.method == "dense_flash"].iloc[0], df[df.method == "block"].iloc[0]
    assert abs(dense["return_norm"] - 1) < 1e-9 and abs(blk["return_norm"] - 0.5) < 1e-9
    assert dense["task_success"] == 0.5
    assert blk["steps_to_conv"] < dense["steps_to_conv"]          # a flat curve is at its final level at once


def test_crossplay_agent_takes_each_team_from_its_own_policy():
    from sparse_maac.runner import CrossPlayAgent

    class Const:
        def __init__(self, v): self.v = v
        def prep_rollouts(self): pass
        def act(self, obs, explore=True): return torch.full(obs.shape[:2], self.v, dtype=torch.long)
    a = CrossPlayAgent([Const(1), Const(3)], [(0, 4), (4, 6)]).act(torch.zeros(2, 6, 5))
    assert a[:, :4].eq(1).all() and a[:, 4:].eq(3).all()


def test_batchnorm_with_per_agent_encoders_does_not_leak():
    """Original MAAC (per-agent encoders): each BatchNorm only sees its own agent, so Q_i stays exact in train mode."""
    N, D, A, i = 10, 19, 5, 3
    for mode in ("dense_loop", "dense_flash"):
        c = AttentionCritic(D, A, [(j, j + 1) for j in range(N)], mode, 32, 4, use_bn=True).train()
        obs = torch.randn(8, N, D)
        a = F.one_hot(torch.randint(0, A, (8, N)), A).float()
        a2 = a.clone(); a2[:, i] = F.one_hot((a[:, i].argmax(-1) + 1) % A, A).float()
        assert (c(obs, a)[1][:, i] - c(obs, a2)[1][:, i]).abs().max() < 1e-6


def test_welch_interval_matches_scipy():
    from scipy import stats
    import collect_results as cr
    rng = np.random.default_rng(0)
    a, b = rng.normal(1.0, 0.05, 5), rng.normal(1.02, 0.12, 7)
    d, lo, hi = cr.welch_ci(a, b, 90)
    ci = stats.ttest_ind(a, b, equal_var=False).confidence_interval(0.90)
    assert abs(d - (a.mean() - b.mean())) < 1e-12 and abs(lo - ci.low) < 1e-9 and abs(hi - ci.high) < 1e-9
    assert cr.welch_ci([1, 1], [1, 1]) == (0.0, 0.0, 0.0)        # zero variance: degenerate interval, no crash


# fused execution backends: same function, different kernels
def test_meanfield_is_plain_neighbour_average():
    from sparse_maac.attention import mean_field_attention
    B, N, H, dh = 2, 30, 2, 4
    v = torch.randn(B, N, H, dh, requires_grad=True)
    cfg = parse(["--attn=meanfield", f"--n_agents={N}"])
    r = make_router("meanfield", N, cfg)(torch.rand(B, N, 2))
    out, _ = mean_field_attention(v, r)
    ref = torch.stack([torch.stack([v[b, r.idx[b, i]].mean(0) for i in range(N)]) for b in range(B)])
    assert torch.allclose(out, ref.reshape(B, N, -1), atol=1e-6)
    oe, _ = mean_field_attention(v, r, agg="ebag")
    assert torch.allclose(out, oe, atol=1e-5)
    g = torch.randn_like(out)
    assert torch.allclose(torch.autograd.grad((out * g).sum(), v)[0], torch.autograd.grad((oe * g).sum(), v)[0], atol=1e-5)


@pytest.mark.parametrize("mode", ["cluster", "random", "knn", "lsh", "hybrid"])
def test_ebag_weighted_sum_matches_gather(mode):
    from sparse_maac.attention import sparse_attention
    B, N, H, dh = 3, 40, 2, 8
    q, k, v = (torch.randn(B, N, H, dh, requires_grad=True) for _ in range(3))
    cfg = parse([f"--attn={mode}", f"--n_agents={N}"])
    torch.manual_seed(1); r = make_router(mode, N, cfg)(torch.rand(B, N, 2))
    a, _ = sparse_attention(q, k, v, r)
    b, _ = sparse_attention(q, k, v, r, agg="ebag")
    g = torch.randn_like(a)
    ga = torch.autograd.grad((a * g).sum(), (q, k, v))
    gb = torch.autograd.grad((b * g).sum(), (q, k, v))
    assert torch.allclose(a, b, atol=1e-5)
    assert all(torch.allclose(x, y, atol=1e-5) for x, y in zip(ga, gb))


@pytest.mark.parametrize("N,b,glob", [(50, 4, True), (50, 7, True), (50, 5, False), (12, 12, True), (12, 1, True),
                                      (9, 3, True)])
def test_block_flex_forward_matches_sdpa(N, b, glob):
    from sparse_maac.attention import block_attention, block_attention_flex
    B, H, dh = 2, 2, 8
    q, k, v = (torch.randn(B, N, H, dh) for _ in range(3))
    cfg = parse(["--attn=block", f"--n_agents={N}", f"--block_size={b}"] + ([] if glob else ["--no_block_global"]))
    r = make_router("block", N, cfg)(torch.rand(B, N, 2))
    assert torch.allclose(block_attention(q, k, v, r)[0], block_attention_flex(q, k, v, r)[0], atol=1e-5)


def test_block_flex_falls_back_to_sdpa_for_cpu_gradients_and_reg():
    from sparse_maac.attention import sparse_attention
    N = 30
    q, k, v = (torch.randn(2, N, 2, 8, requires_grad=True) for _ in range(3))
    r = make_router("block", N, parse(["--attn=block", f"--n_agents={N}"]))(torch.rand(2, N, 2))
    a, _ = sparse_attention(q, k, v, r, block_impl="flex")
    b, _ = sparse_attention(q, k, v, r)
    assert torch.equal(a, b)                                    # CPU + grad -> SDPA path exactly
    _, reg = sparse_attention(q, k, v, r, need_reg=True, block_impl="flex")
    assert reg > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention backward needs CUDA")
def test_block_flex_gradients_match_sdpa_on_cuda():
    from sparse_maac.attention import block_attention, block_attention_flex
    N, dev = 300, torch.device("cuda")
    q, k, v = (torch.randn(2, N, 4, 32, device=dev, requires_grad=True) for _ in range(3))
    r = make_router("block", N, parse(["--attn=block", f"--n_agents={N}"]))(torch.rand(2, N, 2, device=dev))
    a, _ = block_attention(q, k, v, r)
    b, _ = block_attention_flex(q, k, v, r)
    g = torch.randn_like(a)
    ga = torch.autograd.grad((a * g).sum(), (q, k, v))
    gb = torch.autograd.grad((b * g).sum(), (q, k, v))
    assert torch.allclose(a, b, atol=2e-3)
    assert all(torch.allclose(x, y, atol=2e-3) for x, y in zip(ga, gb))


@pytest.mark.parametrize("flags", [["--attn=knn", "--sparse_agg=ebag"], ["--attn=hybrid", "--sparse_agg=ebag"],
                                   ["--attn=block", "--block_impl=flex"]])
def test_fused_backends_train_and_stay_action_independent(flags, tmp_path):
    import train
    out = train.main(flags + ["--env", "tag", "--n_agents", "12", "--n_envs", "2", "--n_episodes", "8",
                              "--batch_size", "16", "--steps_per_update", "4", "--num_updates", "1", "--buffer_size", "500",
                              "--critic_hidden", "32", "--pol_hidden", "32", "--device", "cpu", "--out_dir", str(tmp_path),
                              "--eval_episodes", "2", "--log_interval", "1", "--save_interval", "0"])
    assert math.isfinite(out["return_mean"])
    cfg = parse(flags)
    N, D, A, i = 24, 19, 5, 3
    router = make_router(cfg.attn, N, parse(flags + [f"--n_agents={N}"]))
    c = AttentionCritic(D, A, [(0, N)], cfg.attn, 32, 4, router, use_bn=False, sparse_agg=cfg.sparse_agg,
                        block_impl=cfg.block_impl).train()
    obs = torch.randn(8, N, D); obs[..., 2:4] = torch.rand(8, N, 2)
    a = F.one_hot(torch.randint(0, A, (8, N)), A).float()
    a2 = a.clone(); a2[:, i] = F.one_hot((a[:, i].argmax(-1) + 1) % A, A).float()
    with torch.no_grad():                                       # no-grad pass also exercises the flex kernel on CPU
        torch.manual_seed(1); qa = c(obs, a)[1]
        torch.manual_seed(1); qb = c(obs, a2)[1]
    assert (qa[:, i] - qb[:, i]).abs().max() < 1e-6


def test_benchmark_mode_backend_syntax():
    import benchmark as bm
    assert bm.split_mode("knn+ebag") == ("knn", {"sparse_agg": "ebag"})
    assert bm.split_mode("block+flex") == ("block", {"block_impl": "flex"})
    with pytest.raises(ValueError):
        bm.split_mode("knn+warp")
    a = bm.argparse.Namespace(knn_k=None, rand_k=None, n_clusters=None, lsh_bucket=None, block_size=None,
                              lsh_rounds=None, knn_backend=None, loop_sharing="type", hidden=32, heads=4)
    crit = bm.build_critic("knn+ebag", 32, a, torch.device("cpu"))
    assert crit.attn == "knn" and crit.sparse_agg == "ebag"


def test_package_limits_cpu_threads_unless_omp_set():
    import os, subprocess, sys as _sys
    env = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    code = "import torch, sparse_maac; print(torch.get_num_threads())"
    out = subprocess.run([_sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert out.stdout.strip() == "1"


def test_spread_local_reward_scores_each_agent_on_its_nearest_landmarks():
    from sparse_maac.envs import ParticleEnv
    env = ParticleEnv("spread", 4, 1, 25, "cpu", 0, k_landmarks=1, spread_reward="local")
    env.reset()
    env.lm = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    env.pos = torch.tensor([[[0.0, 0.0], [1.0, 0.02], [0.0, 1.5], [3.0, 3.0]]])   # agent 0 on lm 0, 1 near lm 1
    env.vel.zero_()
    r, _ = env._reward()
    # agent 0's nearest landmark (lm 0) has an agent on it -> 0; agent 1's (lm 1) is 0.02 away; agent 2's nearest is
    # lm 2 at distance 0.5 from agent 2 (its nearest agent); agent 3's nearest is lm 3, whose nearest agent is agent 1
    # (distance 0.98): agent 3 is scored on how well its neighbours cover its landmark, not on its own distance
    expect = torch.tensor([[0.0, -0.02, -0.5, -0.98]])
    assert torch.allclose(r, expect, atol=1e-5)
    g = ParticleEnv("spread", 4, 1, 25, "cpu", 0, spread_reward="global")
    assert g.spread_reward == "global"


def test_tag_vs_random_evaluation_and_primary_key(tmp_path):
    import json
    import pandas as pd
    import collect_results as cr
    from sparse_maac.runner import RandomAgent, evaluate, evaluate_vs_random, make_env, random_vs_random_anchor
    cfg = parse(["--env", "tag", "--n_agents", "9", "--n_envs", "2"])
    env = make_env(cfg, torch.device("cpu"))
    a = RandomAgent(env.n_actions, 1)
    m = evaluate_vs_random(env, a, 1)
    assert set(m) == {"vr_return_team0", "vr_capture_rate", "vr_return_team1", "vr_escape_rate"}
    assert 0 <= m["vr_capture_rate"] <= 1 and 0 <= m["vr_escape_rate"] <= 1
    anc = random_vs_random_anchor(evaluate(env, a, 1))
    assert set(anc) == set(m)
    assert evaluate_vs_random(make_env(parse(["--env", "spread", "--n_agents", "6"]), torch.device("cpu")), a, 1) == {}
    assert cr.ret_key("tag") == "vr_return_team0"
    # a tag run without the vs-random keys is refused, not silently scored on self-play
    d = tmp_path / "t1"; d.mkdir()
    json.dump({"env": "tag", "n_agents": 9, "attn": "dense_flash", "seed": 1, "param_sharing": "auto"}, open(d / "config.json", "w"))
    json.dump({"return_team0": 1.0}, open(d / "final_eval.json", "w"))
    json.dump({"return_team0": 0.0}, open(d / "random_eval.json", "w"))
    pd.DataFrame({"update_ms": [1.0], "peak_mem_mb": [1.0], "elapsed_s": [2.0]}).to_csv(d / "metrics.csv", index=False)
    with pytest.raises(KeyError):
        cr.score_runs(cr.load_runs(tmp_path))


def test_spread_episode_coverage_is_stored_on_request():
    from sparse_maac.runner import RandomAgent, evaluate, make_env
    env = make_env(parse(["--env", "spread", "--n_agents", "8", "--n_envs", "3"]), torch.device("cpu"))
    m = evaluate(env, RandomAgent(env.n_actions, 1), 2, episode_values=True)
    eps = m["final_coverage_episodes"]
    assert len(eps) == 6 and all(0 <= x <= 1 for x in eps)
    assert abs(sum(eps) / len(eps) - m["final_coverage"]) < 1e-4
    assert "final_coverage_episodes" not in evaluate(env, RandomAgent(env.n_actions, 1), 1)


def test_tag_share_radius_rewards_nearby_catches():
    """With a share radius, a predator is paid for a catch by a nearby predator; radius 0 keeps the individual reward."""
    from sparse_maac.envs import ParticleEnv
    def env_with(r):
        e = ParticleEnv("tag", n_agents=6, n_envs=1, tag_share_radius=r)   # 4 predators, 2 prey
        e.reset()
        e.pos = torch.tensor([[[0.0, 0.0], [0.3, 0.0], [5.0, 5.0], [-5.0, -5.0], [0.05, 0.0], [9.0, -9.0]]])
        return e
    r0, _ = env_with(0.0)._reward()
    r1, _ = env_with(1.0)._reward()
    assert r0[0, 0] > 0.5 and r0[0, 1] < 0.5          # only predator 0 touches prey 4
    assert r1[0, 1] - r0[0, 1] > 0.99                 # predator 1 (0.3 away) shares the catch
    assert torch.allclose(r1[0, 2:4], r0[0, 2:4])     # far predators do not


# annealed critic: FlashMAAC below n0, Z-MAAC above n1, block size annealed in between
from sparse_maac.routing import anneal_schedule


def test_anneal_schedule_endpoints_and_monotone():
    n0, n1 = 900, 3600
    assert anneal_schedule(64, n0, n1) == (0.0, 64)                    # full attention
    assert anneal_schedule(900, n0, n1) == (0.0, 900)
    for N in (3600, 4096, 8192):                                       # exactly Z-MAAC's block size
        assert anneal_schedule(N, n0, n1) == (1.0, max(1, round(math.sqrt(N / 3))))
    Ns = list(range(900, 3700, 50))
    ts = [anneal_schedule(N, n0, n1)[0] for N in Ns]
    frac = [anneal_schedule(N, n0, n1)[1] / N for N in Ns]
    assert all(a <= b for a, b in zip(ts, ts[1:])) and all(a >= b - 1e-9 for a, b in zip(frac, frac[1:]))
    assert all(round(math.sqrt(N / 3)) <= anneal_schedule(N, n0, n1)[1] <= N for N in Ns)
    assert anneal_schedule(1000, 900)[0] == anneal_schedule(1000, 900, 3600)[0]   # n1 defaults to 4 n0
    with pytest.raises(ValueError):
        anneal_schedule(1000, 900, 900)


def _twin_critics(mode_a, mode_b, N, extra=()):
    """Two critics with identical weights (same seed) that differ only in the attention mode."""
    out = []
    for mode in (mode_a, mode_b):
        cfg = parse([f"--attn={mode}", f"--n_agents={N}", *extra])
        router = None if mode in NO_ROUTER else make_router(mode, N, cfg)
        torch.manual_seed(7)
        out.append(AttentionCritic(19, 5, [(0, N)], mode, 32, 4, router).eval())
    return out


@pytest.mark.parametrize("N", [12, 40])
def test_anneal_below_n0_is_flashmaac(N):
    a, d = _twin_critics("anneal", "dense_flash", N, ["--anneal_n0=100"])
    obs, acts = torch.randn(3, N, 19), F.one_hot(torch.randint(0, 5, (3, N)), 5).float()
    assert a.route(obs).block == N
    assert torch.allclose(a(obs, acts)[1], d(obs, acts)[1], atol=1e-5)


@pytest.mark.parametrize("N", [48, 75])
def test_anneal_above_n1_is_zmaac(N):
    a, z = _twin_critics("anneal", "block", N, ["--anneal_n0=4", "--anneal_n1=16"])
    obs, acts = torch.randn(3, N, 19), F.one_hot(torch.randint(0, 5, (3, N)), 5).float()
    assert a.route(obs).block == z.route(obs).block == round(math.sqrt(N / 3))
    assert torch.allclose(a(obs, acts)[1], z(obs, acts)[1], atol=1e-6)


def test_anneal_exact_while_at_most_two_blocks():
    """b >= N/2 (two blocks, both in every near field) is still dense attention, through the block path."""
    N = 30
    q, k, v = qkv(2, N)
    out, _ = block_attention(q, k, v, Routing(perm=_rand_perm(2, N), block=15))
    assert torch.allclose(out, flash_attention(q, k, v)[0], atol=1e-5)


@pytest.mark.parametrize("train", [False, True])
def test_anneal_midway_is_sparse_and_keeps_the_baseline_exact(train):
    """Inside the anneal (0 < t < 1): a real near-far critic whose routing ignores actions and whose Q_i[c] does
    not depend on a_i (conditions C1-C3), so the one-pass counterfactual baseline stays exact."""
    N = 60
    cfg = parse(["--attn=anneal", f"--n_agents={N}", "--anneal_n0=20", "--anneal_n1=200"])
    t, b = anneal_schedule(N, 20, 200)
    assert 0 < t < 1 and round(math.sqrt(N / 3)) < b < N / 2
    crit = AttentionCritic(19, 5, [(0, N)], "anneal", 32, 4, make_router("anneal", N, cfg), use_bn=False).train(train)
    obs = torch.randn(4, N, 19); obs[..., 2:4] = torch.rand(4, N, 2)
    a1 = F.one_hot(torch.randint(0, 5, (4, N)), 5).float()
    r = crit.route(obs)
    assert r.block == b and not r.perm.requires_grad
    q1 = crit(obs, a1, r)[1]
    for i in (0, 17, N - 1):                                   # change only a_i: Q_i(., c) for every c is unchanged
        a2 = a1.clone(); a2[:, i] = F.one_hot((a1[:, i].argmax(-1) + 1) % 5, 5).float()
        q2 = crit(obs, a2, r)[1]
        assert (q2[:, i] - q1[:, i]).abs().max() < 1e-6
        assert (q2 - q1).abs().max() > 1e-6                    # other agents do see a_i (credit assignment)
