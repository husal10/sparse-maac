from __future__ import annotations

import argparse

ATTN_MODES = ["dense_loop", "dense_flash", "none", "cluster", "random", "centroid", "knn", "lsh", "hybrid", "block", "meanfield", "anneal"]


def build_parser():
    p = argparse.ArgumentParser(description="Sparse-routing MAAC")
    g = p.add_argument_group("environment")
    g.add_argument("--env", default="spread", choices=["spread", "tag"])
    g.add_argument("--n_agents", type=int, default=8)
    g.add_argument("--n_envs", type=int, default=8)
    g.add_argument("--episode_length", type=int, default=25)
    g.add_argument("--n_landmarks", type=int, default=None)
    g.add_argument("--world_size", type=float, default=None)
    g.add_argument("--k_obs_agents", type=int, default=3)
    g.add_argument("--k_obs_landmarks", type=int, default=3)
    g.add_argument("--tag_share_radius", type=float, default=0.0,
                   help="tag: 0 = each predator is paid for its own catches; r > 0 = also for catches by predators "
                        "within distance r")
    g.add_argument("--spread_reward", default="local", choices=["local", "global"],
                   help="spread: local = each agent scored on its k nearest landmarks; "
                        "global = MPE-style reward shared over all landmarks")
    g.add_argument("--cover_radius", type=float, default=0.1,
                   help="spread: a landmark counts as covered when an agent is this close at the last step")
    g.add_argument("--success_coverage", type=float, default=0.9,
                   help="spread: an episode counts as solved when its final coverage is at least this")
    g = p.add_argument_group("critic / attention")
    g.add_argument("--attn", default="dense_flash", choices=ATTN_MODES,
                   help="dense_loop = MAAC's original per-agent loops; dense_flash = FlashMAAC (one fused SDPA call); "
                        "centroid = C-MAAC; block = Z-MAAC; anneal = FlashMAAC annealed into Z-MAAC as N grows; "
                        "none = critic without attention (Q_i sees only o_i); meanfield = equal weights over the k "
                        "nearest agents (Yang et al. 2018); knn, cluster, random, lsh, hybrid = other routers")
    g.add_argument("--param_sharing", default="auto", choices=["auto", "type", "agent"],
                   help="auto: per-agent params for dense_loop (original MAAC), per-type otherwise")
    g.add_argument("--heads", type=int, default=4)
    g.add_argument("--critic_hidden", type=int, default=128)
    g.add_argument("--pol_hidden", type=int, default=128)
    g.add_argument("--route_feat", default="pos", choices=["pos", "emb"],
                   help="features routers see (always detached, never actions)")
    g.add_argument("--obs_norm", default="running", choices=["running", "bn"],
                   help="running: running mean and variance, held fixed during updates (keeps the baseline exact); "
                        "bn: MAAC's BatchNorm, exact only with per-agent encoders: shared encoders let a_i reach Q_i "
                        "through the batch statistics")
    g.add_argument("--attn_reg", type=float, default=0.0,
                   help="attention-logit regulariser in MAAC units (original uses 1e-3); "
                        ">0 forces the explicit (non-SDPA) attention path")
    g.add_argument("--loss_reduce", default="mean", choices=["mean", "sum"],
                   help="mean: losses averaged over agents, critic clip 10, one policy clip 0.5 (scale-free in N); "
                        "sum: original MAAC, losses summed over agents, shared critic grads x 1/N "
                        "(scale_shared_grads), critic clip 10*N, policy clip 0.5 per agent/type module")
    g.add_argument("--sparse_agg", default="gather", choices=["gather", "ebag"],
                   help="index routers (cluster, random, knn, lsh, hybrid): weighted sum over attended agents by "
                        "gathering a (B,N,K,H,dh) copy of the values (gather) or one fused F.embedding_bag call (ebag); "
                        "same function")
    g.add_argument("--block_impl", default="sdpa", choices=["sdpa", "flex"],
                   help="block router: padded dense blocks + SDPA (sdpa) or one FlexAttention call with a static "
                        "block-sparse mask (flex, compiled on CUDA); same function")
    g = p.add_argument_group("routing hyper-parameters (default sqrt(N))")
    g.add_argument("--knn_k", type=int, default=None)
    g.add_argument("--rand_k", type=int, default=None)
    g.add_argument("--n_clusters", type=int, default=None)
    g.add_argument("--kmeans_iters", type=int, default=5)
    g.add_argument("--knn_backend", default="auto", choices=["auto", "exact", "zorder"])
    g.add_argument("--lsh_rounds", type=int, default=2)
    g.add_argument("--lsh_bucket", type=int, default=None)
    g.add_argument("--block_size", type=int, default=None, help="block mode: b (default sqrt(N/3))")
    g.add_argument("--anneal_n0", type=int, default=900,
                   help="anneal mode: full attention (FlashMAAC) up to this many agents; set it to the crossover "
                        "N* measured on your GPU and minibatch size (fit_crossover.py)")
    g.add_argument("--anneal_n1", type=int, default=None,
                   help="anneal mode: exactly Z-MAAC from this many agents on (default 4 x anneal_n0); the block size is "
                        "annealed geometrically in log N in between (routing.anneal_schedule)")
    g.add_argument("--no_block_global", dest="block_global", action="store_false",
                   help="block mode: local window only, no far-field block-centroid tokens")
    g.add_argument("--no_balanced", dest="balanced", action="store_false",
                   help="cluster mode: exact (possibly unbalanced) k-means clusters instead of equal chunks")
    g = p.add_argument_group("training")
    g.add_argument("--n_episodes", type=int, default=10000)
    g.add_argument("--buffer_size", type=int, default=100000)
    g.add_argument("--buffer_fp32", dest="buffer_fp16", action="store_false")
    g.add_argument("--batch_size", type=int, default=256)
    g.add_argument("--steps_per_update", type=int, default=100)
    g.add_argument("--num_updates", type=int, default=4)
    g.add_argument("--pi_lr", type=float, default=1e-3)
    g.add_argument("--q_lr", type=float, default=1e-3)
    g.add_argument("--tau", type=float, default=1e-3)
    g.add_argument("--gamma", type=float, default=0.99)
    g.add_argument("--reward_scale", type=float, default=10.0)
    g.add_argument("--amp", action="store_true", help="bf16 autocast for policy/critic")
    g = p.add_argument_group("run")
    g.add_argument("--device", default="auto")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--run_name", default=None)
    g.add_argument("--out_dir", default="runs")
    g.add_argument("--log_interval", type=int, default=10, help="episodes (counted in env batches)")
    g.add_argument("--eval_every", type=int, default=0, help="episodes; 0 disables")
    g.add_argument("--eval_episodes", type=int, default=8)
    g.add_argument("--save_interval", type=int, default=1000)
    return p


def parse(argv=None):
    return build_parser().parse_args(argv)
