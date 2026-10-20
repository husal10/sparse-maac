| Critic | Parameters (N=64 / 512) | Update ms, N=64 (B=8/16) | Update ms, N=512 (B=8/16) | Slow-down vs ours at N=512 | Memory MB, N=512 (B=8/16) | Time exponent (B=8/16) | Largest N |
|---|---|---|---|---|---|---|---|
| MAAC loops, original structure | 2,564,544 / 20,171,392 | 394.6 / 390.0 | 3097.4 / 3026.3 | 697x / 373x | 2,677 / 4,878 | 0.98 / 0.99 | 512 |
| MAAC loops, shared parameters | 88,581 / 88,581 | 284.5 / 299.3 | 2751.6 / 2383.5 | 619x / 293x | 2,274 / 4,479 | 1.08 / 1.00 | 512 |
| FlashMAAC (batched) | 88,581 / 88,581 | 3.8 / 3.8 | 4.4 / 8.1 | 1x | 107 / 144 | 2.10 / 2.12 | 8,192 |

*Fitted exponent of update time in N: loops over N = 64-512, ours over N = 512-8,192 (below 512 the fused call is dominated by fixed overheads).
