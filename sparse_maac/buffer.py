from __future__ import annotations

import torch


class ReplayBuffer:
    """Ring buffer of joint transitions kept on-device. Rows = env-steps (all N agents together).
    Memory ~ capacity * N * (2*obs_dim*bytes + 5) ; obs stored fp16 by default."""

    def __init__(self, capacity, n_agents, obs_dim, device, fp16=True):
        dt = torch.float16 if fp16 else torch.float32
        self.cap, self.dev, self.ptr, self.size = capacity, torch.device(device), 0, 0
        self.obs = torch.zeros(capacity, n_agents, obs_dim, dtype=dt, device=device)
        self.nobs = torch.zeros_like(self.obs)
        self.acs = torch.zeros(capacity, n_agents, dtype=torch.uint8, device=device)
        self.rews = torch.zeros(capacity, n_agents, dtype=torch.float32, device=device)
        self.mem_gb = sum(t.numel() * t.element_size() for t in (self.obs, self.nobs, self.acs, self.rews)) / 1e9

    def __len__(self):
        return self.size

    def push(self, obs, acs, rews, next_obs):
        E = obs.shape[0]
        idx = (self.ptr + torch.arange(E, device=self.dev)) % self.cap
        self.obs[idx], self.nobs[idx] = obs.to(self.obs.dtype), next_obs.to(self.obs.dtype)
        self.acs[idx], self.rews[idx] = acs.to(torch.uint8), rews
        self.ptr = (self.ptr + E) % self.cap
        self.size = min(self.size + E, self.cap)

    def sample(self, batch_size):
        i = torch.randint(0, self.size, (batch_size,), device=self.dev)
        return (self.obs[i].float(), self.acs[i].long(), self.rews[i].unsqueeze(-1), self.nobs[i].float())
