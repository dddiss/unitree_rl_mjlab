# Adapted from supermimic_vibed; Copyright (c) 2026 ETH Zurich, Chong Zhang.
# MIT license: see THIRD_PARTY_NOTICES.md.
"""Default SuperMimic Policy."""
import torch
from torch import nn
from . import factory
from .auxiliary import mlp, run_mlp
from .normalization import RunningNorm
from ..core.state import NUM_J


class Policy(nn.Module):
    """a = dec(enc(g), o_{t-9:t}); deterministic, latent z in R^32 bounded by tanh."""

    def __init__(self, obs_dim: int, goal_dim: int, hist: int = 10, z_dim: int = 32, hidden=(2048, 1024, 512, 256), depth_cfg=None, *, depth_encoder=None):
        super().__init__()
        self.hist = hist
        self.visual = depth_cfg is not None and depth_cfg.depth_enabled
        visual_dim = depth_cfg.depth_latent_dim if self.visual else 0
        if self.visual:
            self.depth_encoder = depth_encoder if depth_encoder is not None else factory.make_depth_encoder(depth_cfg)
        self.enc = mlp(goal_dim, list(hidden), z_dim)
        self.dec = mlp(z_dim + obs_dim * hist + visual_dim, list(hidden), NUM_J)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, depth=None,
                *, depth_latent=None) -> torch.Tensor:
        """obs_hist: (B, hist, obs_dim) noisy observations, newest last."""
        z = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        o = self.obs_norm(obs_hist).flatten(-2)
        x = torch.cat((z, o), -1)
        if self.visual:
            if depth_latent is not None:
                visual = depth_latent
            elif depth is not None:
                visual = self.depth_encoder(depth)
            else:
                raise ValueError("Visual policy requires a depth history")
            x = torch.cat((x, visual), -1)
        return run_mlp(self.dec, x)
