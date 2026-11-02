# Adapted from supermimic_vibed; Copyright (c) 2026 ETH Zurich, Chong Zhang.
# MIT license: see THIRD_PARTY_NOTICES.md.
"""Default SuperMimic Critic."""
import torch
from torch import nn
from . import factory
from .auxiliary import mlp, run_mlp
from .normalization import RunningNorm


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden=(256, 128), depth_cfg=None, *, depth_encoder=None):
        super().__init__()
        self.visual = depth_cfg is not None and depth_cfg.depth_enabled
        visual_dim = depth_cfg.depth_latent_dim if self.visual else 0
        if self.visual:
            self.depth_encoder = depth_encoder if depth_encoder is not None else factory.make_depth_encoder(depth_cfg)
        self.net = mlp(in_dim + visual_dim, list(hidden), 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x, depth=None, *, depth_latent=None):
        x = self.norm(x)
        if self.visual:
            if depth_latent is not None:
                visual = depth_latent
            elif depth is not None:
                visual = self.depth_encoder(depth)
            else:
                raise ValueError("Visual critic requires a depth history")
            x = torch.cat((x, visual), -1)
        return run_mlp(self.net, x).squeeze(-1)
