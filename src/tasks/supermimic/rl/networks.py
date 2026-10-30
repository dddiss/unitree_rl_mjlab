# Adapted from supermimic_vibed; Copyright (c) 2026 ETH Zurich, Chong Zhang.
# MIT license: see THIRD_PARTY_NOTICES.md.
"""World model, policy (motion encoder + shared decoder), and critic (paper Table A.4)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..mdp.state import INV_DIM, NUM_J, XI_DIM, invariant_features
from ..mdp.rotations import so3_exp, yaw_mat
from .depth_encoder import DepthEncoder
from .depth_decoder import DepthDecoder


def make_depth_encoder(cfg):
    return DepthEncoder(cfg.depth_latent_dim, cfg.depth_frame_dim, cfg.depth_checkpoint,
                        cfg.depth_batch_size, cfg.depth_frames)


def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if self.count >= self.max_count:
            return
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)


WM_OUT = 3 + 3 + 3 + 3 + NUM_J + NUM_J

# bf16 autocast for MLP matmuls only (state integration, FK, rewards stay fp32)
AMP_DTYPE: torch.dtype | None = None


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None or x.device.type != "cuda":
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()


class WorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi) predicting frame-local state increments."""

    def __init__(self, hidden=(2048, 2048, 2048), depth_cfg=None, foot_reward_names=()):
        super().__init__()
        self.visual = depth_cfg is not None and depth_cfg.depth_enabled
        visual_dim = depth_cfg.depth_latent_dim if self.visual else 0
        self.in_dim = INV_DIM + NUM_J + XI_DIM
        self.net = mlp(self.in_dim + visual_dim, list(hidden), WM_OUT + visual_dim, act=nn.SiLU, layer_norm=True)
        with torch.no_grad():
            self.net[-1].weight[:WM_OUT].zero_()
            self.net[-1].bias[:WM_OUT].zero_()
        if self.visual:
            self.depth_encoder = make_depth_encoder(depth_cfg)
            self.depth_decoder = DepthDecoder(visual_dim)
            self.decode_batch_size = depth_cfg.depth_batch_size * depth_cfg.depth_frames
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(WM_OUT, eps=1e-4)
        # Branch from the final hidden features (2048 by default), not from
        # state deltas or h[t+1]. One shared MLP predicts the nonnegative RAW
        # costs in replay order: feet_stumble, feet_at_plane (when configured).
        self.foot_reward_names = tuple(foot_reward_names)
        self.foot_head = nn.Sequential(
            mlp(hidden[-1], [256], len(self.foot_reward_names), act=nn.SiLU), nn.Softplus(),
        ) if self.foot_reward_names else None

    def inputs(self, S, a, xi_n):
        return torch.cat((invariant_features(S), a, xi_n), -1)

    @staticmethod
    def targets(S, S1):
        """Frame-local increments from S to S1 (the regression targets)."""
        from ..mdp.rotations import so3_log

        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ R1),
                (Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    def _predict(self, S, a, xi_n, *, depth_latent=None, with_foot_costs=False):
        p, R, v, w, q, qd = S
        x = self.in_norm(self.inputs(S, a, xi_n))
        if self.visual:
            if depth_latent is None:
                raise ValueError("Visual WM requires the aggregated history latent h")
            x = torch.cat((x, depth_latent), -1)
        if with_foot_costs:
            if self.foot_head is None:
                raise ValueError("No foot-cost outputs are configured for this world model")
            features = run_mlp(self.net[:-1], x)
            y = run_mlp(self.net[-1], features)
            costs = run_mlp(self.foot_head, features)
        else:
            y = run_mlp(self.net, x)
        d = y[..., :WM_OUT] * self.out_norm.std() + self.out_norm.mean
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, NUM_J, NUM_J), -1)
        p1 = p + (H @ dp.unsqueeze(-1)).squeeze(-1)
        R1 = R @ so3_exp(dth)
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        result = (p1, R1, v1, w1, q + dq, qd + dqd), y[..., WM_OUT:]
        return (*result, costs) if with_foot_costs else result

    def step_with_foot_costs(self, S, a, xi_n, depth_latent=None):
        """Joint state/h/cost prediction in one trunk pass for this transition."""
        return self._predict(S, a, xi_n, depth_latent=depth_latent, with_foot_costs=True)

    def step(self, S, a, xi_n):
        return self._predict(S, a, xi_n)[0]

    def step_visual(self, S, a, xi_n, depth_latent):
        """Predict next state and next history latent; no image-space feedback."""
        return self._predict(S, a, xi_n, depth_latent=depth_latent)

    def decode_depth(self, latents):
        """Decode batch/time rows after rollout, bounding convolution workspace."""
        rows = latents.reshape(-1, latents.shape[-1])
        images = [self.depth_decoder(chunk) for chunk in rows.split(self.decode_batch_size)]
        return torch.cat(images, 0).reshape(*latents.shape[:-1], 18, 32)

    @staticmethod
    def state_error(S_hat, S):
        """Frame-local state error components (same layout as ``targets``)."""
        from ..mdp.rotations import so3_log

        p, R, v, w, q, qd = S
        ph, Rh, vh, wh, qh, qdh = S_hat
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (ph - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ Rh),
                (Rt @ (vh - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (wh - w).unsqueeze(-1)).squeeze(-1),
                qh - q,
                qdh - qd,
            ),
            -1,
        )


class Policy(nn.Module):
    """a = dec(enc(g), o_{t-9:t}); deterministic, latent z in R^32 bounded by tanh."""

    def __init__(self, obs_dim: int, goal_dim: int, hist: int = 10, z_dim: int = 32, hidden=(2048, 1024, 512, 256), depth_cfg=None, *, depth_encoder=None):
        super().__init__()
        self.hist = hist
        self.visual = depth_cfg is not None and depth_cfg.depth_enabled
        visual_dim = depth_cfg.depth_latent_dim if self.visual else 0
        if self.visual:
            self.depth_encoder = depth_encoder if depth_encoder is not None else make_depth_encoder(depth_cfg)
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


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden=(256, 128), depth_cfg=None, *, depth_encoder=None):
        super().__init__()
        self.visual = depth_cfg is not None and depth_cfg.depth_enabled
        visual_dim = depth_cfg.depth_latent_dim if self.visual else 0
        if self.visual:
            self.depth_encoder = depth_encoder if depth_encoder is not None else make_depth_encoder(depth_cfg)
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
