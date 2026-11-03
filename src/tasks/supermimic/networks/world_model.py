# Adapted from supermimic_vibed; Copyright (c) 2026 ETH Zurich, Chong Zhang.
# MIT license: see THIRD_PARTY_NOTICES.md.
"""Default SuperMimic WorldModel."""
import torch
from torch import nn
from . import factory
from .auxiliary import mlp, run_mlp
from .normalization import RunningNorm
from ..core.state import INV_DIM, NUM_J, XI_DIM, invariant_features
from ..tools.rotations import so3_exp, yaw_mat

WM_OUT = 3 + 3 + 3 + 3 + NUM_J + NUM_J


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
            self.depth_encoder = factory.make_depth_encoder(depth_cfg)
            from .factory import resolve_class
            cls = resolve_class(getattr(depth_cfg, "depth_decoder_class", ".depth_decoder:DepthDecoder"))
            self.depth_decoder = cls(visual_dim)
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

    def set_reward_output_domains(self, nonnegative):
        # Preserve the shared trunk/head and all checkpoint parameter names.
        from .auxiliary import RewardOutputDomains
        self.foot_head[-1] = RewardOutputDomains(nonnegative)

    def inputs(self, S, a, xi_n):
        return torch.cat((invariant_features(S), a, xi_n), -1)

    @staticmethod
    def targets(S, S1):
        """Frame-local increments from S to S1 (the regression targets)."""
        from ..tools.rotations import so3_log

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
        next_h = y[..., WM_OUT:]
        if self.visual:
            # Match the real GRU's range for supervision and imagined recurrence.
            next_h = torch.tanh(next_h)
        result = (p1, R1, v1, w1, q + dq, qd + dqd), next_h
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
        from ..tools.rotations import so3_log

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
