"""Shared differentiable MDP for measured and imagined states.

Adapted from vibed (THIRD_PARTY_NOTICES.md). f is the CURRENT target,
not vibed's previous reference index. No environment origins are stored here.
"""
from dataclasses import dataclass, field

import torch

from .rotations import mat_to_6d, quat_to_mat, rot_angle_sq, yaw_mat
from .state import GOAL_FRAME_DIM, invariant_features

REWARD_WEIGHTS = {
  "motion_global_root_pos": 0.5, "motion_global_root_ori": 0.5,
  "motion_body_pos": 1.0, "motion_body_ori": 1.0,
  "motion_body_lin_vel": 1.0, "motion_body_ang_vel": 1.0,
  "action_rate_l2": -0.1, "joint_limit": -2.0,
}


@dataclass
class MdpConfig:
  weights: dict[str, float] = field(default_factory=lambda: dict(REWARD_WEIGHTS))
  root_pos_std: float = 0.3
  root_ori_std: float = 0.4
  body_pos_std: float = 0.3
  body_ori_std: float = 0.4
  body_linvel_std: float = 1.0
  body_angvel_std: float = 3.14
  # None disables the corresponding hard AND imagined soft termination.
  term_height: float | None = 0.25
  term_tilt: float | None = 0.8
  step_dt: float = 0.02

  @classmethod
  def from_env_cfg(cls, rewards, terminations, step_dt):
    """Mirror enabled manager terms without requiring every optional component."""
    cfg = cls(weights={k: v.weight for k, v in rewards.items() if v is not None},
              step_dt=step_dt)
    for key, attribute in (("motion_global_root_pos", "root_pos_std"),
                           ("motion_global_root_ori", "root_ori_std"),
                           ("motion_body_pos", "body_pos_std"),
                           ("motion_body_ori", "body_ori_std"),
                           ("motion_body_lin_vel", "body_linvel_std"),
                           ("motion_body_ang_vel", "body_angvel_std")):
      term = rewards.get(key)
      if term is not None:
        setattr(cfg, attribute, term.params["std"])
    for key, attribute in (("anchor_pos", "term_height"), ("anchor_ori", "term_tilt")):
      term = terminations.get(key)
      setattr(cfg, attribute, None if term is None else term.params["threshold"])
    return cfg


class FunctionalMDP:
  def __init__(self, motion, kin, default_q, body_indexes, root_body_index,
               soft_limits=None, cfg=None):
    self.motion, self.kin = motion, kin
    self.default_q = default_q
    self.body_indexes = torch.as_tensor(body_indexes, device=motion.device, dtype=torch.long)
    self.root_body_index = int(root_body_index)
    self.cfg = cfg or MdpConfig(step_dt=motion.step_dt)
    if soft_limits is None:
      mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) * 0.45
      self.soft_lo, self.soft_hi = mid - half, mid + half
    else:
      self.soft_lo, self.soft_hi = soft_limits[:, 0], soft_limits[:, 1]
    self.obs_noise = torch.tensor([0.01] * 29 + [0.5] * 29 + [0.2] * 3
                                  + [0.05] * 3 + [0.0] * 29,
                                  device=motion.device)

  @property
  def max_step_reward(self):
    return sum(max(0.0, w) for w in self.cfg.weights.values()) * self.cfg.step_dt

  def validate_imagination_rewards(self, learned_rewards=()):
    unsupported = [name for name, weight in self.cfg.weights.items()
                   if weight != 0 and name not in REWARD_WEIGHTS and name not in learned_rewards]
    if unsupported:
      raise ValueError(
        f"No imagined reward implementation for {unsupported}. "
        "Keep their weights at 0 until matching WM reward heads are implemented."
      )

  def reference_root(self, f):
    m, b = self.motion, self.root_body_index
    return (m.body_pos_w[f, b], quat_to_mat(m.body_quat_w[f, b]),
            m.body_lin_vel_w[f, b], m.body_ang_vel_w[f, b])

  def goal(self, state, f, count=10):
    """Encode f..f+K-1 as reference q, qd and relative root rotation.

    Each 64D frame uses the same current robot root orientation from state.
    Reference indices clamp within their clip, including terminal bootstrap.
    Imagined states retain gradients through the relative rotation features.
    """
    m = self.motion
    clip = m.clip_id[f]
    targets = torch.minimum(f[..., None] + torch.arange(count, device=f.device),
                            m.clip_end[clip][..., None])
    reference_r = quat_to_mat(m.body_quat_w[targets, self.root_body_index])
    robot_rt = state[1].transpose(-1, -2).unsqueeze(-3)
    goal = torch.cat((m.joint_pos[targets], m.joint_vel[targets],
                      mat_to_6d(robot_rt @ reference_r)), -1)
    return goal.flatten(-2)

  def obs(self, state, a_prev, xi, f):
    """93D proprioception; reference-relative orientation is encoded in goal."""
    _, r, _, w, q, qd = state
    rt = r.transpose(-1, -2)
    return torch.cat((q + xi[..., 1:30] - self.default_q, qd,
                      (rt @ w.unsqueeze(-1)).squeeze(-1),
                      -r[..., 2, :], a_prev), -1)

  def add_noise(self, obs):
    return obs + (2 * torch.rand_like(obs) - 1) * self.obs_noise

  def add_goal_noise(self, goal):
    """Add U(-0.05, 0.05) to each frame's 6D orientation for policy input only."""
    frames = goal.unflatten(-1, (-1, GOAL_FRAME_DIM))
    orientation = frames[..., -6:]
    noisy_orientation = orientation + (2 * torch.rand_like(orientation) - 1) * 0.05
    return torch.cat((frames[..., :-6], noisy_orientation), -1).flatten(-2)

  def aligned_reference(self, state, f):
    p, r = state[:2]
    rp, rr, _, _ = self.reference_root(f)
    bp = self.motion.body_pos_w[f][..., self.body_indexes, :]
    br = quat_to_mat(self.motion.body_quat_w[f][..., self.body_indexes, :])
    delta = yaw_mat(r @ rr.transpose(-1, -2))
    anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
    return (anchor.unsqueeze(-2) + (delta.unsqueeze(-3) @ (bp - rp.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1),
            delta.unsqueeze(-3) @ br)

  def reward_terms(self, state, f, action, a_prev):
    c, m = self.cfg, self.motion
    p, r, _, _, q, _ = state
    bp, br, bv, bw = self.kin.forward(*state)
    rp, rr, _, _ = self.reference_root(f)
    ref_bp, ref_br = self.aligned_reference(state, f)
    rv = m.body_lin_vel_w[f][..., self.body_indexes, :]
    rw = m.body_ang_vel_w[f][..., self.body_indexes, :]
    return {
      "motion_global_root_pos": torch.exp(-(rp - p).square().sum(-1) / c.root_pos_std**2),
      "motion_global_root_ori": torch.exp(-rot_angle_sq(rr, r) / c.root_ori_std**2),
      "motion_body_pos": torch.exp(-(ref_bp - bp).square().sum(-1).mean(-1) / c.body_pos_std**2),
      "motion_body_ori": torch.exp(-rot_angle_sq(ref_br, br).mean(-1) / c.body_ori_std**2),
      "motion_body_lin_vel": torch.exp(-(rv - bv).square().sum(-1).mean(-1) / c.body_linvel_std**2),
      "motion_body_ang_vel": torch.exp(-(rw - bw).square().sum(-1).mean(-1) / c.body_angvel_std**2),
      "action_rate_l2": (action - a_prev).square().sum(-1),
      "joint_limit": (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1),
    }

  def reward(self, state, f, action, a_prev):
    terms = self.reward_terms(state, f, action, a_prev)
    return sum(self.cfg.weights.get(k, 0.0) * value for k, value in terms.items()) * self.cfg.step_dt

  def termination_errors(self, state, f):
    rp, rr, _, _ = self.reference_root(f)
    return (rp[..., 2] - state[0][..., 2]).abs(), (rr[..., 2, 2] - state[1][..., 2, 2]).abs()

  def termination(self, state, f):
    z, tilt = self.termination_errors(state, f)
    done = torch.zeros_like(z, dtype=torch.bool)
    if self.cfg.term_height is not None:
      done = done | (z > self.cfg.term_height)
    if self.cfg.term_tilt is not None:
      done = done | (tilt > self.cfg.term_tilt)
    return done

  def soft_termination(self, state, f):
    z, tilt = self.termination_errors(state, f)
    alive = torch.ones_like(z)
    if self.cfg.term_height is not None:
      alive = alive * (1 - torch.sigmoid((z - self.cfg.term_height) / 0.02))
    if self.cfg.term_tilt is not None:
      alive = alive * (1 - torch.sigmoid((tilt - self.cfg.term_tilt) / 0.04))
    return 1 - alive

  def critic_features(self, state, f, normalized_xi, a_prev):
    """Privileged state/errors and the action that produced this state.

    Root-relative orientation is already present in the first goal frame, so
    it is not repeated here. The previous action is a single 29D vector.
    """
    p, r, _, _, q, _ = state
    rp, _, _, _ = self.reference_root(f)
    bp = self.kin.forward(*state)[0]
    ref_bp, _ = self.aligned_reference(state, f)
    ht = yaw_mat(r).transpose(-1, -2)
    return torch.cat((invariant_features(state), self.motion.joint_pos[f] - q,
                      (ht.unsqueeze(-3) @ (ref_bp - bp).unsqueeze(-1)).squeeze(-1).flatten(-2),
                      (ht @ (rp - p).unsqueeze(-1)).squeeze(-1),
                      a_prev, normalized_xi), -1)
