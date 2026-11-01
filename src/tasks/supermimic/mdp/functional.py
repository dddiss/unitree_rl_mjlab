"""Shared differentiable MDP for measured and imagined states.

Adapted from vibed (THIRD_PARTY_NOTICES.md). f is the CURRENT target,
not vibed's previous reference index. No environment origins are stored here.
"""
from dataclasses import dataclass, field
import math

import torch

from .rotations import mat_to_6d, quat_to_mat, rot_angle_sq, yaw_mat
from .state import GOAL_FRAME_DIM, invariant_features
from .rewards import (
  root_relative_body_position_reward, body_orientation_tolerance_reward,
  motion_root_relative_body_position_error_exp, motion_body_orientation_tolerance_error_exp,
  terrain_tolerance_radius,
)

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
  # New manager-configured terms; legacy body rewards remain available unchanged.
  body_position_rewards: dict[str, dict] = field(default_factory=dict)
  body_orientation_rewards: dict[str, dict] = field(default_factory=dict)

  @classmethod
  def from_env_cfg(cls, rewards, terminations, step_dt, *, body_names=()):
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
    for name, term in rewards.items():
      if term is None:
        continue
      for function, destination in (
        (motion_root_relative_body_position_error_exp, cfg.body_position_rewards),
        (motion_body_orientation_tolerance_error_exp, cfg.body_orientation_rewards),
      ):
        if getattr(term, "func", None) is not function:
          continue
        selected = tuple(term.params["body_names"])
        if not selected or any(body not in body_names for body in selected):
          raise ValueError(f"{name}: body_names must select tracked robot bodies")
        if body_names[0] in selected:
          raise ValueError(f"{name}: exclude the root; it has a separate tracking reward")
        std, tolerance = term.params["std"], term.params.get("tolerance", 0.0)
        if not math.isfinite(std) or std <= 0 or not math.isfinite(tolerance) or tolerance < 0:
          raise ValueError(f"{name}: std must be positive and tolerance nonnegative, both finite")
        destination[name] = dict(indices=tuple(i for i, b in enumerate(body_names) if b in selected),
                                 std=std, tolerance=tolerance)
    return cfg


class FunctionalMDP:
  def __init__(self, motion, kin, default_q, body_indexes, root_body_index,
               soft_limits=None, cfg=None):
    self.motion, self.kin = motion, kin
    self.default_q = default_q
    self.body_indexes = torch.as_tensor(body_indexes, device=motion.device, dtype=torch.long)
    self.root_body_index = int(root_body_index)
    self.cfg = cfg or MdpConfig(step_dt=motion.step_dt)
    # Reference-XY ray query used only for height termination (None = plane).
    self.termination_ground = None
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
                   if weight != 0 and name not in REWARD_WEIGHTS and name not in learned_rewards
                   and name not in self.cfg.body_position_rewards
                   and name not in self.cfg.body_orientation_rewards]
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

  def reward_terms(self, state, f, action, a_prev, *, terrain_tolerance=None):
    c, m = self.cfg, self.motion
    p, r, _, _, q, _ = state
    bp, br, bv, bw = self.kin.forward(*state)
    rp, rr, _, _ = self.reference_root(f)
    ref_bp, ref_br = self.aligned_reference(state, f)
    rv = m.body_lin_vel_w[f][..., self.body_indexes, :]
    rw = m.body_ang_vel_w[f][..., self.body_indexes, :]
    terms = {
      "motion_global_root_pos": torch.exp(-(rp - p).square().sum(-1) / c.root_pos_std**2),
      "motion_global_root_ori": torch.exp(-rot_angle_sq(rr, r) / c.root_ori_std**2),
      "motion_body_pos": torch.exp(-(ref_bp - bp).square().sum(-1).mean(-1) / c.body_pos_std**2),
      "motion_body_ori": torch.exp(-rot_angle_sq(ref_br, br).mean(-1) / c.body_ori_std**2),
      "motion_body_lin_vel": torch.exp(-(rv - bv).square().sum(-1).mean(-1) / c.body_linvel_std**2),
      "motion_body_ang_vel": torch.exp(-(rw - bw).square().sum(-1).mean(-1) / c.body_angvel_std**2),
      "action_rate_l2": (action - a_prev).square().sum(-1),
      "joint_limit": (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1),
    }
    # Only dispatch/configuration lives here; the new formulas live in rewards.py
    # and are shared by the measured manager adapters and imagined FK states.
    if c.body_position_rewards or c.body_orientation_rewards:
      reference_pos = m.body_pos_w[f][..., self.body_indexes, :]
      reference_rot = quat_to_mat(m.body_quat_w[f][..., self.body_indexes, :])
      for name, params in c.body_position_rewards.items():
        ids = params["indices"]
        terms[name] = root_relative_body_position_reward(
          bp[..., ids, :], p, r, reference_pos[..., ids, :], rp, rr,
          std=params["std"], tolerance=terrain_tolerance_radius(params["tolerance"], terrain_tolerance),
        )
      for name, params in c.body_orientation_rewards.items():
        ids = params["indices"]
        terms[name] = body_orientation_tolerance_reward(
          br[..., ids, :, :], r, reference_rot[..., ids, :, :], rr,
          std=params["std"], tolerance=terrain_tolerance_radius(params["tolerance"], terrain_tolerance),
        )
    return terms

  def reward(self, state, f, action, a_prev, *, terrain_tolerance=None):
    terms = self.reward_terms(state, f, action, a_prev, terrain_tolerance=terrain_tolerance)
    return sum(self.cfg.weights.get(k, 0.0) * value for k, value in terms.items()) * self.cfg.step_dt

  @torch.no_grad()
  def termination_height_offset(self, f, env_origin=None, xy_offset=None):
    """Reference ground in the same environment-relative Z frame as state.

    Reference XY is fixed by the sampled episode, never by a predicted state.
    Query future frames before BPTT, without requiring recorded future rows.
    """
    if self.termination_ground is None:
      ground = torch.zeros_like(f, dtype=torch.float32)
    else:
      if env_origin is None or xy_offset is None:
        raise ValueError("Terrain termination requires the sampled episode's origin and XY offset")
      xy = self.motion.body_pos_w[f, self.root_body_index, :2] + env_origin[..., :2] + xy_offset
      ground = self.termination_ground(xy)
    return ground if env_origin is None else ground - env_origin[..., 2]

  def termination_errors(self, state, f, *, height_offset=None):
    rp, rr, _, _ = self.reference_root(f)
    target_z = rp[..., 2] if height_offset is None else rp[..., 2] + height_offset
    return (target_z - state[0][..., 2]).abs(), (rr[..., 2, 2] - state[1][..., 2, 2]).abs()

  def termination(self, state, f, *, height_offset=None):
    z, tilt = self.termination_errors(state, f, height_offset=height_offset)
    done = torch.zeros_like(z, dtype=torch.bool)
    if self.cfg.term_height is not None:
      done = done | (z > self.cfg.term_height)
    if self.cfg.term_tilt is not None:
      done = done | (tilt > self.cfg.term_tilt)
    return done

  def soft_termination(self, state, f, *, height_offset=None):
    z, tilt = self.termination_errors(state, f, height_offset=height_offset)
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
