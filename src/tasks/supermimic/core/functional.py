"""Shared differentiable MDP for measured and imagined states.

Adapted from vibed (THIRD_PARTY_NOTICES.md). f is the CURRENT target,
not vibed's previous reference index. No environment origins are stored here.
"""
from dataclasses import dataclass, field

import torch

from ..tools.rotations import mat_to_6d, quat_to_mat, rot_angle_sq, yaw_mat
from .state import GOAL_FRAME_DIM, invariant_features
from .context import StepContext
from .components import RewardBinding, FailureBinding, reward_binding, failure_binding
from ..mdp.terminations import HEIGHT_FAILURE, TILT_FAILURE, anchor_height_error, anchor_tilt_error
from ..mdp.rewards import (
  LEGACY_REWARDS, LEGACY_SENSOR_REWARDS,
  motion_root_relative_body_position_error_exp, motion_body_orientation_tolerance_error_exp,
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

  # None supports the original tensor-only MdpConfig constructor. Live manager
  # configurations always resolve explicit components once at startup.
  reward_components: tuple | None = None
  failure_components: tuple | None = None
  unsupported_rewards: tuple[str, ...] = ()

  @classmethod
  def from_env_cfg(cls, rewards, terminations, step_dt, *, body_names=()):
    from types import SimpleNamespace
    cfg = cls(weights={k: v.weight for k, v in rewards.items() if v is not None}, step_dt=step_dt)
    bindings, unsupported = [], []
    for name, term in rewards.items():
      if term is None:
        continue
      # Compatibility for old config snapshots without callable metadata.
      if not hasattr(term, "func"):
        measured = (LEGACY_REWARDS[name][0] if name in LEGACY_REWARDS
                    else LEGACY_SENSOR_REWARDS.get(name))
        term = SimpleNamespace(func=measured, params=term.params)
      binding = reward_binding(name, term, body_names)
      if binding is not None:
        bindings.append(binding)
      else:
        unsupported.append(name)
      if (name in LEGACY_REWARDS and LEGACY_REWARDS[name][2] is not None
          and binding is not None
          and binding.component is LEGACY_REWARDS[name][0].supermimic_reward):
        setattr(cfg, LEGACY_REWARDS[name][2], term.params["std"])
      # Retain these read-only compatibility views for existing callers/tests.
      if getattr(term, "func", None) is motion_root_relative_body_position_error_exp:
        cfg.body_position_rewards[name] = binding.params
      if getattr(term, "func", None) is motion_body_orientation_tolerance_error_exp:
        cfg.body_orientation_rewards[name] = binding.params
    cfg.reward_components = tuple(bindings)
    cfg.unsupported_rewards = tuple(unsupported)
    failures = []
    cfg.term_height = cfg.term_tilt = None
    for name, term in terminations.items():
      if term is None or getattr(term, "time_out", False):
        continue
      if not hasattr(term, "func") and name in ("anchor_pos", "anchor_ori"):
        component = HEIGHT_FAILURE if name == "anchor_pos" else TILT_FAILURE
        binding = FailureBinding(name, component, term.params["threshold"])
      else:
        binding = failure_binding(name, term, body_names)
      if binding is None:
        raise ValueError(f"{name}: failure needs a differentiable SuperMimic component; "
                         "declare a soft failure error or mark a genuine truncation time_out=True")
      failures.append(binding)
      if binding.component is HEIGHT_FAILURE:
        cfg.term_height = binding.threshold
      if binding.component is TILT_FAILURE:
        cfg.term_tilt = binding.threshold
    cfg.failure_components = tuple(failures)
    return cfg

  def resolved_rewards(self):
    # Preserve the legacy summation order (including its zero-weight terms).
    result = {name: RewardBinding(name, measured.supermimic_reward,
                                  {} if attr is None else {"std": getattr(self, attr)})
              for name, (measured, _, attr) in LEGACY_REWARDS.items()}
    if self.reward_components is not None:
      result.update((binding.name, binding) for binding in self.reward_components)
    else:
      for name, function in LEGACY_SENSOR_REWARDS.items():
        if name in self.weights:
          result[name] = RewardBinding(name, function.supermimic_reward)
      for entries, function in (
        (self.body_position_rewards, motion_root_relative_body_position_error_exp),
        (self.body_orientation_rewards, motion_body_orientation_tolerance_error_exp),
      ):
        result.update((name, RewardBinding(name, function.supermimic_reward, params))
                      for name, params in entries.items())
    return tuple(result.values())

  def resolved_failures(self):
    if self.failure_components is not None:
      return self.failure_components
    return tuple(FailureBinding(name, component, threshold) for name, component, threshold in (
      ("anchor_pos", HEIGHT_FAILURE, self.term_height),
      ("anchor_ori", TILT_FAILURE, self.term_tilt),
    ) if threshold is not None)


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

  @property
  def prediction_components(self):
    bindings = (self.cfg.resolved_rewards() if self.cfg.reward_components is None
                else self.cfg.reward_components)
    return tuple(binding for binding in bindings
                 if binding.component.prediction is not None and binding.name in self.cfg.weights)

  @property
  def requires_height_offset(self):
    return any(binding.component.requires_height for binding in self.cfg.resolved_failures())

  def validate_imagination_rewards(self, learned_rewards=()):
    analytic = {binding.name for binding in self.cfg.resolved_rewards()
                if binding.component.tensor is not None
                and binding.name not in self.cfg.unsupported_rewards}
    unsupported = [name for name, weight in self.cfg.weights.items()
                   if weight != 0 and name not in analytic and name not in learned_rewards]
    if unsupported:
      raise ValueError(f"No imagined reward implementation for {unsupported}. "
                       "Register a tensor component or matching WM reward heads before enabling them.")

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
    ctx = StepContext(self, state, f, action, a_prev, terrain_tolerance)
    return {binding.name: binding.value(ctx) for binding in self.cfg.resolved_rewards()
            if binding.component.tensor is not None}

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
    # Legacy diagnostics API. Composition itself uses the bindings below.
    ctx = StepContext(self, state, f, height_offset=height_offset)
    return anchor_height_error(ctx), anchor_tilt_error(ctx)

  def termination(self, state, f, *, height_offset=None):
    ctx = StepContext(self, state, f, height_offset=height_offset)
    done = torch.zeros_like(f, dtype=torch.bool)
    for binding in self.cfg.resolved_failures():
      done = done | binding.hard(ctx)
    return done

  def failure_terms(self, state, f, *, height_offset=None):
    ctx = StepContext(self, state, f, height_offset=height_offset)
    return {binding.name: binding.hard(ctx) for binding in self.cfg.resolved_failures()}

  def soft_termination(self, state, f, *, height_offset=None):
    ctx = StepContext(self, state, f, height_offset=height_offset)
    alive = torch.ones_like(f, dtype=state[0].dtype)
    for binding in self.cfg.resolved_failures():
      alive = alive * (1 - binding.soft(ctx))
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
