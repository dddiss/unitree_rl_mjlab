from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor

from .commands import MotionCommand
from .foot_rewards import clipped_height_clearance, feet_at_plane_penalty, feet_stumble_penalty

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  return env.shared_reward_terms()["motion_global_root_pos"]


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  return env.shared_reward_terms()["motion_global_root_ori"]


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  return env.shared_reward_terms()["motion_body_pos"]


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  return env.shared_reward_terms()["motion_body_ori"]


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  return env.shared_reward_terms()["motion_body_lin_vel"]


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  return env.shared_reward_terms()["motion_body_ang_vel"]


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)

def supermimic_action_rate(env):
  return env.shared_reward_terms()["action_rate_l2"]


def supermimic_joint_limit(env, asset_cfg=None):
  return env.shared_reward_terms()["joint_limit"]


def feet_stumble(
  env, sensor_name: str = "foot_collision_contacts", horizontal_ratio: float = 1.0,
  force_threshold: float = 20.0, force_scale: float = 100.0, max_penalty: float = 1.0,
) -> torch.Tensor:
  """P-BFM cost for the strongest horizontal-dominant foot impact this step.

  Requires world-frame, per-geometry contact history covering the decimation.
  Its unweighted value supervises the WM's feet_stumble prediction head.
  """
  history = env.scene[sensor_name].data.force_history
  if history is None:
    raise ValueError("feet_stumble requires a contact sensor with force history")
  return feet_stumble_penalty(history, horizontal_ratio, force_threshold, force_scale, max_penalty)


def feet_at_plane(
  env, sensor_name: str = "foot_height_scanners", contact_sensor_name: str = "foot_support_contacts",
  foot_body_names: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link"),
  height_tolerance: float = 0.03, transition_width: float = 0.005,
  contact_threshold: float = 1.0,
) -> torch.Tensor:
  """P-BFM unsupported-sole cost, gated by each foot's vertical net contact.

  The scanner must have sensed the post-action state, before target advancement.
  Swing feet are unpenalized. A ray miss is treated as unsupported, not as ground.
  """
  sensor = env.scene[sensor_name]
  contacts = env.scene[contact_sensor_name]
  history = contacts.data.force_history
  if history is None:
    raise ValueError("feet_at_plane requires a contact sensor with force history")
  names = [name.rsplit("/", 1)[-1] for name in contacts.primary_names]
  indices = [names.index(name) for name in foot_body_names]
  is_contact = history[:, indices, :, 2].abs().amax(dim=-1) > contact_threshold
  shape = (env.num_envs, len(foot_body_names), sensor.num_rays_per_frame)
  data = sensor.data
  clearance = clipped_height_clearance(
    sensor.sample_points_w[..., 2].reshape(shape),
    data.hit_pos_w[..., 2].reshape(shape), data.distances.reshape(shape),
  )
  return feet_at_plane_penalty(clearance, is_contact, sensor.point_weights,
                               height_tolerance=height_tolerance, transition_width=transition_width)
