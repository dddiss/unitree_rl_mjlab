from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .foot_rewards import clipped_height_clearance, feet_at_plane_penalty, feet_stumble_penalty
from .rotations import quat_to_mat, rot_angle_sq, yaw_mat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor
  from .commands import MotionCommand


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


def terrain_tolerance_radius(tolerance, enabled):
  """Resolve the configured radius per sample, keeping flat rows strict."""
  if tolerance == 0.0:
    return 0.0
  if enabled is None:
    raise ValueError("Terrain-dependent body rewards require the sampled terrain_tolerance flag")
  return enabled.to(dtype=torch.float32) * tolerance


def root_relative_body_position_reward(
  body_pos, root_pos, root_rot, reference_pos, reference_root_pos,
  reference_root_rot, *, std: float, tolerance: float = 0.0,
) -> torch.Tensor:
  """XYZ root-relative tracking, with the existing yaw alignment.

  Tolerance is a distance in metres, applied per body before squaring/averaging.
  Inputs may have any leading batch/trajectory dimensions. No terrain query is
  needed, so the same formula differentiates through imagined states and FK.
  """
  alignment = yaw_mat(root_rot @ reference_root_rot.transpose(-1, -2))
  reference_relative = reference_pos - reference_root_pos.unsqueeze(-2)
  aligned = (alignment.unsqueeze(-3) @ reference_relative.unsqueeze(-1)).squeeze(-1)
  error = torch.linalg.vector_norm(body_pos - root_pos.unsqueeze(-2) - aligned, dim=-1)
  radius = tolerance.unsqueeze(-1) if isinstance(tolerance, torch.Tensor) else tolerance
  residual = torch.relu(error - radius)
  return torch.exp(-residual.square().mean(-1) / std**2)


def body_orientation_tolerance_reward(
  body_rot, root_rot, reference_rot, reference_root_rot, *,
  std: float, tolerance: float = 0.0,
) -> torch.Tensor:
  """Yaw-aligned body rotation tracking; std and tolerance are in radians."""
  alignment = yaw_mat(root_rot @ reference_root_rot.transpose(-1, -2))
  aligned = alignment.unsqueeze(-3) @ reference_rot
  # rot_angle_sq clamps acos away from its singular endpoints, also keeping
  # the square root's gradient finite for exactly matching rotations.
  error = rot_angle_sq(aligned, body_rot).sqrt()
  radius = tolerance.unsqueeze(-1) if isinstance(tolerance, torch.Tensor) else tolerance
  residual = torch.relu(error - radius)
  return torch.exp(-residual.square().mean(-1) / std**2)


def motion_root_relative_body_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float,
  body_names: tuple[str, ...], tolerance: float = 0.0,
) -> torch.Tensor:
  """Measured counterpart of root_relative_body_position_reward; target is f."""
  command = env.command_manager.get_term(command_name)
  indexes = _get_body_indexes(command, body_names)
  root = command.root_body_index
  tolerance = terrain_tolerance_radius(tolerance, env.terrain_tolerance) if tolerance else 0.0
  return root_relative_body_position_reward(
    command.robot_body_pos_w[:, indexes], command.robot.data.body_link_pos_w[:, root],
    quat_to_mat(command.robot.data.body_link_quat_w[:, root]),
    command.body_pos_w[:, indexes], command.body_pos_w[:, 0],
    quat_to_mat(command.body_quat_w[:, 0]), std=std, tolerance=tolerance,
  )


def motion_body_orientation_tolerance_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float,
  body_names: tuple[str, ...], tolerance: float = 0.0,
) -> torch.Tensor:
  """Measured counterpart of body_orientation_tolerance_reward; target is f."""
  command = env.command_manager.get_term(command_name)
  indexes = _get_body_indexes(command, body_names)
  tolerance = terrain_tolerance_radius(tolerance, env.terrain_tolerance) if tolerance else 0.0
  return body_orientation_tolerance_reward(
    quat_to_mat(command.robot_body_quat_w[:, indexes]),
    quat_to_mat(command.robot.data.body_link_quat_w[:, command.root_body_index]),
    quat_to_mat(command.body_quat_w[:, indexes]), quat_to_mat(command.body_quat_w[:, 0]),
    std=std, tolerance=tolerance,
  )


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
