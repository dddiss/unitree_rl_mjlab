from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..tools.rotations import quat_to_mat, rot_angle_sq, yaw_mat

# Foot costs adapted from RoboParty/P-BFM; CC BY-NC 4.0.
# See ../perception/LICENSE-P-BFM and THIRD_PARTY_NOTICES.md.
# Canonical output/replay order. Values are costs, without reward weights or dt.
FOOT_REWARD_NAMES = ("feet_stumble", "feet_at_plane")

HEIGHT_SCAN_MIN = -5.0
HEIGHT_SCAN_MAX = 5.0

def clipped_height_clearance(
    sample_height: torch.Tensor,
    hit_height: torch.Tensor,
    distance: torch.Tensor,
    *,
    offset: float = 0.0,
) -> torch.Tensor:
    """Return finite sample-to-terrain clearance, filling invalid rays with the clip maximum."""

    clearance = sample_height - hit_height - float(offset)
    valid = (distance >= 0.0) & torch.isfinite(distance) & torch.isfinite(clearance)
    clearance = torch.where(valid, clearance, HEIGHT_SCAN_MAX)
    return clearance.clamp(HEIGHT_SCAN_MIN, HEIGHT_SCAN_MAX)


def fixed_sole_weights(
    local_xy: torch.Tensor,
    edge_weight: float = 0.1,
    center_weight: float = 1.0,
) -> torch.Tensor:
    """Return a terrain-independent 2-D sole-edge-to-center weight map."""

    if local_xy.ndim != 2 or local_xy.shape[-1] != 2 or local_xy.shape[0] == 0:
        raise ValueError(f"local_xy must be a non-empty (points, 2) tensor, got {tuple(local_xy.shape)}")
    if not 0.0 <= edge_weight <= center_weight:
        raise ValueError(
            f"Expected 0 <= edge_weight <= center_weight, got {edge_weight} and {center_weight}"
        )

    def axis_center_profile(values: torch.Tensor) -> torch.Tensor:
        span = values.max() - values.min()
        if float(span) <= 0.0:
            return torch.ones_like(values)
        fraction = ((values - values.min()) / span).clamp(0.0, 1.0)
        profile = 1.0 - 2.0 * torch.abs(fraction - 0.5)
        return profile / profile.max().clamp_min(torch.finfo(values.dtype).eps)

    x_profile = axis_center_profile(local_xy[:, 0])
    y_profile = axis_center_profile(local_xy[:, 1])
    center_profile = torch.minimum(x_profile, y_profile)
    return edge_weight + (center_weight - edge_weight) * center_profile


def feet_stumble_penalty(
    force_history: torch.Tensor,
    horizontal_ratio: float = 1.0,
    force_threshold: float = 20.0,
    force_scale: float = 100.0,
    max_penalty: float = 1.0,
) -> torch.Tensor:
    """Penalize the largest horizontal-dominant foot contact in a policy step."""

    if force_history.ndim != 4 or force_history.shape[-1] != 3:
        raise ValueError(f"Expected force history shaped (N, contacts, history, 3), got {tuple(force_history.shape)}")
    if force_scale <= 0.0 or max_penalty <= 0.0:
        raise ValueError("force_scale and max_penalty must be positive")
    if force_history.shape[1] == 0 or force_history.shape[2] == 0:
        return torch.zeros(force_history.shape[0], dtype=force_history.dtype, device=force_history.device)
    force_xy = torch.linalg.norm(force_history[..., :2], dim=-1)
    force_z = torch.abs(force_history[..., 2])
    excess = torch.relu(force_xy - horizontal_ratio * force_z - force_threshold)
    return torch.clamp(excess / force_scale, max=max_penalty).amax(dim=(1, 2))


def _support_deficiency(
    clearance: torch.Tensor,
    point_weights: torch.Tensor | None,
    *,
    height_tolerance: float,
    transition_width: float,
) -> torch.Tensor:
    width = max(float(transition_width), 1.0e-6)
    support = torch.sigmoid((height_tolerance - clearance) / width)
    unsupported = 1.0 - support
    boundary_eps = 8.0 * torch.finfo(clearance.dtype).eps
    unsupported = torch.where(
        clearance > height_tolerance + boundary_eps,
        unsupported,
        torch.zeros_like(unsupported),
    )
    if point_weights is None:
        return unsupported.mean(dim=-1)
    weights = point_weights.to(device=unsupported.device, dtype=unsupported.dtype)
    return (unsupported * weights).sum(dim=-1) / weights.sum().clamp_min(torch.finfo(unsupported.dtype).eps)


def feet_at_plane_penalty(
    clearance: torch.Tensor,
    is_contact: torch.Tensor,
    point_weights: torch.Tensor,
    *,
    height_tolerance: float = 0.030,
    transition_width: float = 0.005,
) -> torch.Tensor:
    """Penalize unsupported sole area for both feet on every terrain type."""

    if clearance.ndim != 3 or is_contact.shape != clearance.shape[:2]:
        raise ValueError(
            f"Expected clearance=(N, 2, points), is_contact=(N, 2), got "
            f"{tuple(clearance.shape)}, {tuple(is_contact.shape)}"
        )
    if point_weights.ndim != 1 or point_weights.numel() != clearance.shape[-1]:
        raise ValueError(f"Expected one weight per terrain point, got {tuple(point_weights.shape)}")
    weighted = _support_deficiency(
        clearance,
        point_weights,
        height_tolerance=height_tolerance,
        transition_width=transition_width,
    )
    return torch.sum(weighted * is_contact.to(weighted.dtype), dim=1)


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


# Components below own the differentiable counterpart of each measured term.
# Register new terms here (or in another imported module), not in Algorithm.
from ..core.components import Prediction, analytic_reward, reward_component
import math


def _root_position(ctx, std):
  return torch.exp(-(ctx.reference_root[0] - ctx.state[0]).square().sum(-1) / std**2)


@analytic_reward(requires_height=True)
def motion_terrain_anchor_position_error_exp(ctx, std: float) -> torch.Tensor:
  """Track reference XY and terrain-adjusted Z in the same frame as state.

  The offset is queried at the placed reference XY, independently of predicted
  state. Leave the shared reference (including velocities) unchanged.
  """
  if ctx.height_offset is None:
    raise ValueError("Terrain anchor reward requires a reference height_offset")
  reference = ctx.reference_root[0]
  target = torch.cat((reference[..., :2],
                      (reference[..., 2] + ctx.height_offset).unsqueeze(-1)), -1)
  return torch.exp(-(target - ctx.state[0]).square().sum(-1) / std**2)


def _root_orientation(ctx, std):
  return torch.exp(-rot_angle_sq(ctx.reference_root[1], ctx.state[1]) / std**2)


def _body_position(ctx, std, body_names=None):
  return torch.exp(-(ctx.aligned_reference[0] - ctx.bodies[0]).square().sum(-1).mean(-1) / std**2)


def _body_orientation(ctx, std, body_names=None):
  return torch.exp(-rot_angle_sq(ctx.aligned_reference[1], ctx.bodies[1]).mean(-1) / std**2)


def _body_linear_velocity(ctx, std, body_names=None):
  return torch.exp(-(ctx.reference_bodies[2] - ctx.bodies[2]).square().sum(-1).mean(-1) / std**2)


def _body_angular_velocity(ctx, std, body_names=None):
  return torch.exp(-(ctx.reference_bodies[3] - ctx.bodies[3]).square().sum(-1).mean(-1) / std**2)


def _action_rate(ctx):
  return (ctx.action - ctx.previous_action).square().sum(-1)


def _joint_limit(ctx):
  q = ctx.state[4]
  return (torch.relu(q - ctx.mdp.soft_hi) + torch.relu(ctx.mdp.soft_lo - q)).sum(-1)


def _prepare_body_subset(params, body_names):
  selected = tuple(params["body_names"])
  if not selected or any(body not in body_names for body in selected):
    raise ValueError("body_names must select tracked robot bodies")
  if body_names[0] in selected:
    raise ValueError("exclude the root; it has a separate tracking reward")
  std, tolerance = params["std"], params.get("tolerance", 0.0)
  if not math.isfinite(std) or std <= 0 or not math.isfinite(tolerance) or tolerance < 0:
    raise ValueError("std must be positive and tolerance nonnegative, both finite")
  return dict(indices=tuple(i for i, b in enumerate(body_names) if b in selected),
              std=std, tolerance=tolerance)


def _root_relative_position(ctx, indices, std, tolerance=0.0):
  return root_relative_body_position_reward(
    ctx.bodies[0][..., indices, :], *ctx.state[:2],
    ctx.reference_bodies[0][..., indices, :], *ctx.reference_root[:2],
    std=std, tolerance=terrain_tolerance_radius(tolerance, ctx.terrain_tolerance))


def _tolerant_orientation(ctx, indices, std, tolerance=0.0):
  return body_orientation_tolerance_reward(
    ctx.bodies[1][..., indices, :, :], ctx.state[1],
    ctx.reference_bodies[1][..., indices, :, :], ctx.reference_root[1],
    std=std, tolerance=terrain_tolerance_radius(tolerance, ctx.terrain_tolerance))


# Defaults also read old programmatic MdpConfig/flat checkpoints. The live
# environment resolves by function identity, so term labels can be renamed.
LEGACY_REWARDS = {
  "motion_global_root_pos": (motion_global_anchor_position_error_exp, _root_position, "root_pos_std"),
  "motion_global_root_ori": (motion_global_anchor_orientation_error_exp, _root_orientation, "root_ori_std"),
  "motion_body_pos": (motion_relative_body_position_error_exp, _body_position, "body_pos_std"),
  "motion_body_ori": (motion_relative_body_orientation_error_exp, _body_orientation, "body_ori_std"),
  "motion_body_lin_vel": (motion_global_body_linear_velocity_error_exp, _body_linear_velocity, "body_linvel_std"),
  "motion_body_ang_vel": (motion_global_body_angular_velocity_error_exp, _body_angular_velocity, "body_angvel_std"),
  "action_rate_l2": (supermimic_action_rate, _action_rate, None),
  "joint_limit": (supermimic_joint_limit, _joint_limit, None),
}
for _measured, _tensor, _ in LEGACY_REWARDS.values():
  reward_component(_tensor)(_measured)
  _measured.use_tensor_measurement = True
reward_component(_root_relative_position, prepare=_prepare_body_subset)(motion_root_relative_body_position_error_exp)
reward_component(_tolerant_orientation, prepare=_prepare_body_subset)(motion_body_orientation_tolerance_error_exp)

reward_component(prediction=Prediction(loss_weight_field="feet_stumble_loss_weight"))(feet_stumble)
reward_component(prediction=Prediction(loss_weight_field="feet_at_plane_loss_weight", loss_weight=0.0),
                 sense_before_reward=True)(feet_at_plane)
LEGACY_SENSOR_REWARDS = {"feet_stumble": feet_stumble, "feet_at_plane": feet_at_plane}
