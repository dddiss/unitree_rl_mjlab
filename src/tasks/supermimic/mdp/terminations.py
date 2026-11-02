from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from ..core.components import FailureComponent
from .rewards import _get_body_indexes

if TYPE_CHECKING:
  from .commands import MotionCommand
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def anchor_height_error(ctx):
  z = ctx.reference_root[0][..., 2]
  if ctx.height_offset is not None:
    z = z + ctx.height_offset
  return (z - ctx.state[0][..., 2]).abs()


def anchor_tilt_error(ctx):
  return (ctx.reference_root[1][..., 2, 2] - ctx.state[1][..., 2, 2]).abs()


HEIGHT_FAILURE = FailureComponent(anchor_height_error, soft_width=.02, requires_height=True)
TILT_FAILURE = FailureComponent(anchor_tilt_error, soft_width=.04)


def bad_anchor_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast("MotionCommand", env.command_manager.get_term(command_name))
  return (
    torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
  )


def bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  from ..core.state import read_state
  cmd = env.command_manager.get_term(command_name)
  mdp = env.functional_mdp
  offset = mdp.termination_height_offset(cmd.time_steps, env.scene.env_origins, cmd.reference_xy_offset)
  return mdp.termination_errors(read_state(env), cmd.time_steps, height_offset=offset)[0] > threshold


def bad_anchor_ori(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
  from ..core.state import read_state
  cmd = env.command_manager.get_term(command_name)
  return env.functional_mdp.termination_errors(read_state(env), cmd.time_steps)[1] > threshold


def bad_motion_body_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast("MotionCommand", env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.norm(
    command.body_pos_relative_w[:, body_indexes]
    - command.robot_body_pos_w[:, body_indexes],
    dim=-1,
  )
  return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast("MotionCommand", env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.abs(
    command.body_pos_relative_w[:, body_indexes, -1]
    - command.robot_body_pos_w[:, body_indexes, -1]
  )
  return torch.any(error > threshold, dim=-1)


def motion_clip_end(env, command_name="motion"):
  cmd = env.command_manager.get_term(command_name)
  return cmd.time_steps >= cmd.motion.clip_end[cmd.motion_ids]


def episode_time_out(env):
  """Same limit as mjlab.time_out, tagged independently of the configured name."""
  return env.episode_length_buf >= env.max_episode_length


episode_time_out.supermimic_truncation = "time_limit"
motion_clip_end.supermimic_truncation = "motion_end"


bad_anchor_pos_z_only.supermimic_failure = HEIGHT_FAILURE
bad_anchor_ori.supermimic_failure = TILT_FAILURE
