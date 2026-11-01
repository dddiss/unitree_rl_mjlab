"""P-BFM reset clearance, using the startup collision height map.

Height rule adapted from RoboParty/P-BFM envs/mjlab/reset.py (CC BY-NC 4.0).
See ../perception/LICENSE-P-BFM and THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import sys

import torch

from .pattern import ResetFootprintCfg, reference_point_index


@dataclass(frozen=True)
class TerrainResetCfg:
  heightmap_resolution: float = 0.025
  pattern: ResetFootprintCfg = field(default_factory=ResetFootprintCfg)
  footprint_radius: float = 0.65
  extra_clearance: float = 0.05
  randomize_xy: bool = False
  xy_max_attempts: int = 16


def reset_height(
  desired_clearance: torch.Tensor,
  hit_heights: torch.Tensor,
  valid: torch.Tensor,
  offsets: torch.Tensor,
  center_index: int,
  cfg: TerrainResetCfg,
) -> torch.Tensor:
  """P-BFM: local ground + motion clearance + optional footprint lift."""
  valid = valid & torch.isfinite(hit_heights)
  if not bool(valid[:, center_index].all()):
    raise RuntimeError(
      "Reset XY is outside the collision terrain or its height-map cell is invalid. "
      "Use a larger terrain/border or motions whose XY remains inside the map."
    )
  ground = hit_heights[:, center_index]
  footprint = offsets[:, :2].norm(dim=-1) <= cfg.footprint_radius
  highest = hit_heights.masked_fill(~(valid & footprint[None]), -torch.inf).amax(dim=-1)
  delta = (highest - ground).clamp_min(0)
  lift = torch.where(delta > 1e-4, delta + cfg.extra_clearance, 0.0)
  return ground + desired_clearance + lift


def terrain_tile_xy_bounds(terrain, env_ids, margin):
  """Bounds of the assigned grid tile, not its possibly off-center spawn patch."""
  tile_size = terrain.cfg.terrain_generator.size
  if min(tile_size) <= 2 * margin:
    raise ValueError("Terrain tiles must be wider than twice the reset footprint radius")
  size = torch.tensor(tile_size, device=env_ids.device, dtype=torch.float32)
  half = size / 2 - margin
  grid = torch.tensor(terrain.terrain_origins.shape[:2], device=env_ids.device)
  cells = torch.stack((terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]), -1)
  centers = (cells + 0.5 - grid / 2) * size
  return centers - half, centers + half


def _sample_ground(env, root_state, offsets):
  # Match the former yaw-aligned footprint, including reset orientation noise.
  from ..mdp.rotations import quat_to_mat
  rotation = quat_to_mat(root_state[:, 3:7])
  forward = rotation[:, :2, 0]
  norm = forward.norm(dim=-1, keepdim=True)
  # Same near-vertical fallback as mjlab 1.6 RayCastSensor._extract_yaw_rotation.
  left = rotation[:, :2, 1]
  left = left / left.norm(dim=-1, keepdim=True).clamp_min(1e-6)
  forward = torch.where(norm < .1, torch.stack((left[:, 1], -left[:, 0]), -1),
                        forward / norm.clamp_min(1e-6))
  c, s = forward[:, :1], forward[:, 1:2]
  x, y = offsets[:, 0][None], offsets[:, 1][None]
  xy = root_state[:, None, :2] + torch.stack((c*x - s*y, s*x + c*y), -1)
  return env.terrain_height_map.sample(xy)


def _checked_heights(env, env_ids, root_state, cfg, heights, valid, offsets, center):
  try:
    return reset_height(root_state[:, 2] - env.scene.env_origins[env_ids, 2],
                        heights, valid, offsets, center, cfg)
  except RuntimeError:
    try:
      cmd = env.command_manager.get_term("motion")
      for row in (~valid[:, center]).nonzero().flatten().tolist():
        env_id = int(env_ids[row])
        frame = int(cmd._initialization_frames[env_id])
        clip = int(cmd.motion.clip_id[frame])
        record = dict(env_id=env_id, motion_name=cmd.motion.motion_names[clip],
                      motion_file=str(cmd.motion.motion_files[clip]),
                      init_frame_global_k=frame,
                      init_frame_local_k=frame-int(cmd.motion.clip_start[clip]),
                      target_frame_global_f=int(cmd.time_steps[env_id]),
                      reset_xy_w=root_state[row, :2].tolist())
        print("[TerrainReset] invalid height-map query: " + json.dumps(record, ensure_ascii=False),
              file=sys.stderr, flush=True)
    except Exception as error:
      print(f"[TerrainReset] Could not collect height-map diagnostics: {error}", file=sys.stderr, flush=True)
    raise


@torch.no_grad()
def place_on_terrain(env, robot, env_ids, root_state, cfg: TerrainResetCfg):
  """Keep XY sampling and clearance rules; queries do not touch simulator state."""
  del robot
  offsets = cfg.pattern.sample_offsets(root_state.device)
  center = reference_point_index(offsets)
  result = root_state.clone()
  if cfg.randomize_xy:
    if cfg.xy_max_attempts < 1:
      raise ValueError("xy_max_attempts must be positive")
    low, high = terrain_tile_xy_bounds(env.scene.terrain, env_ids, cfg.footprint_radius)
    pending = torch.arange(len(env_ids), device=env_ids.device)
    for _ in range(cfg.xy_max_attempts):
      result[pending, :2] = low[pending] + torch.rand(
        len(pending), 2, device=root_state.device, dtype=root_state.dtype,
      ) * (high[pending] - low[pending])
      heights, valid = _sample_ground(env, result[pending], offsets)
      accepted = valid[:, center]
      ids = pending[accepted]
      if ids.numel():
        result[ids, 2] = _checked_heights(env, env_ids[ids], result[ids], cfg,
                                        heights[accepted], valid[accepted], offsets, center)
      rejected_heights, rejected_valid = heights[~accepted], valid[~accepted]
      pending = pending[~accepted]
      if not pending.numel():
        break
    if pending.numel():
      _checked_heights(env, env_ids[pending], result[pending], cfg,
                       rejected_heights, rejected_valid, offsets, center)
  else:
    heights, valid = _sample_ground(env, result, offsets)
    result[:, 2] = _checked_heights(env, env_ids, result, cfg, heights, valid, offsets, center)
  return result
