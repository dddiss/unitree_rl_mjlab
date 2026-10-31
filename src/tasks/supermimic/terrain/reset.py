"""Reset-only placement on collision terrain; no training-time height features.

Height rule adapted from RoboParty/P-BFM envs/mjlab/reset.py (CC BY-NC 4.0).
See ../perception/LICENSE-P-BFM and THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import sys

import mujoco
import numpy as np
import torch


@dataclass(frozen=True)
class TerrainResetCfg:
  sensor_name: str = "reset_ground"
  footprint_radius: float = 0.65
  extra_clearance: float = 0.05
  randomize_xy: bool = False
  xy_max_attempts: int = 16


def terrain_vertical_bounds(model: mujoco.MjModel) -> tuple[float, float]:
  """World-space bounds of the static group-5 terrain, computed once on CPU."""
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  ids = np.flatnonzero(model.geom_group == 5)
  if not len(ids):
    raise ValueError("Reset height queries require collision terrain in geom group 5")
  rotation = data.geom_xmat[ids].reshape(-1, 3, 3)
  aabb = model.geom_aabb[ids]
  center = data.geom_xpos[ids] + np.einsum("nij,nj->ni", rotation, aabb[:, :3])
  extent_z = (np.abs(rotation[:, 2]) * aabb[:, 3:]).sum(axis=-1)
  return float((center[:, 2] - extent_z).min()), float((center[:, 2] + extent_z).max())


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
      "Reset XY is outside the collision terrain or its ground ray missed. "
      "Use a larger terrain/border or motions whose XY remains inside the map."
    )
  ground = hit_heights[:, center_index]
  footprint = offsets[:, :2].norm(dim=-1) <= cfg.footprint_radius
  highest = hit_heights.masked_fill(~(valid & footprint[None]), -torch.inf).amax(dim=-1)
  delta = (highest - ground).clamp_min(0)
  lift = torch.where(delta > 1e-4, delta + cfg.extra_clearance, 0.0)
  return ground + desired_clearance + lift


def _print_center_ray_failures(env, env_ids, root_state, sensor, data, center_index):
  """Report failed reset rays only; frame indices are zero-based."""
  distances = data.distances[env_ids, center_index]
  hit_heights = data.hit_pos_w[env_ids, center_index, 2]
  valid = (distances >= 0) & torch.isfinite(distances) & torch.isfinite(hit_heights)
  failed_rows = torch.nonzero(~valid, as_tuple=False).flatten().tolist()
  if not failed_rows:
    return

  command = env.command_manager.get_term("motion")
  motion = command.motion
  # Read the actual raycast origins rather than inferring them from root_state.
  ray_origins = sensor._cached_world_origins
  for row in failed_rows:
    env_id = int(env_ids[row])
    frame = int(command._initialization_frames[env_id])
    clip = int(motion.clip_id[frame])
    record = {
      "env_id": env_id,
      "motion_id": clip,
      "motion_name": motion.motion_names[clip],
      "motion_file": str(motion.motion_files[clip]),
      "init_frame_global_k": frame,
      "init_frame_local_k": frame - int(motion.clip_start[clip]),
      "target_frame_global_f": int(command.time_steps[env_id]),
      "reset_xy_w": root_state[row, :2].tolist(),
      "ray_origin_w": (
        ray_origins[env_id, center_index].tolist() if ray_origins is not None else None
      ),
      "ray_distance": float(distances[row]),
      "ray_hit_z_w": float(hit_heights[row]),
    }
    print("[TerrainReset] center ray failed: " + json.dumps(record, ensure_ascii=False),
          file=sys.stderr, flush=True)


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


def _probe_ground(env, robot, env_ids, root_state, sensor):
  """Sense at candidate XY, from above the entire map, without taking a step."""
  probe = root_state.clone()
  probe[:, 2] = sensor.probe_world_z
  robot.write_root_state_to_sim(probe, env_ids=env_ids)
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.sim.sense()
  # Explicit invalidation also covers older cached-data implementations.
  sensor.update(0.0)
  return sensor.data


def _checked_heights(env, env_ids, root_state, cfg, sensor, data, offsets, center_index):
  try:
    return reset_height(
      root_state[:, 2] - env.scene.env_origins[env_ids, 2], data.hit_pos_w[env_ids, :, 2],
      (data.distances[env_ids] >= 0) & torch.isfinite(data.distances[env_ids]),
      offsets, center_index, cfg,
    )
  except RuntimeError:
    try:
      _print_center_ray_failures(env, env_ids, root_state, sensor, data, center_index)
    except Exception as diagnostic_error:
      # Diagnostics must not replace the original reset failure.
      print(f"[TerrainReset] Could not collect ray diagnostics: {diagnostic_error}",
            file=sys.stderr, flush=True)
    raise


@torch.no_grad()
def place_on_terrain(env, robot, env_ids, root_state, cfg: TerrainResetCfg):
  """Training may translate XY; height, velocity and pose-noise rules stay intact.

  Candidate root XY is uniform inside the assigned tile with a footprint margin.
  Reject invalid center rays, retry only rejected rows, and retain the original
  failure/diagnostics if no valid point is found. The caller applies the same XY
  displacement to the reference, preserving the original robot/reference error.
  """
  from .pattern import reference_ray_index

  sensor = env.scene[cfg.sensor_name]
  offsets, _ = sensor.cfg.pattern.generate_rays(None, root_state.device)
  center_index = reference_ray_index(offsets)
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
      ids = env_ids[pending]
      data = _probe_ground(env, robot, ids, result[pending], sensor)
      distances = data.distances[ids, center_index]
      valid = ((distances >= 0) & torch.isfinite(distances)
               & torch.isfinite(data.hit_pos_w[ids, center_index, 2]))
      accepted = pending[valid]
      if accepted.numel():
        result[accepted, 2] = _checked_heights(
          env, env_ids[accepted], result[accepted], cfg, sensor, data, offsets, center_index,
        )
      pending = pending[~valid]
      if not pending.numel():
        break
    if pending.numel():
      # Exhausted retries: print the last actual candidates and raise as before.
      _checked_heights(env, env_ids[pending], result[pending], cfg, sensor,
                       data, offsets, center_index)
  else:
    data = _probe_ground(env, robot, env_ids, root_state, sensor)
    result[:, 2] = _checked_heights(
      env, env_ids, root_state, cfg, sensor, data, offsets, center_index,
    )
  sensor.update(0.0)
  return result
