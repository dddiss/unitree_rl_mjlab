"""Full-body collision-AABB placement on the unsmoothed terrain height map.

Local-ground placement retains the P-BFM adaptation (CC BY-NC 4.0);
see ../perception/LICENSE-P-BFM and THIRD_PARTY_NOTICES.md.
The per-geom clearance check is implemented here, inspired by DeepMimic's
ResolveCharGroundIntersect, and queries all covered map cells, not five points.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import sys

import mujoco
import numpy as np
import torch

from ..tools.kinematics import G1Kinematics
from ..tools.rotations import quat_to_mat


@dataclass(frozen=True)
class TerrainResetCfg:
  heightmap_resolution: float = 0.025
  extra_clearance: float = 0.01
  xy_margin: float = 0.65
  randomize_xy: bool = False
  xy_max_attempts: int = 16

  def __post_init__(self):
    if not math.isfinite(self.extra_clearance) or self.extra_clearance < 0:
      raise ValueError("extra_clearance must be finite and nonnegative")
    if not math.isfinite(self.xy_margin) or self.xy_margin < 0:
      raise ValueError("xy_margin must be finite and nonnegative")
    if self.xy_max_attempts < 1:
      raise ValueError("xy_max_attempts must be positive")


class RobotCollisionBounds:
  """Cache compiled geometry and FK indexing once; no simulator state reads.

  Geoms belong to this robot only. Disabled/visual geoms are excluded, while
  contype=0, conaffinity!=0 collision geoms remain included. Sphere and capsule
  AABBs are analytic (tight even when rotated); other bounded geoms use their
  transformed local MuJoCo AABB conservatively.
  """

  def __init__(self, model, joint_ids, geom_ids, root_body_id, device="cpu"):
    geom_ids = np.asarray(geom_ids, dtype=int)
    active = (model.geom_contype[geom_ids] != 0) | (model.geom_conaffinity[geom_ids] != 0)
    geom_ids = geom_ids[active]
    if not len(geom_ids):
      raise ValueError("Terrain reset requires robot collision geoms")
    types = model.geom_type[geom_ids]
    if np.isin(types, [mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD]).any():
      raise ValueError("Robot reset collision geoms must be bounded rigid shapes")
    body_ids, body_index = np.unique(model.geom_bodyid[geom_ids], return_inverse=True)
    self.kin = G1Kinematics(model, joint_ids, body_ids, root_body_id, device)
    self.geom_ids = geom_ids
    self.geom_names = tuple(model.geom(int(g)).name for g in geom_ids)
    tensor = lambda x: torch.as_tensor(np.array(x, copy=True), dtype=torch.float32, device=device)
    self.body_index = torch.as_tensor(body_index, dtype=torch.long, device=device)
    self.pos = tensor(model.geom_pos[geom_ids])
    self.rot = quat_to_mat(tensor(model.geom_quat[geom_ids]))
    self.local_center = tensor(model.geom_aabb[geom_ids, :3])
    self.local_extent = tensor(model.geom_aabb[geom_ids, 3:])
    self.radius = tensor(model.geom_size[geom_ids, 0])
    self.half_length = tensor(model.geom_size[geom_ids, 1])
    self.sphere = torch.as_tensor(types == mujoco.mjtGeom.mjGEOM_SPHERE, device=device)
    self.capsule = torch.as_tensor(types == mujoco.mjtGeom.mjGEOM_CAPSULE, device=device)

  def aabbs(self, root_pos, root_quat, joint_pos):
    """World lower/upper corners [..., num_collision_geoms, 3]."""
    body_pos, body_rot = self.kin.forward_pose(root_pos, quat_to_mat(root_quat), joint_pos)
    body_pos, body_rot = body_pos[..., self.body_index, :], body_rot[..., self.body_index, :, :]
    geom_pos = body_pos + (body_rot @ self.pos[..., None]).squeeze(-1)
    geom_rot = body_rot @ self.rot
    center = geom_pos + (geom_rot @ self.local_center[..., None]).squeeze(-1)
    extent = (geom_rot.abs() @ self.local_extent[..., None]).squeeze(-1)
    sphere_or_capsule = self.sphere | self.capsule
    center = torch.where(sphere_or_capsule[:, None], geom_pos, center)
    capsule_extent = self.radius[:, None] + geom_rot[..., :, 2].abs() * self.half_length[:, None]
    extent = torch.where(self.capsule[:, None], capsule_extent, extent)
    extent = torch.where(self.sphere[:, None], self.radius[:, None], extent)
    return center - extent, center + extent


def terrain_tile_xy_bounds(terrain, env_ids, margin):
  """Bounds of the assigned grid tile, not its possibly off-center spawn patch."""
  tile_size = terrain.cfg.terrain_generator.size
  if min(tile_size) <= 2 * margin:
    raise ValueError("Terrain tiles must be wider than twice the reset XY margin")
  size = torch.tensor(tile_size, device=env_ids.device, dtype=torch.float32)
  half = size / 2 - margin
  grid = torch.tensor(terrain.terrain_origins.shape[:2], device=env_ids.device)
  cells = torch.stack((terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]), -1)
  centers = (cells + 0.5 - grid / 2) * size
  return centers - half, centers + half


def _placement_heights(env, env_ids, root_state, lower, upper, cfg):
  """Root-local Z + centre ground, then full-body penetration correction."""
  ground, center_valid = env.terrain_height_map.sample(root_state[:, :2])
  xy = root_state[:, None, :2]
  highest, footprint_valid = env.terrain_height_map.max_in_rect(
    xy + lower[..., :2], xy + upper[..., :2])
  base_z = ground + root_state[:, 2] - env.scene.env_origins[env_ids, 2]
  penetration = (highest - (base_z[:, None] + lower[..., 2])).amax(-1).clamp_min(0)
  valid = center_valid & footprint_valid.all(-1)
  return base_z + penetration + cfg.extra_clearance, valid, footprint_valid


def _raise_invalid_placement(env, env_ids, root_state, valid, footprint_valid, geometry):
  try:
    cmd = env.command_manager.get_term("motion")
    for row in (~valid).nonzero().flatten().tolist():
      env_id = int(env_ids[row])
      frame = int(cmd._initialization_frames[env_id])
      clip = int(cmd.motion.clip_id[frame])
      bad_geoms = (~footprint_valid[row]).nonzero().flatten().tolist()
      record = dict(env_id=env_id, motion_name=cmd.motion.motion_names[clip],
                    motion_file=str(cmd.motion.motion_files[clip]),
                    init_frame_global_k=frame,
                    init_frame_local_k=frame-int(cmd.motion.clip_start[clip]),
                    target_frame_global_f=int(cmd.time_steps[env_id]),
                    reset_xy_w=root_state[row, :2].tolist(),
                    invalid_collision_geoms=[geometry.geom_names[i] for i in bad_geoms])
      print("[TerrainReset] invalid height-map query: " + json.dumps(record, ensure_ascii=False),
            file=sys.stderr, flush=True)
  except Exception as error:
    print(f"[TerrainReset] Could not collect height-map diagnostics: {error}", file=sys.stderr, flush=True)
  raise RuntimeError(
    "Reset XY is outside the collision terrain or a robot collision footprint has invalid height-map cells. "
    "Use a larger terrain/border or motions whose bodies remain inside the map."
  )


@torch.no_grad()
def place_on_terrain(env, env_ids, root_state, joint_pos, geometry, cfg: TerrainResetCfg):
  """Apply the final perturbed/clipped pose, with no physics step or ray query.

  FK is evaluated once relative to root translation. Candidate XY placements
  translate those same bounds. Only invalid candidates are resampled; a valid
  candidate is never lowered, even for a jump or another airborne reference.
  """
  result = root_state.clone()
  if not len(env_ids):
    return result
  lower, upper = geometry.aabbs(torch.zeros_like(root_state[:, :3]), root_state[:, 3:7], joint_pos)
  if cfg.randomize_xy:
    low, high = terrain_tile_xy_bounds(env.scene.terrain, env_ids, cfg.xy_margin)
    pending = torch.arange(len(env_ids), device=env_ids.device)
    for _ in range(cfg.xy_max_attempts):
      result[pending, :2] = low[pending] + torch.rand(
        len(pending), 2, device=root_state.device, dtype=root_state.dtype,
      ) * (high[pending] - low[pending])
      height, valid, footprint_valid = _placement_heights(
        env, env_ids[pending], result[pending], lower[pending], upper[pending], cfg)
      result[pending[valid], 2] = height[valid]
      rejected_footprints = footprint_valid[~valid]
      pending = pending[~valid]
      if not pending.numel():
        break
    if pending.numel():
      _raise_invalid_placement(env, env_ids[pending], result[pending],
                               torch.zeros_like(pending, dtype=torch.bool), rejected_footprints, geometry)
  else:
    height, valid, footprint_valid = _placement_heights(env, env_ids, result, lower, upper, cfg)
    if not bool(valid.all()):
      _raise_invalid_placement(env, env_ids, result, valid, footprint_valid, geometry)
    result[:, 2] = height
  return result
