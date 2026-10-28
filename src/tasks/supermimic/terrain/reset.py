"""Reset-only placement on collision terrain; no training-time height features.

Height rule adapted from RoboParty/P-BFM envs/mjlab/reset.py (CC BY-NC 4.0).
See ../perception/LICENSE-P-BFM and THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch


@dataclass(frozen=True)
class TerrainResetCfg:
  sensor_name: str = "reset_ground"
  footprint_radius: float = 0.65
  extra_clearance: float = 0.05


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


@torch.no_grad()
def place_on_terrain(env, robot, env_ids, root_state, cfg: TerrainResetCfg):
  """Change only root Z, after the original pose/velocity/joint noise draws.

  Joints have already been written. Probe from above the entire collision map:
  unlike P-BFM's region-selected reset XY, a reference clip can leave its tile.
  The caller writes the final root state; normal env.reset then forwards/senses
  again before observations or depth-history initialization.
  """
  from .pattern import reference_ray_index

  sensor = env.scene[cfg.sensor_name]
  desired_clearance = root_state[:, 2] - env.scene.env_origins[env_ids, 2]
  probe = root_state.clone()
  probe[:, 2] = sensor.probe_world_z
  robot.write_root_state_to_sim(probe, env_ids=env_ids)
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.sim.sense()
  # Explicit invalidation also covers older cached-data implementations.
  sensor.update(0.0)
  data = sensor.data
  offsets, _ = sensor.cfg.pattern.generate_rays(None, root_state.device)
  result = root_state.clone()
  result[:, 2] = reset_height(
    desired_clearance, data.hit_pos_w[env_ids, :, 2],
    (data.distances[env_ids] >= 0) & torch.isfinite(data.distances[env_ids]),
    offsets, reference_ray_index(offsets), cfg,
  )
  sensor.update(0.0)
  return result
