# Adapted from RoboParty/P-BFM (2aa985ac50162e446532d1e7231f81902c221e49).
# CC BY-NC 4.0; see ../perception/LICENSE-P-BFM and THIRD_PARTY_NOTICES.md.
"""Foot-attached sole points queried vertically against collision terrain."""
from dataclasses import dataclass

import torch
import warp as wp
from mjlab.sensor.raycast_sensor import RayCastSensor, RayCastSensorCfg

from ..mdp.foot_rewards import fixed_sole_weights


@dataclass
class WorldUpTerrainHeightSensorCfg(RayCastSensorCfg):
  ray_start_height: float = 20.0
  local_center: tuple[float, float, float] = (0.025, 0.0, -0.045)

  def build(self):
    return WorldUpTerrainHeightSensor(self)


class WorldUpTerrainHeightSensor(RayCastSensor):
  cfg: WorldUpTerrainHeightSensorCfg

  def initialize(self, mj_model, model, data, device):
    super().initialize(mj_model, model, data, device)
    assert self._local_offsets is not None
    self._local_offsets = self._local_offsets.clone()
    self._local_offsets += self._local_offsets.new_tensor(self.cfg.local_center)
    self.point_weights = fixed_sole_weights(self._local_offsets[:, :2])
    self._sample_points_w: torch.Tensor | None = None

  @property
  def sample_points_w(self):
    if self._sample_points_w is None:
      raise RuntimeError("Foot terrain samples require sim.sense() after sim.forward()")
    return self._sample_points_w

  def prepare_rays(self):
    super().prepare_rays()
    assert self._cached_world_origins is not None
    assert self._cached_world_rays is not None
    assert self._ray_pnt is not None and self._ray_vec is not None
    # The full foot rotation moves sole points; rays always point world-down.
    self._sample_points_w = self._cached_world_origins.clone()
    self._cached_world_origins[..., 2] += self.cfg.ray_start_height
    self._cached_world_rays.zero_()
    self._cached_world_rays[..., 2] = -1.0
    wp.to_torch(self._ray_pnt).view_as(self._cached_world_origins).copy_(self._cached_world_origins)
    wp.to_torch(self._ray_vec).view_as(self._cached_world_rays).copy_(self._cached_world_rays)

  def postprocess_rays(self):
    super().postprocess_rays()
    # mjlab 1.6 rebinds hit tensors here. Discard any pre-sense cached data.
    self._invalidate_cache()
