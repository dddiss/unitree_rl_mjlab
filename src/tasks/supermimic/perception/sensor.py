# Housing exclusion adapted from RoboParty/P-BFM, CC BY-NC 4.0;
# see THIRD_PARTY_NOTICES.md.
"""mjlab sensor binding, imported only when constructing a simulation."""
from dataclasses import dataclass

from mjlab.sensor.raycast_sensor import RayCastSensor, RayCastSensorCfg

from .camera import exclude_camera_housing


class DepthRayCastSensor(RayCastSensor):
  def edit_spec(self, scene_spec, entities):
    del entities
    exclude_camera_housing(scene_spec, self.cfg.excluded_housing_meshes)


@dataclass
class DepthRayCastSensorCfg(RayCastSensorCfg):
  excluded_housing_meshes: tuple[str, ...] = ("head_link",)

  def build(self):
    if self.include_geom_groups is None or 4 in self.include_geom_groups:
      raise ValueError("Depth rays must exclude camera-housing geom group 4")
    return DepthRayCastSensor(self)
