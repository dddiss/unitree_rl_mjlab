# Camera convention and G1 calibration adapted from RoboParty/P-BFM,
# CC BY-NC 4.0; see THIRD_PARTY_NOTICES.md.
"""mjlab 1.6 ray camera with fixed calibration and synchronous optical-Z depth."""
from dataclasses import dataclass
import math
import torch

from .preprocessing import preprocess_depth


@dataclass(frozen=True)
class DepthCameraCfg:
  name: str = "supermimic_depth"
  mount_body: str = "torso_link"
  mount_pos: tuple[float, ...] = (0.0487988662332928, 0.01, 0.4378029937970051)
  optical_quat_xyzw: tuple[float, ...] = (-0.6579550475696607, 0.6623183568544073,
                                         -0.25121840449386046, 0.25558171377860706)
  horizontal_fov: float = 89.51
  vertical_fov: float = 58.29
  max_distance: float = 2.5
  include_geom_groups: tuple[int, ...] = (0, 2, 5)
  excluded_housing_meshes: tuple[str, ...] = ("head_link",)

  def optical_rays(self, device="cpu"):
    y, x = torch.meshgrid(torch.arange(36, device=device) + .5,
                          torch.arange(64, device=device) + .5, indexing="ij")
    fx = 32 / math.tan(math.radians(self.horizontal_fov) / 2)
    fy = 18 / math.tan(math.radians(self.vertical_fov) / 2)
    rays = torch.stack(((x - 32) / fx, (y - 18) / fy, torch.ones_like(x)), -1)
    return torch.nn.functional.normalize(rays, dim=-1)

  def sensor_cfg(self):
    from mjlab.sensor import ObjRef
    from .sensor import DepthRayCastSensorCfg
    return DepthRayCastSensorCfg(
      name=self.name, frame=ObjRef(type="body", name=self.mount_body, entity="robot"),
      pattern=DepthPatternCfg(self), ray_alignment="base", exclude_parent_body=False,
      include_geom_groups=self.include_geom_groups,
      excluded_housing_meshes=self.excluded_housing_meshes,
      # A 2.5 m optical-Z plane is farther than 2.5 m along oblique rays.
      max_distance=self.max_distance / float(self.optical_rays()[..., 2].min()) + 1e-4,
    )


def exclude_camera_housing(scene_spec, mesh_names, group=4):
  """Hide only the camera enclosure from rays; collision flags stay untouched."""
  found = set()
  for geom in scene_spec.geoms:
    mesh = str(geom.meshname)
    for name in mesh_names:
      if mesh == name or mesh.endswith("/" + name):
        geom.group = group
        found.add(name)
  missing = set(mesh_names) - found
  if missing:
    raise ValueError(f"Camera housing meshes not found in scene: {sorted(missing)}")


@dataclass
class DepthPatternCfg:
  camera: DepthCameraCfg

  def generate_rays(self, mj_model, device):
    del mj_model
    rays = self.camera.optical_rays(device).reshape(-1, 3)
    q = torch.tensor(self.camera.optical_quat_xyzw, device=device)
    q = q / q.norm()
    xyz = q[:3].expand_as(rays)
    cross = 2 * torch.cross(xyz, rays, dim=-1)
    directions = rays + q[3] * cross + torch.cross(xyz, cross, dim=-1)
    offsets = torch.tensor(self.camera.mount_pos, device=device).expand_as(rays).clone()
    return offsets, directions


class DepthCamera:
  def __init__(self, cfg, scene, device):
    self.cfg, self.sensor = cfg, scene.sensors[cfg.name]
    self.ray_z = cfg.optical_rays(device)[..., 2]

  @torch.no_grad()
  def read(self):
    """Caller must run sim.forward()/sense() first, including before reset."""
    distances = self.sensor.data.distances.reshape(-1, 36, 64)
    depth = distances * self.ray_z
    depth = torch.where((distances >= 0) & torch.isfinite(depth), depth, self.cfg.max_distance)
    return preprocess_depth(depth, self.cfg.max_distance)
