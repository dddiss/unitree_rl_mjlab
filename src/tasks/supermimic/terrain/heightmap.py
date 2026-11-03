"""Static collision heights, rasterized once before collection starts.

Cell-centre samples are stored in float32; queries select their containing cell.
There is no Gaussian smoothing, neighbour filling, or interpolation across steps.
Spatial accuracy is limited by resolution (2.5 cm by default), unlike exact rays.
Warp misses are rechecked at startup against the original float64 MuJoCo model.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
import torch
from tqdm.auto import tqdm


def mark_terrain_padding_ray_group(scene_spec: mujoco.MjSpec) -> None:
  """Keep generator border boxes visible to ground and depth sensors (group 5)."""
  terrain = scene_spec.body("terrain")
  if terrain is None:
    raise RuntimeError("Missing terrain body")
  padding = [geom for geom in terrain.geoms if int(geom.group) == 0]
  if len(padding) != 4:
    raise RuntimeError(f"Expected four generator padding boxes, found {len(padding)}")
  for geom in padding:
    geom.group = 5


def terrain_bounds(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
  """World AABB of group-5 collision terrain, including transformed heightfields."""
  ids = np.flatnonzero(model.geom_group == 5)
  if not len(ids):
    raise ValueError("Height map requires collision terrain in geom group 5")
  if np.any(model.body_dofnum[model.body_rootid[model.geom_bodyid[ids]]] != 0):
    raise ValueError("Height map requires static terrain")
  rotation = data.geom_xmat[ids].reshape(-1, 3, 3)
  aabb = model.geom_aabb[ids]
  center = data.geom_xpos[ids] + np.einsum("nij,nj->ni", rotation, aabb[:, :3])
  extent = np.einsum("nij,nj->ni", np.abs(rotation), aabb[:, 3:])
  return (center - extent).min(axis=0), (center + extent).max(axis=0)


def _native_heights(model, data, xy, ray_z):
  """Use original CPU geometry, not a float32 model converted back to double."""
  mask = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)
  direction = np.array([0., 0., -1.], dtype=np.float64)
  origin = np.array([0., 0., ray_z], dtype=np.float64)
  geom_id = np.empty(1, dtype=np.int32)
  heights = np.full(len(xy), np.nan, dtype=np.float64)
  for i, point in enumerate(xy):
    origin[:2] = point
    distance = mujoco.mj_ray(model, data, origin, direction, mask, True, -1, geom_id)
    if math.isfinite(distance) and distance >= 0:
      heights[i] = ray_z - distance
  return heights


class _WarpRasterizer:
  """Temporary single-world BVH context; never uses or mutates live sim data."""

  def __init__(self, model, data, device):
    import mujoco_warp as mjwarp
    import warp as wp

    self.device = device
    with wp.ScopedDevice(str(device)), wp.ScopedStream(
        wp.stream_from_torch(torch.cuda.current_stream(device))):
      self.model = mjwarp.put_model(model)
      self.data = mjwarp.put_data(model, data, nworld=1, nconmax=max(64, data.ncon),
                                 njmax=max(512, data.nefc))
      self.context = mjwarp.create_render_context(
        model, nworld=1, cam_active=[False] * model.ncam, enabled_geom_groups=[5],
        use_textures=False, use_shadows=False)
    self.groups = wp.types.vector(length=6, dtype=float)(0, 0, 0, 0, 0, -1)

  def __call__(self, xy, ray_z):
    import mujoco_warp as mjwarp
    import warp as wp

    points = torch.as_tensor(xy, dtype=torch.float32, device=self.device)
    origins = torch.empty(1, len(xy), 3, device=self.device)
    origins[0, :, :2] = points
    origins[..., 2] = ray_z
    directions = torch.zeros_like(origins)
    directions[..., 2] = -1
    distances = torch.empty(1, len(xy), device=self.device)
    geom_ids = torch.empty_like(distances, dtype=torch.int32)
    normals = torch.empty_like(origins)
    exclude = torch.full((len(xy),), -1, dtype=torch.int32, device=self.device)
    with wp.ScopedDevice(str(self.device)), wp.ScopedStream(
        wp.stream_from_torch(torch.cuda.current_stream(self.device))):
      mjwarp.rays(
        self.model, self.data, wp.from_torch(origins, dtype=wp.vec3),
        wp.from_torch(directions, dtype=wp.vec3), self.groups, True,
        wp.from_torch(exclude), wp.from_torch(distances), wp.from_torch(geom_ids),
        wp.from_torch(normals, dtype=wp.vec3), rc=self.context)
    heights = origins[..., 2] - distances
    heights.masked_fill_((distances < 0) | ~torch.isfinite(distances), float('nan'))
    return heights[0].cpu().numpy()


class TerrainHeightMap:
  def __init__(self, heights, xy_min, xy_max, *, cpu_rechecks=0):
    self.heights = heights
    self.xy_min = heights.new_tensor(xy_min)
    self.xy_max = heights.new_tensor(xy_max)
    self.cell_size = (self.xy_max - self.xy_min) / heights.new_tensor(heights.shape)
    self.cpu_rechecks = cpu_rechecks

  @classmethod
  @torch.no_grad()
  def from_model(cls, model, device, *, resolution=0.025, chunk_size=262144):
    if not math.isfinite(resolution) or resolution <= 0 or chunk_size < 1:
      raise ValueError("Height-map resolution and chunk_size must be positive")
    device = torch.device(device)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lower, upper = terrain_bounds(model, data)
    nx, ny = np.ceil((upper[:2] - lower[:2]) / resolution).astype(int)
    total = int(nx * ny)
    spacing = (upper[:2] - lower[:2]) / (nx, ny)
    ray_z = float(upper[2] + 1.5)
    # CPU is also supported for small models and regression tests.
    raster = _WarpRasterizer(model, data, device) if device.type == 'cuda' else None
    values = np.empty(total, dtype=np.float32)
    missed = []
    print(f"[TerrainHeightMap] {nx} x {ny} cells, resolution <= {resolution:g} m; no smoothing", flush=True)
    with tqdm(total=total, desc="Terrain height map", unit="point", dynamic_ncols=True) as progress:
      for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        indices = np.arange(start, stop)
        xy = lower[:2] + (np.stack((indices // ny, indices % ny), -1) + .5) * spacing
        heights = raster(xy, ray_z) if raster is not None else _native_heights(model, data, xy, ray_z)
        invalid = ~np.isfinite(heights) | (heights < lower[2] - 1e-5) | (heights > upper[2] + 1e-5)
        values[start:stop] = heights
        missed.append(indices[invalid])
        progress.update(stop - start)
    del raster  # Release the startup-only Warp model and render context.
    bad = np.concatenate(missed)
    with tqdm(total=len(bad), desc="CPU MuJoCo recheck", unit="point", dynamic_ncols=True,
              disable=not len(bad)) as progress:
      for start in range(0, len(bad), 1024):
        indices = bad[start:start + 1024]
        xy = lower[:2] + (np.stack((indices // ny, indices % ny), -1) + .5) * spacing
        values[indices] = _native_heights(model, data, xy, ray_z)
        progress.update(len(indices))
    unresolved = np.flatnonzero(~np.isfinite(values) | (values < lower[2] - 1e-5)
                               | (values > upper[2] + 1e-5))
    if len(unresolved):
      indices = unresolved[:8]
      xy = lower[:2] + (np.stack((indices // ny, indices % ny), -1) + .5) * spacing
      raise RuntimeError(f"Height map has {len(unresolved)} invalid cells after CPU MuJoCo recheck; "
                         f"xy_w={xy.tolist()}. Collision terrain must cover the map.")
    result = cls(torch.from_numpy(values.reshape(nx, ny)).to(device), lower[:2], upper[:2],
                 cpu_rechecks=len(bad))
    print(f"[TerrainHeightMap] ready; CPU rechecked {len(bad)} points; "
          f"storage={result.heights.numel() * result.heights.element_size() / 2**20:.1f} MiB", flush=True)
    return result

  @torch.no_grad()
  def sample(self, xy):
    """Containing-cell heights and validity; no rays, CPU transfers, or gradients."""
    valid = torch.isfinite(xy).all(-1) & (xy >= self.xy_min).all(-1) & (xy <= self.xy_max).all(-1)
    ij = ((xy - self.xy_min) / self.cell_size).nan_to_num().floor().long()
    ix = ij[..., 0].clamp(0, self.heights.shape[0] - 1)
    iy = ij[..., 1].clamp(0, self.heights.shape[1] - 1)
    height = self.heights[ix, iy]
    valid = valid & torch.isfinite(height)
    return torch.where(valid, height, float('nan')), valid

  @torch.no_grad()
  def max_in_rect(self, xy_lower, xy_upper, *, max_cells=262144):
    """Maximum over EVERY cell covered by each closed XY rectangle.

    This reset-only query leaves the map and reference point lookups unchanged.
    Rectangles extending outside the map, or containing invalid cells, are
    invalid rather than silently clipped. Bounded chunks avoid allocating a
    full [num_resets, num_geoms, footprint_grid] tensor. Only two window-size
    integers are read back for batching; all height gathering stays on device.
    """
    if xy_lower.shape != xy_upper.shape or xy_lower.shape[-1] != 2:
      raise ValueError("Rectangle bounds must have matching [..., 2] shapes")
    if max_cells < 1:
      raise ValueError("max_cells must be positive")
    shape = xy_lower.shape[:-1]
    lower, upper = xy_lower.reshape(-1, 2), xy_upper.reshape(-1, 2)
    valid = (torch.isfinite(lower).all(-1) & torch.isfinite(upper).all(-1)
             & (lower <= upper).all(-1) & (lower >= self.xy_min).all(-1)
             & (upper <= self.xy_max).all(-1))
    if not len(lower):
      return self.heights.new_empty(shape), valid.reshape(shape)
    # Clamp only indices used for safe gathers; validity still rejects any
    # out-of-map footprint. Include the cell containing the upper edge, just
    # as sample() does when a point lands exactly on a cell boundary.
    first = ((lower - self.xy_min) / self.cell_size).nan_to_num(0, 0, 0).floor().long()
    last = ((upper - self.xy_min) / self.cell_size).nan_to_num(0, 0, 0).floor().long()
    limit = first.new_tensor(self.heights.shape) - 1
    first = torch.minimum(first.clamp_min(0), limit)
    last = torch.minimum(last.clamp_min(0), limit)
    counts = (last - first + 1).clamp_min(1)
    nx, ny = counts.amax(dim=0).tolist()
    area = nx * ny
    batch_size = max(1, max_cells // min(area, max_cells))
    result = self.heights.new_full((len(lower),), -torch.inf)
    for start in range(0, len(lower), batch_size):
      stop = min(start + batch_size, len(lower))
      cell_batch = max(1, max_cells // (stop - start))
      for cell_start in range(0, area, cell_batch):
        offsets = torch.arange(cell_start, min(cell_start + cell_batch, area), device=lower.device)
        dx, dy = offsets // ny, offsets % ny
        inside = (dx[None] < counts[start:stop, :1]) & (dy[None] < counts[start:stop, 1:])
        ix = (first[start:stop, :1] + dx).clamp_max(self.heights.shape[0] - 1)
        iy = (first[start:stop, 1:] + dy).clamp_max(self.heights.shape[1] - 1)
        heights = self.heights[ix, iy]
        valid[start:stop] &= (~inside | torch.isfinite(heights)).all(-1)
        maximum = heights.masked_fill(~inside, -torch.inf).amax(-1)
        result[start:stop] = torch.maximum(result[start:stop], maximum)
    return torch.where(valid, result, float('nan')).reshape(shape), valid.reshape(shape)

  @torch.no_grad()
  def height(self, xy):
    """Checked reference lookup: never silently clamp an out-of-map target."""
    heights, valid = self.sample(xy)
    if not bool(valid.all()):
      bad = (~valid).reshape(-1).nonzero().flatten()[:8]
      raise RuntimeError("Reference XY is outside the terrain height map or invalid: "
                         f"query_shape={tuple(xy.shape[:-1])}, flat_indices={bad.tolist()}, "
                         f"reference_xy_w={xy.reshape(-1, 2)[bad].tolist()}")
    return heights
