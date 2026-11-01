# Adapted from RoboParty/P-BFM (2aa985ac50162e446532d1e7231f81902c221e49).
# CC BY-NC 4.0; see ../perception/LICENSE-P-BFM and THIRD_PARTY_NOTICES.md.
"""Ground-only raycasts, including P-BFM's generator-padding handling."""
from __future__ import annotations
from dataclasses import dataclass
import mujoco
import torch
from mjlab.entity import Entity
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor, TerrainHeightSensorCfg
from .reset import terrain_vertical_bounds

def terrain_padding_regions(
    scene_spec: mujoco.MjSpec,
    *,
    expected_count: int = 4,
) -> tuple[tuple[float, float, float, float, float], ...]:
    """Return the XY rectangles and top heights of generator border boxes."""
    terrain_body = scene_spec.body("terrain")
    if terrain_body is None:
        raise RuntimeError("terrain body is missing while configuring height rays")
    padding_geoms = [geom for geom in terrain_body.geoms if int(geom.group) == 0]
    if len(padding_geoms) != expected_count:
        raise RuntimeError(
            "expected exactly four group-0 global terrain padding geoms, "
            f"found {len(padding_geoms)}"
        )
    return tuple(
        (
            float(geom.pos[0] - geom.size[0]),
            float(geom.pos[0] + geom.size[0]),
            float(geom.pos[1] - geom.size[1]),
            float(geom.pos[1] + geom.size[1]),
            float(geom.pos[2] + geom.size[2]),
        )
        for geom in padding_geoms
    )


def mark_terrain_padding_ray_group(
    scene_spec: mujoco.MjSpec,
    *,
    ray_group: int = 5,
    expected_count: int = 4,
) -> int:
    """Move only generator-owned global border boxes into the terrain ray group."""
    terrain_body = scene_spec.body("terrain")
    if terrain_body is None:
        raise RuntimeError("terrain body is missing while configuring height rays")
    padding_geoms = [geom for geom in terrain_body.geoms if int(geom.group) == 0]
    if len(padding_geoms) != expected_count:
        raise RuntimeError(
            "expected exactly four group-0 global terrain padding geoms, "
            f"found {len(padding_geoms)}"
        )
    for geom in padding_geoms:
        geom.group = ray_group
    return len(padding_geoms)


def repair_padding_ray_misses(
    distances: torch.Tensor,
    hit_positions: torch.Tensor,
    normals: torch.Tensor,
    origins: torch.Tensor,
    padding_regions: tuple[tuple[float, float, float, float, float], ...],
    *,
    max_distance: float,
) -> None:
    """Repair only ray misses whose XY lies on known flat generator padding."""
    misses = distances < 0.0
    for x_min, x_max, y_min, y_max, ground_z in padding_regions:
        in_region = (
            (origins[..., 0] >= x_min)
            & (origins[..., 0] <= x_max)
            & (origins[..., 1] >= y_min)
            & (origins[..., 1] <= y_max)
        )
        clearance = (origins[..., 2] - ground_z).clamp(min=0.0)
        repair = misses & in_region & (clearance <= max_distance)
        distances[repair] = clearance[repair]
        hit_positions[repair] = origins[repair]
        hit_positions[..., 2][repair] = origins[..., 2][repair] - clearance[repair]
        normals[..., 0][repair] = 0.0
        normals[..., 1][repair] = 0.0
        normals[..., 2][repair] = 1.0


@dataclass
class PbfmTerrainHeightSensorCfg(TerrainHeightSensorCfg):
    """Terrain sensor that makes MJLab generator padding visible to its rays."""

    padding_ray_group: int = 5
    probe_clearance: float = 1.5

    def build(self) -> PbfmTerrainHeightSensor:
        return PbfmTerrainHeightSensor(self)


class PbfmTerrainHeightSensor(TerrainHeightSensor):
    cfg: PbfmTerrainHeightSensorCfg

    def __init__(self, cfg: PbfmTerrainHeightSensorCfg) -> None:
        super().__init__(cfg)
        self._padding_regions: tuple[tuple[float, float, float, float, float], ...] = ()
        self._height_query_buffers = {}

    def initialize(self, mj_model, model, data, device: str) -> None:
        lower, upper = terrain_vertical_bounds(mj_model)
        self.probe_world_z = upper + self.cfg.probe_clearance
        # Only the probe origin/range changes from P-BFM: retain its 1.5 m
        # clearance above the highest geometry, even across tile boundaries.
        self.cfg.max_distance = self.probe_world_z - lower + 1.0
        super().initialize(mj_model, model, data, device)

    def edit_spec(
        self,
        scene_spec: mujoco.MjSpec,
        entities: dict[str, Entity],
    ) -> None:
        terrain = entities.get("terrain")
        generator_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        # MJLab creates generator padding directly in geom group 0.  The
        # legacy G1 terrains move all physical terrain into a dedicated group
        # and therefore need the padding moved and miss-repaired as well.
        # RP1 terrain intentionally stays in source group 0, so its padding is
        # already visible and must not be rediscovered by scanning every
        # group-0 terrain geom.
        needs_padding_group_remap = int(self.cfg.padding_ray_group) != 0
        if (
            generator_cfg is not None
            and float(generator_cfg.border_width) > 0.0
            and needs_padding_group_remap
        ):
            self._padding_regions = terrain_padding_regions(scene_spec)
            mark_terrain_padding_ray_group(
                scene_spec,
                ray_group=self.cfg.padding_ray_group,
            )
        super().edit_spec(scene_spec, entities)

    def postprocess_rays(self) -> None:
        super().postprocess_rays()
        if not self._padding_regions:
            return
        assert self._cached_world_origins is not None
        assert self._distances is not None
        assert self._hit_pos_w is not None
        assert self._normals_w is not None

        repair_padding_ray_misses(
            self._distances,
            self._hit_pos_w,
            self._normals_w,
            self._cached_world_origins,
            self._padding_regions,
            max_distance=self.cfg.max_distance,
        )

    @torch.no_grad()
    def query_height_at_xy(self, xy: torch.Tensor) -> torch.Tensor:
        """Exact collision height at arbitrary reference world XY, without reset.

        The generated terrain is static and identical across simulation worlds.
        Pack arbitrary replay/reference batches into the existing nworld ray
        context; these slots are unrelated to the episode's environment ID.
        Dedicated buffers leave robot state and normal sensor readings intact.
        """
        shape = xy.shape[:-1]
        points = xy.reshape(-1, 2)
        count = len(points)
        if not count:
            return torch.empty(shape, dtype=torch.float32, device=xy.device)
        num_worlds = self._data.nworld
        rays_per_world = (count + num_worlds - 1) // num_worlds
        buffers = self._height_query_buffers.get(rays_per_world)
        if buffers is None:
            origins = torch.empty(num_worlds, rays_per_world, 3, device=self._device)
            origins[..., 2] = self.probe_world_z
            directions = torch.zeros_like(origins)
            directions[..., 2] = -1
            distances = torch.empty(num_worlds, rays_per_world, device=self._device)
            geom_ids = torch.empty_like(distances, dtype=torch.int32)
            normals = torch.empty_like(origins)
            exclude = torch.full((rays_per_world,), -1, dtype=torch.int32, device=self._device)
            buffers = origins, directions, distances, geom_ids, normals, exclude
            self._height_query_buffers[rays_per_world] = buffers
        origins, directions, distances, geom_ids, normals, exclude = buffers
        origins.view(-1, 3)[:count, :2].copy_(points)
        origins.view(-1, 3)[count:, :2].copy_(points[0])
        self._cast_height_rays(*buffers)
        distances.masked_fill_(~torch.isfinite(distances) | (distances > self.cfg.max_distance), -1)
        hit_positions = origins + directions * distances.clamp_min(0)[..., None]
        repair_padding_ray_misses(distances, hit_positions, normals, origins,
                                  self._padding_regions, max_distance=self.cfg.max_distance)
        heights = hit_positions[..., 2].reshape(-1)[:count]
        valid = (distances.reshape(-1)[:count] >= 0) & torch.isfinite(heights)
        if not bool(valid.all()):
            bad = (~valid).nonzero().flatten()[:8]
            raise RuntimeError(
                "Reference ground ray missed collision terrain: "
                f"query_shape={tuple(shape)}, flat_indices={bad.tolist()}, "
                f"reference_xy_w={points[bad].tolist()}, ray_origin_z={self.probe_world_z}, "
                f"distances={distances.reshape(-1)[bad].tolist()}"
            )
        return heights.reshape(shape)

    def _cast_height_rays(self, origins, directions, distances, geom_ids, normals, exclude):
        """Same MuJoCo Warp rays API/context as RayCastSensor.raycast_kernel."""
        from contextlib import nullcontext
        import warp as wp
        from mujoco_warp import rays

        assert self._ctx is not None
        stream = (wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self._device)))
                  if self._wp_device.is_cuda else nullcontext())
        with wp.ScopedDevice(self._wp_device), stream:
            rays(
                m=self._model.struct, d=self._data.struct,
                pnt=wp.from_torch(origins, dtype=wp.vec3),
                vec=wp.from_torch(directions, dtype=wp.vec3),
                geomgroup=self._geomgroup, flg_static=True,
                bodyexclude=wp.from_torch(exclude), dist=wp.from_torch(distances),
                geomid=wp.from_torch(geom_ids), normal=wp.from_torch(normals, dtype=wp.vec3),
                rc=self._ctx.render_context,
            )
