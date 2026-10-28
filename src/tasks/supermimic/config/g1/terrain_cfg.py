"""P-BFM G1 terrain defaults. Edit G1_TERRAIN_MODE to select one family/plane.

Adapted from P-BFM terrain_ufo_v0.yaml and terrains/presets.py (CC BY-NC 4.0).
See THIRD_PARTY_NOTICES.md and ../../perception/LICENSE-P-BFM.
"""
from __future__ import annotations

import math

from mjlab.utils.spec_config import GeomCfg
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg

from ...terrain.primitives import (
  NeutralHfPerlinNoiseTerrainCfg,
  NeutralHfPyramidSlopedTerrainCfg,
  ScatteredBoxPlatformsTerrainCfg,
  TerrainBoxFlatCfg,
  TerrainSeparatedPyramidStairsCfg,
  spawn_patch_sampling,
)

G1_TERRAIN_MODE = "mixed"


def unitree_g1_terrain_cfg(mode: str | None = None) -> TerrainEntityCfg:
  """Build the connected 10×6 map; generation difficulty is fixed by row.

  This does not enable an online terrain curriculum or change reset XY.
  Ground friction retains SuperMimic's existing training value.
  """
  mode = G1_TERRAIN_MODE if mode is None else mode
  geoms = (GeomCfg(geom_names_expr=(".*",), friction=(0.3,)),)
  if mode == "plane":
    return TerrainEntityCfg(terrain_type="plane", geoms=geoms)
  size = (8.0, 8.0)
  terrains = {
    "flat": TerrainBoxFlatCfg(proportion=0.15),
    "slope": NeutralHfPyramidSlopedTerrainCfg(
      proportion=0.15,
      slope_range=(math.tan(math.radians(5)), math.tan(math.radians(8))),
      platform_width=1.5, inverted=True, border_width=0.5,
      horizontal_scale=0.1, vertical_scale=0.005,
    ),
    "stairs_up": TerrainSeparatedPyramidStairsCfg(
      proportion=0.20, direction="up", step_height_range=(0.10, 0.20),
      step_width=0.30, platform_width=0.80, num_steps=10, border_width=0.5,
    ),
    "stairs_down": TerrainSeparatedPyramidStairsCfg(
      proportion=0.20, direction="down", step_height_range=(0.10, 0.20),
      step_width=0.30, platform_width=0.80, num_steps=10, border_width=0.5,
    ),
    "rough": NeutralHfPerlinNoiseTerrainCfg(
      proportion=0.15, height_range=(0.03, 0.05), octaves=2,
      persistence=0.25, lacunarity=2.0, scale=20.0,
      horizontal_scale=0.1, resolution=0.1, border_width=0.5,
      flat_patch_sampling=spawn_patch_sampling(
        patch_radius=0.45, patch_center_range=1.25, patch_size=size,
      ),
    ),
    "platforms": ScatteredBoxPlatformsTerrainCfg(
      proportion=0.15, easy_height_range=(0.05, 0.10),
      hard_height_range=(0.10, 0.18), center_width=1.2, border_width=0.5,
      gap_width=0.25, min_platform_width=0.60, raised_fraction=1.0,
      row_count_range=(5, 7), column_count_range=(4, 7),
      spawn_patch_margin=0.25, num_spawn_patches=12,
    ),
  }
  if mode != "mixed":
    if mode not in terrains:
      raise ValueError(f"Unknown terrain {mode!r}; choose plane, mixed, or {tuple(terrains)}")
    terrains = {mode: terrains[mode]}
    terrains[mode].proportion = 1.0
  generator = TerrainGeneratorCfg(
    seed=0, size=size, border_width=17.0, num_rows=10, num_cols=len(terrains),
    curriculum=True, difficulty_range=(0.0, 1.0), color_scheme="none",
    sub_terrains=terrains, add_lights=False,
  )
  return TerrainEntityCfg(
    terrain_type="generator", terrain_generator=generator,
    env_spacing=None, max_init_terrain_level=generator.num_rows - 1, geoms=geoms,
  )
