"""Reward tolerance follows the environment's assigned terrain family."""


def tolerance_by_terrain_column(cfg):
  """mjlab 1.6 curriculum maps sub_terrains insertion order to columns.

  This identifies the configured family, not whether a particular point on a
  stair/platform happens to be level. The G1 factory names its flat family flat.
  """
  if cfg is None or cfg.terrain_type == "plane":
    return (False,)
  if cfg.terrain_type != "generator" or cfg.terrain_generator is None:
    raise ValueError("Terrain-dependent tolerance requires plane or generated terrain")
  generator = cfg.terrain_generator
  if not generator.curriculum:
    # Random generation can put different families in the same column; mjlab
    # 1.6 does not expose their per-tile family IDs on TerrainEntity.
    raise ValueError("Terrain-dependent tolerance requires curriculum=True (one family per column)")
  return tuple(name != "flat" for name in generator.sub_terrains)
