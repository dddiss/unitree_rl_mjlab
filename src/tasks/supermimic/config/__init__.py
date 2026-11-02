"""Configuration factory for the independent SuperMimic entry points."""

def make_env(motion_file, *, num_envs=1, device="cuda:0", evaluation=False,
             gamma=0.99, seed=0, render_mode=None,
             terrain_type=None, collect_foot_rewards=None,
             randomize_terrain_on_reset=False, viewer_geom_group=None,
             task_id="Unitree-G1-SuperMimic-Terrain"):
  from ..tools.registry import load_env_cfg
  from ..env import SuperMimicEnv
  if collect_foot_rewards is None:
    collect_foot_rewards = not evaluation
  cfg = load_env_cfg(task_id,
    play=evaluation, terrain_type=terrain_type,
    collect_foot_rewards=collect_foot_rewards,
  )
  cfg.seed = seed
  cfg.scene.num_envs = num_envs
  if viewer_geom_group is not None:
    # Set before environment construction: its offscreen renderer reads this once.
    cfg.viewer.geom_group = viewer_geom_group
  cmd = cfg.commands["motion"]
  if isinstance(motion_file, (list, tuple)):
    cmd.motion_files = tuple(str(p) for p in motion_file)
  else:
    cmd.motion_file = str(motion_file)
  cmd.gamma = gamma
  return SuperMimicEnv(cfg, device=device, render_mode=render_mode,
                       collect_foot_rewards=collect_foot_rewards,
                       randomize_terrain_on_reset=randomize_terrain_on_reset)
