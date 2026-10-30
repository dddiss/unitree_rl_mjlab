"""Configuration factory for the independent SuperMimic entry points."""

def make_env(motion_file, *, num_envs=1, device="cuda:0", evaluation=False,
             gamma=0.99, seed=0, render_mode=None, depth_enabled=False,
             terrain_type=None, collect_foot_rewards=None):
  from .g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
  from ..env import SuperMimicEnv
  from ..perception.camera import DepthCameraCfg
  if collect_foot_rewards is None:
    collect_foot_rewards = not evaluation
  camera = DepthCameraCfg() if depth_enabled else None
  cfg = unitree_g1_flat_tracking_env_cfg(
    play=evaluation, depth_camera=camera, terrain_type=terrain_type,
    collect_foot_rewards=collect_foot_rewards,
  )
  cfg.seed = seed
  cfg.scene.num_envs = num_envs
  cmd = cfg.commands["motion"]
  if isinstance(motion_file, (list, tuple)):
    cmd.motion_files = tuple(str(p) for p in motion_file)
  else:
    cmd.motion_file = str(motion_file)
  cmd.gamma = gamma
  return SuperMimicEnv(cfg, device=device, render_mode=render_mode, depth_camera=camera,
                       collect_foot_rewards=collect_foot_rewards)
