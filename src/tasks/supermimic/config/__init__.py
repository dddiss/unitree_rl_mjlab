"""Configuration factory for the independent SuperMimic entry points."""

def make_env(motion_file, *, num_envs=1, device="cuda:0", evaluation=False,
             gamma=0.99, seed=0, render_mode=None):
  from .g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
  from ..env import SuperMimicEnv
  cfg = unitree_g1_flat_tracking_env_cfg(play=evaluation)
  cfg.seed = seed
  cfg.scene.num_envs = num_envs
  cmd = cfg.commands["motion"]
  if isinstance(motion_file, (list, tuple)):
    cmd.motion_files = tuple(str(p) for p in motion_file)
  else:
    cmd.motion_file = str(motion_file)
  cmd.gamma = gamma
  return SuperMimicEnv(cfg, device=device, render_mode=render_mode)
