"""Register G1 tasks with the independent SuperMimic registry."""
from ...tools.registry import register_task

register_task(
  task_id="Unitree-G1-SuperMimic",
  env_cfg_factory=".config.g1.env_cfgs:unitree_g1_env_cfg",
  runner_cfg_factory=".config.g1.rl_cfg:unitree_g1_runner_cfg",
)
register_task(
  task_id="Unitree-G1-SuperMimic-Terrain",
  env_cfg_factory=".config.g1.env_cfgs:unitree_g1_terrain_env_cfg",
  runner_cfg_factory=".config.g1.rl_cfg:unitree_g1_terrain_runner_cfg",
)
