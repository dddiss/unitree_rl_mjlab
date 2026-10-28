"""CPU model compilation with installed mjlab, plus optional GPU lifecycle."""
import sys

import mujoco
import numpy as np
import pytest
import torch

from supermimic.terrain.reset import terrain_vertical_bounds
from test_env_lifecycle import ROOT


def terrain_config(mode=None):
  pytest.importorskip('mjlab.terrains')
  from supermimic.config.g1.terrain_cfg import unitree_g1_terrain_cfg
  return unitree_g1_terrain_cfg(mode)


@pytest.mark.parametrize('family', ['flat', 'slope', 'stairs_up', 'stairs_down', 'rough', 'platforms'])
def test_each_pbfm_family_compiles_and_raycast_sees_origin(family):
  from dataclasses import replace
  config = terrain_config(family)
  cfg = replace(config.terrain_generator.sub_terrains[family], size=(8., 8.))
  for difficulty in (0., 1.):
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name='terrain')
    output = cfg.function(difficulty, spec, np.random.default_rng(0))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert np.all(model.geom_group == 5)
    lower, upper = terrain_vertical_bounds(model)
    origin = output.origin.copy()
    origin[2] = upper + 1.5
    distance = mujoco.mj_ray(model, data, origin, np.array([0., 0., -1.]),
                             np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8),
                             True, -1, np.zeros(1, dtype=np.int32))
    assert distance >= 0 and lower <= origin[2] - distance <= upper + 1e-6
    if family != 'rough':
      assert origin[2] - distance == pytest.approx(output.origin[2], abs=1e-5)


def test_default_map_padding_and_environment_config():
  cfg = terrain_config()
  from mjlab.terrains import TerrainGenerator
  from supermimic.terrain.sensor import mark_terrain_padding_ray_group, terrain_padding_regions
  from src.tasks.supermimic.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
  gen = cfg.terrain_generator
  assert gen.size == (8., 8.) and gen.num_rows == 10 and gen.num_cols == 6
  assert gen.border_width == 17. and gen.curriculum and gen.seed == 0
  assert [s.proportion for s in gen.sub_terrains.values()] == [.15, .15, .2, .2, .15, .15]
  spec = mujoco.MjSpec()
  generator = TerrainGenerator(gen)
  generator.compile(spec)
  assert generator.terrain_origins.shape == (10, 6, 3)
  assert len(terrain_padding_regions(spec)) == 4
  assert mark_terrain_padding_ray_group(spec) == 4
  model = spec.compile()
  assert model.nhfield > 0 and np.all(model.geom_group == 5)
  # Creating all configs must preserve reset noise, reward/termination functions,
  # and physics options while adding only the ground-reset sensor/config.
  for play in (False, True):
    old = unitree_g1_flat_tracking_env_cfg(play=play, terrain_type='plane')
    new = unitree_g1_flat_tracking_env_cfg(play=play)
    assert old.rewards == new.rewards and old.terminations == new.terminations
    assert old.sim == new.sim and old.events == new.events
    a, b = old.commands['motion'], new.commands['motion']
    assert a.pose_range == b.pose_range and a.velocity_range == b.velocity_range
    assert a.joint_position_range == b.joint_position_range
    assert a.terrain_reset is None and b.terrain_reset is not None
    assert len(new.scene.sensors) == len(old.scene.sensors) + 1
    assert new.scene.terrain.geoms[0].friction == ((1.,) if play else (.3,))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires Linux NVIDIA GPU and mjlab 1.6.0')
def test_real_terrain_partial_reset_preserves_other_envs(mdp):
  sys.path.insert(0, str(ROOT))
  from src.tasks.supermimic.config import make_env
  env = make_env(mdp.motion.motion_files, num_envs=4, evaluation=True, depth_enabled=True)
  try:
    env.reset()
    cmd = env.command_manager.get_term('motion')
    ids = torch.tensor([0, 2], device=env.device)
    keep = torch.tensor([1, 3], device=env.device)
    before = {k: v.clone() for k, v in {
      'frames': cmd.time_steps, 'steps': env.episode_length_buf,
      'qpos': env.sim.data.qpos, 'qvel': env.sim.data.qvel,
      'actions': env.action_manager.action,
    }.items()}
    cmd.request_frames(torch.zeros(2, dtype=torch.long, device=env.device), ids)
    env.reset(env_ids=ids)
    for name, value in {
      'frames': cmd.time_steps, 'steps': env.episode_length_buf,
      'qpos': env.sim.data.qpos, 'qvel': env.sim.data.qvel,
      'actions': env.action_manager.action,
    }.items():
      torch.testing.assert_close(value[keep], before[name][keep])
    assert cmd.time_steps[ids].eq(1).all()
    assert torch.isfinite(env.sim.data.qpos).all()
    assert env.depth_image.shape == (4, 18, 32)
    for _ in range(16):
      env.step(torch.zeros(4, 29, device=env.device))
      tr = env.last_transition
      assert torch.isfinite(tr['s1']).all()
      done = (tr['terminated'] | tr['truncated']).nonzero().flatten()
      if len(done):
        env.reset(env_ids=done)
  finally:
    env.close()
