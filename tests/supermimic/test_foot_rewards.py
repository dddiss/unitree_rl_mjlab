"""P-BFM foot-cost formulas, manager adapters and sensor configuration."""
import ast
from types import SimpleNamespace as NS

import pytest
import torch
import mujoco
import numpy as np

from supermimic.mdp.foot_rewards import (
  clipped_height_clearance, feet_at_plane_penalty, feet_stumble_penalty, fixed_sole_weights,
)
from test_env_lifecycle import ROOT


def test_stumble_threshold_vertical_support_peak_and_cap():
  forces = torch.zeros(5, 4, 4, 3)
  forces[0, :, :, 2] = 200.  # Pure support is not stumbling.
  forces[1, 2, 1] = torch.tensor([60., 80., -30.])  # (100 - 30 - 20)/100 = .5
  forces[2, 0, 3] = torch.tensor([20., 0., 0.])  # Threshold.
  forces[3, 3, 2] = torch.tensor([300., 0., 0.])  # Clamped to 1.
  forces[4, 1, 0] = torch.tensor([80., 0., 30.])
  forces[4, 2, 1] = torch.tensor([100., 0., 30.])  # Peak, not sum.
  torch.testing.assert_close(feet_stumble_penalty(forces), torch.tensor([0., .5, 0., 1., .5]))
  assert feet_stumble_penalty(torch.zeros(2, 0, 4, 3)).eq(0).all()


def sole_grid():
  x, y = torch.meshgrid(torch.linspace(-.115, .115, 24), torch.linspace(-.02, .02, 5), indexing='ij')
  return torch.stack((x.flatten(), y.flatten()), -1)


def test_support_cost_full_sole_overhang_swing_and_weighted_center():
  xy = sole_grid()
  weights = fixed_sole_weights(xy)
  assert weights.min().item() == pytest.approx(.1)
  assert weights.max().item() == pytest.approx(1.)
  clearance = torch.zeros(5, 2, len(xy))
  contact = torch.ones(5, 2, dtype=torch.bool)
  clearance[0] = .03  # At the tolerance, explicitly zero.
  clearance[1, 0, xy[:, 0] > 0] = .2  # Half of one foot over a step edge.
  clearance[2] = .5; contact[2] = False  # Swing feet must remain unpenalized.
  clearance[3] = -.01  # No additional penetration penalty in this P-BFM term.
  clearance[4] = .2
  cost = feet_at_plane_penalty(clearance, contact, weights)
  torch.testing.assert_close(cost, torch.tensor([0., .5, 0., 0., 2.]), atol=1e-6, rtol=1e-5)
  # The same unsupported area is penalized more near the center of the sole.
  center, edge = weights.argmax(), weights.argmin()
  two = torch.zeros(2, 2, len(xy)); two[0, 0, center] = .2; two[1, 0, edge] = .2
  cost = feet_at_plane_penalty(two, torch.ones(2, 2, dtype=torch.bool), weights)
  assert cost[0] > cost[1]


def test_clearance_handles_invalid_rays_and_world_height_offsets():
  sole = torch.tensor([.02, .02, .02, 8., -8.])
  ground = torch.tensor([0., 0., float('nan'), 0., 0.])
  distances = torch.tensor([20., -1., 20., 20., 20.])
  expected = torch.tensor([.02, 5., 5., 5., -5.])
  torch.testing.assert_close(clipped_height_clearance(sole, ground, distances), expected)
  torch.testing.assert_close(clipped_height_clearance(sole+3., ground+3., distances), expected)


def manager_adapters():
  path = ROOT / 'src/tasks/supermimic/mdp/rewards.py'
  body = [n for n in ast.parse(path.read_text()).body
          if isinstance(n, ast.FunctionDef) and n.name in {'feet_stumble', 'feet_at_plane'}]
  ns = dict(torch=torch, clipped_height_clearance=clipped_height_clearance,
            feet_at_plane_penalty=feet_at_plane_penalty, feet_stumble_penalty=feet_stumble_penalty)
  exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), 'exec'), ns)
  return ns


def test_manager_adapters_reorder_contact_bodies_and_handle_ray_misses():
  ns = manager_adapters()
  forces = torch.zeros(2, 2, 4, 3)
  forces[0, 1, 2, 2] = -10.  # Sensor order: right, left. Only left is loaded.
  forces[1, 0, 0, 2] = 20.   # Only right is loaded in row 1.
  contact = NS(data=NS(force_history=forces),
               primary_names=['robot/right_ankle_roll_link', 'robot/left_ankle_roll_link'])
  points = torch.zeros(2, 4, 3)
  points[..., 2] = .02
  hits = torch.zeros_like(points)
  distances = torch.ones(2, 4)*20
  distances[:, :2] = -1.  # Left foot misses; right foot supported.
  scanner = NS(num_rays_per_frame=2, point_weights=torch.ones(2), sample_points_w=points,
               data=NS(hit_pos_w=hits, distances=distances))
  env = NS(num_envs=2, scene={'foot_height_scanners': scanner, 'foot_support_contacts': contact,
                              'foot_collision_contacts': contact})
  torch.testing.assert_close(ns['feet_at_plane'](env), torch.tensor([1., 0.]))
  assert ns['feet_stumble'](env).eq(0).all()
  contact.data.force_history = None
  for name in ('feet_stumble', 'feet_at_plane'):
    with pytest.raises(ValueError, match='force history'):
      ns[name](env)


def sensor_configurator():
  path = ROOT / 'src/tasks/supermimic/config/g1/env_cfgs.py'
  func = next(n for n in ast.parse(path.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == 'configure_foot_reward_sensors')
  func.body = [n for n in func.body if not isinstance(n, ast.ImportFrom)]
  ns = {name: NS for name in ('ContactSensorCfg', 'ContactMatch', 'GridPatternCfg',
                              'ObjRef', 'WorldUpTerrainHeightSensorCfg')}
  exec(compile(ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[])), str(path), 'exec'), ns)
  return ns[func.name]


@pytest.mark.parametrize('stumble', [False, True])
@pytest.mark.parametrize('support', [False, True])
@pytest.mark.parametrize('terrain', ['plane', 'generator'])
def test_only_enabled_costs_allocate_correct_sensors(stumble, support, terrain):
  cfg = NS(decimation=4, scene=NS(sensors=(), terrain=NS(terrain_type=terrain)), rewards={
    'feet_stumble': NS(weight=-2. if stumble else 0., params={'sensor_name': 'stumble'}),
    'feet_at_plane': NS(weight=-1. if support else 0., params={
      'sensor_name': 'sole', 'contact_sensor_name': 'support'})})
  sensor_configurator()(cfg)
  sensors = {s.name: s for s in cfg.scene.sensors}
  assert set(sensors) == ({'stumble'} if stumble else set()) | ({'sole', 'support'} if support else set())
  if stumble:
    s = sensors['stumble']
    assert s.global_frame and s.reduce == 'maxforce' and s.num_slots == 2 and s.history_length == 4
    assert s.fields == ('force', 'normal', 'tangent')
  if support:
    s = sensors['sole']
    assert sensors['support'].reduce == 'netforce' and sensors['support'].history_length == 4
    assert [f.name for f in s.frame] == ['left_ankle_roll_link', 'right_ankle_roll_link']
    assert s.include_geom_groups == ((5,) if terrain == 'generator' else (0,))
    assert s.ray_alignment == 'base' and s.local_center == (.025, 0., -.045)
    assert s.pattern.size == (.23, .04) and s.pattern.resolution == .01


def test_world_down_scanner_preserves_rotated_sole_points_and_refreshes_cache():
  """Execute the actual scanner override with tensor-backed Warp buffers."""
  path = ROOT / 'src/tasks/supermimic/terrain/foot_sensor.py'
  cls = next(n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.ClassDef) and n.name == 'WorldUpTerrainHeightSensor')
  class RayBase:
    def prepare_rays(self):
      self._cached_world_origins = self.points.clone()
      self._cached_world_rays = torch.ones_like(self.points)
    def postprocess_rays(self):
      self.processed = True
    def _invalidate_cache(self):
      self._cache_valid = False
  ns = dict(RayCastSensor=RayBase, WorldUpTerrainHeightSensorCfg=NS, torch=torch,
            wp=NS(to_torch=lambda x: x), fixed_sole_weights=fixed_sole_weights)
  exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), 'exec'), ns)
  sensor = ns['WorldUpTerrainHeightSensor']()
  sensor.cfg = NS(ray_start_height=20.)
  # Sole points after a tilted foot's body-to-world transform.
  angle = .3
  rotation = torch.tensor([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                           [-np.sin(angle), 0., np.cos(angle)]], dtype=torch.float32)
  local = torch.tensor([[-.09, -.02, -.045], [.14, .02, -.045]])
  sensor.points = (local @ rotation.T + torch.tensor([0., 0., .3]))[None]
  sensor._ray_pnt = torch.zeros_like(sensor.points)
  sensor._ray_vec = torch.zeros_like(sensor.points)
  sensor.prepare_rays()
  torch.testing.assert_close(sensor.sample_points_w, sensor.points)
  torch.testing.assert_close(sensor._ray_pnt[..., :2], sensor.points[..., :2])
  torch.testing.assert_close(sensor._ray_pnt[..., 2], sensor.points[..., 2]+20.)
  torch.testing.assert_close(sensor._ray_vec, torch.tensor([[[0., 0., -1.], [0., 0., -1.]]]))
  # Query a real MuJoCo step: the front point hits the upper box, the rear the floor.
  model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <geom type="plane" size="2 2 .1" group="5"/>
    <geom type="box" size=".5 .5 .1" pos=".5 0 .1" group="5"/>
  </worldbody></mujoco>''')
  data = mujoco.MjData(model); mujoco.mj_forward(model, data)
  heights = []
  for origin, direction in zip(sensor._ray_pnt[0].numpy(), sensor._ray_vec[0].numpy()):
    distance = mujoco.mj_ray(model, data, origin.astype(float), direction.astype(float),
                             np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8), True, -1,
                             np.zeros(1, dtype=np.int32))
    heights.append(float(origin[2])-distance)
  np.testing.assert_allclose(heights, [0., .2], atol=1e-6)
  sensor._cache_valid = True
  sensor.postprocess_rays()
  assert sensor.processed and not sensor._cache_valid


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires Linux NVIDIA GPU and mjlab 1.6')
def test_real_foot_sensors_reward_logging_and_local_reset(mdp):
  import sys
  sys.path.insert(0, str(ROOT))
  from src.tasks.supermimic.config.g1.env_cfgs import (
    configure_foot_reward_sensors, unitree_g1_flat_tracking_env_cfg,
  )
  from src.tasks.supermimic.env import SuperMimicEnv
  cfg = unitree_g1_flat_tracking_env_cfg(play=True, terrain_type='plane', collect_foot_rewards=True)
  cfg.scene.num_envs = 4
  cfg.sim.nconmax, cfg.sim.njmax = 128, 512
  cfg.commands['motion'].motion_files = tuple(mdp.motion.motion_files)
  cfg.terminations.pop('anchor_pos')
  cfg.rewards['feet_stumble'].weight = -2.
  cfg.rewards['feet_at_plane'].weight = -1.
  env = SuperMimicEnv(cfg, device='cuda:0', collect_foot_rewards=True)
  try:
    env.reset()
    assert env.functional_mdp.cfg.term_height is None
    # Explicitly require learned counterparts; analytic-only validation rejects them.
    with pytest.raises(ValueError, match='WM reward heads'):
      env.functional_mdp.validate_imagination_rewards()
    env.functional_mdp.validate_imagination_rewards(env.foot_reward_names)
    for _ in range(8):
      env.step(torch.zeros(4, 29, device=env.device))
      terms = env.last_step_metrics['reward']
      torch.testing.assert_close(sum(terms.values()), env.last_transition['r'])
      for name in ('feet_stumble', 'feet_at_plane'):
        assert torch.isfinite(terms[name]).all() and (terms[name] <= 0).all()
      for i, name in enumerate(env.foot_reward_names):
        torch.testing.assert_close(terms[name], env.last_transition['foot_costs'][:, i]
                                    * cfg.rewards[name].weight * env.step_dt)
      env.reset(env_ids=(env.last_transition['terminated'] | env.last_transition['truncated']).nonzero().flatten())
    sensor = env.scene['foot_support_contacts']
    before = sensor.data.force_history.clone()
    env.reset(env_ids=torch.tensor([0, 2], device=env.device))
    torch.testing.assert_close(sensor.data.force_history[[1, 3]], before[[1, 3]])
  finally:
    env.close()
