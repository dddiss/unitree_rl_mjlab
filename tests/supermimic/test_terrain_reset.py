"""Height placement, local-reset isolation, and compiled collision queries."""
import ast
import json
import math
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import pytest
import torch

from supermimic.terrain.pattern import RobotCentricGridPatternCfg, reference_ray_index
from supermimic.terrain.reset import (
  TerrainResetCfg, place_on_terrain, reset_height, terrain_vertical_bounds,
)
from test_env_lifecycle import ROOT, methods


def test_pbfm_grid_and_height_rule():
  pattern = RobotCentricGridPatternCfg()
  offsets, directions = pattern.generate_rays(None, 'cpu')
  center = reference_ray_index(offsets)
  assert pattern.shape == (21, 13) and len(offsets) == 273
  assert directions[:, 2].eq(-1).all()
  heights = torch.tensor([0., 1.2, -1.5, 0.]).unsqueeze(1).repeat(1, 273)
  valid = torch.ones_like(heights, dtype=torch.bool)
  clearance = torch.tensor([0.8, 0.82, 0.79, 0.8])
  # A higher stair inside the footprint lifts the root; a farther obstacle does not.
  near = ((offsets[:, 0] - .2).abs() + offsets[:, 1].abs()).argmin()
  far = offsets[:, 0].argmax()
  heights[3, near] = .3
  heights[:, far] = 3.
  actual = reset_height(clearance, heights, valid, offsets, center, TerrainResetCfg())
  torch.testing.assert_close(actual, torch.tensor([.8, 2.02, -.71, 1.15]))
  # Misses around the edge are excluded, never interpreted as ground at probe Z.
  valid[3, near] = False
  actual = reset_height(clearance, heights, valid, offsets, center, TerrainResetCfg())
  assert actual[3] == clearance[3]
  valid[0, center] = False
  with pytest.raises(RuntimeError, match='Reset XY'):
    reset_height(clearance, heights, valid, offsets, center, TerrainResetCfg())


def test_height_matches_actual_mujoco_surface_and_bounds():
  model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <geom type="box" size="3 3 .1" pos="0 0 -.1" group="5"/>
    <geom type="box" size=".4 .4 .15" pos=".5 0 .15" group="5"/>
    <geom type="box" size=".1 .1 .1" pos="0 0 20" group="2"/>
  </worldbody></mujoco>''')
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  lower, upper = terrain_vertical_bounds(model)
  assert lower == pytest.approx(-.2) and upper == pytest.approx(.3)
  pattern = RobotCentricGridPatternCfg()
  offsets, _ = pattern.generate_rays(None, 'cpu')
  ground = []
  mask = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)
  for xy in offsets[:, :2].numpy():
    pos = np.array([*xy, upper + 1.5], dtype=float)
    distance = mujoco.mj_ray(model, data, pos, np.array([0., 0., -1.]), mask,
                             True, -1, np.zeros(1, dtype=np.int32))
    ground.append(pos[2] - distance)
  ground = torch.tensor([ground], dtype=torch.float32)
  result = reset_height(torch.tensor([.8]), ground, torch.ones_like(ground, dtype=torch.bool),
                        offsets, reference_ray_index(offsets), TerrainResetCfg())
  # Adjacent step raises the placement to top(.3) + .8 + margin(.05).
  assert result.item() == pytest.approx(1.15)


def test_padding_only_repairs_misses_on_actual_border_and_does_not_touch_robot():
  # Run the actual sensor helpers without importing the optional Warp runtime.
  path = ROOT / 'src/tasks/supermimic/terrain/sensor.py'
  helpers = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)]
  ns = {'mujoco': mujoco, 'torch': torch}
  exec(compile(ast.Module(helpers, type_ignores=[]), str(path), 'exec'), ns)
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name='terrain')
  for x, y, z in ((-3, 0, 0), (3, 0, 0), (0, -3, 0), (0, 3, 0)):
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(1, 1, .1), pos=(x, y, z-.1), group=0)
  robot = spec.worldbody.add_body(name='robot')
  robot.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(.1, 0, 0), group=0)
  regions = ns['terrain_padding_regions'](spec)
  assert ns['mark_terrain_padding_ray_group'](spec) == 4
  assert all(g.group == 5 for g in body.geoms)
  assert robot.geoms[0].group == 0
  origins = torch.tensor([[[-3., 0., 2.], [-3., 0., 2.], [10., 0., 2.]]])
  distance = torch.tensor([[-1., 1.7, -1.]])
  hit = torch.ones_like(origins)*8
  normal = torch.zeros_like(origins)
  ns['repair_padding_ray_misses'](distance, hit, normal, origins, regions, max_distance=5.)
  torch.testing.assert_close(distance, torch.tensor([[2., 1.7, -1.]]))
  torch.testing.assert_close(hit[0, 0], torch.tensor([-3., 0., 0.]))
  assert hit[0, 1:].eq(8).all()
  torch.testing.assert_close(normal[0, 0], torch.tensor([0., 0., 1.]))


def test_heightfield_bounds_include_elevation_and_static_body_transform():
  spec = mujoco.MjSpec()
  spec.add_hfield(name='height', size=(1., 1., .7, .1), nrow=2, ncol=2,
                  userdata=[0., .5, .8, 1.])
  body = spec.worldbody.add_body(name='terrain', pos=(3., 2., -2.))
  body.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname='height', group=5)
  lower, upper = terrain_vertical_bounds(spec.compile())
  assert lower == pytest.approx(-2.1) and upper == pytest.approx(-1.3)


def test_probe_changes_only_selected_root_height_without_advancing_history(capsys):
  pattern = RobotCentricGridPatternCfg()
  offsets, _ = pattern.generate_rays(None, 'cpu')
  n = len(offsets)
  all_states = torch.randn(3, 13)
  before = all_states.clone()
  ids = torch.tensor([0, 2])
  root = all_states[ids].clone()
  origins = torch.tensor([[10., 20., -1.], [10., 20., 0.], [10., 20., 2.]])
  root[:, 2] = origins[ids, 2] + torch.tensor([.81, .79])
  sensor = NS(cfg=NS(pattern=pattern), probe_world_z=5., update=lambda dt: None)
  events = []
  class Scene(dict):
    env_origins = origins
    def write_data_to_sim(self): events.append('write')
  scene = Scene(reset_ground=sensor)
  def write(state, env_ids):
    events.append('root');all_states[env_ids] = state
  def sense():
    events.append('sense')
    hit = torch.zeros(3, n, 3)
    hit[:, :, 2] = torch.tensor([.2, .5, 1.2])[:, None]
    sensor.data = NS(hit_pos_w=hit, distances=5. - hit[:, :, 2])
  env = NS(scene=scene, sim=NS(forward=lambda: events.append('forward'), sense=sense))
  robot = NS(write_root_state_to_sim=write)
  rng = torch.random.get_rng_state()
  placed = place_on_terrain(env, robot, ids, root, TerrainResetCfg())
  assert events == ['root', 'write', 'forward', 'sense']
  torch.testing.assert_close(placed[:, 2], torch.tensor([1.01, 1.99]))
  torch.testing.assert_close(placed[:, :2], root[:, :2])
  torch.testing.assert_close(placed[:, 3:], root[:, 3:])
  torch.testing.assert_close(all_states[1], before[1])
  assert torch.equal(rng, torch.random.get_rng_state())
  assert capsys.readouterr().err == ''


@pytest.fixture
def failed_reset_rays():
  pattern = RobotCentricGridPatternCfg()
  offsets, _ = pattern.generate_rays(None, 'cpu')
  center = reference_ray_index(offsets)
  ids = torch.tensor([3, 1, 4])  # Subset row indices differ from environment IDs.
  root = torch.zeros(3, 13)
  root[:, :3] = torch.tensor([[12., -7., .8], [1., 2., .9], [-4., 8., 1.]])
  hit = torch.zeros(5, len(offsets), 3)
  distances = torch.full((5, len(offsets)), 5.)
  distances[[0, 3, 4], center] = -1.  # Environment 0 is not being reset.
  distances[1, 0] = -1.  # A noncentral miss does not fail environment 1.
  ray_origins = torch.zeros_like(hit)
  ray_origins[3, center] = torch.tensor([12.125, -6.75, 5.25])
  ray_origins[4, center] = torch.tensor([-4., 8., 5.])
  updates, writes = [], []
  sensor = NS(cfg=NS(pattern=pattern), probe_world_z=5.,
              update=updates.append, _cached_world_origins=ray_origins,
              data=NS(hit_pos_w=hit, distances=distances))
  class Scene(dict):
    env_origins = torch.zeros(5, 3)
    def write_data_to_sim(self): pass
  motion = NS(clip_id=torch.cat((torch.zeros(50), torch.ones(10))).long(),
              clip_start=torch.tensor([0, 50]),
              motion_names=('dance', 'walk'),
              motion_files=('/motions/dance.npz', '/motions/walk.npz'))
  command = NS(motion=motion, _initialization_frames=torch.tensor([0, 7, 3, 53, 59]),
               time_steps=torch.tensor([1, 8, 4, 54, 59]))
  env = NS(scene=Scene(reset_ground=sensor),
           sim=NS(forward=lambda: None, sense=lambda: None),
           command_manager=NS(get_term=lambda _: command))
  robot = NS(write_root_state_to_sim=lambda state, env_ids: writes.append(state.clone()))
  return NS(env=env, robot=robot, ids=ids, root=root, sensor=sensor, center=center,
            command=command, updates=updates, writes=writes)


@pytest.mark.parametrize('distance,hit_z', [(-1., 0.), (float('nan'), 0.),
                                          (float('inf'), 0.), (5., float('nan'))])
def test_failed_center_rays_print_exact_motion_frames_and_sensor_data(
  failed_reset_rays, capsys, distance, hit_z,
):
  case = failed_reset_rays
  case.sensor.data.distances[3, case.center] = distance
  case.sensor.data.hit_pos_w[3, case.center, 2] = hit_z
  root_before = case.root.clone()
  frames_before = case.command.time_steps.clone()
  rng = torch.random.get_rng_state()
  with pytest.raises(RuntimeError, match='Reset XY is outside the collision terrain'):
    place_on_terrain(case.env, case.robot, case.ids, case.root, TerrainResetCfg())
  output = capsys.readouterr()
  assert output.out == ''
  prefix = '[TerrainReset] center ray failed: '
  lines = output.err.splitlines()
  assert len(lines) == 2 and all(line.startswith(prefix) for line in lines)
  records = [json.loads(line[len(prefix):]) for line in lines]
  assert [row['env_id'] for row in records] == [3, 4]
  first, last = records
  assert first['motion_id'] == 1 and first['motion_name'] == 'walk'
  assert first['motion_file'] == '/motions/walk.npz'
  assert first['init_frame_global_k'] == 53 and first['init_frame_local_k'] == 3
  assert first['target_frame_global_f'] == 54
  assert last['init_frame_global_k'] == 59 and last['init_frame_local_k'] == 9
  assert last['target_frame_global_f'] == 59  # Last-frame playback clamps f.
  assert first['reset_xy_w'] == [12., -7.]
  assert first['ray_origin_w'] == [12.125, -6.75, 5.25]
  assert math.isnan(first['ray_distance']) if math.isnan(distance) else first['ray_distance'] == distance
  assert math.isnan(first['ray_hit_z_w']) if math.isnan(hit_z) else first['ray_hit_z_w'] == hit_z
  assert last['reset_xy_w'] == [-4., 8.] and last['ray_distance'] == -1.
  assert case.updates == [0.] and len(case.writes) == 1  # No retry or fallback reset.
  torch.testing.assert_close(case.root, root_before)
  torch.testing.assert_close(case.command.time_steps, frames_before)
  assert torch.equal(rng, torch.random.get_rng_state())


def test_ray_diagnostic_error_preserves_original_reset_exception(failed_reset_rays, capsys):
  case = failed_reset_rays
  del case.env.command_manager
  with pytest.raises(RuntimeError, match='Reset XY is outside the collision terrain'):
    place_on_terrain(case.env, case.robot, case.ids, case.root, TerrainResetCfg())
  assert 'Could not collect ray diagnostics' in capsys.readouterr().err


@pytest.mark.parametrize('terrain', [False, True])
def test_command_writes_clipped_joints_before_probe_and_final_state_after(terrain):
  events = []
  def place(env, robot, ids, state, cfg):
    events.append('probe')
    result = state.clone();result[:, 2] += .4
    return result
  ns = methods(ROOT / 'src/tasks/supermimic/mdp/commands.py', 'MotionCommand',
               {'_write_reference_state_to_sim'}, {'torch': torch, 'place_on_terrain': place})
  cmd = NS(cfg=NS(terrain_reset=TerrainResetCfg() if terrain else None), _env=NS())
  limits = torch.tensor([[[-1., 1.]]]).expand(3, 29, 2)
  def joints(q, v, env_ids):
    events.append('joints');assert q.eq(1).all()
  def root(state, env_ids): events.append('root');cmd.written = state
  cmd.robot = NS(data=NS(soft_joint_pos_limits=limits), write_joint_state_to_sim=joints,
                 write_root_state_to_sim=root, reset=lambda env_ids: events.append('reset'))
  ns['_write_reference_state_to_sim'](
    cmd, torch.tensor([1]), torch.tensor([[3., 4., .8]]), torch.tensor([[1., 0., 0., 0.]]),
    torch.ones(1, 3), torch.ones(1, 3)*2, torch.ones(1, 29)*3, torch.zeros(1, 29),
  )
  assert events == (['joints', 'probe', 'root', 'reset'] if terrain else ['joints', 'root', 'reset'])
  assert cmd.written[0, 2] == pytest.approx(1.2 if terrain else .8)
  torch.testing.assert_close(cmd.written[0, 7:], torch.tensor([1., 1., 1., 2., 2., 2.]))
