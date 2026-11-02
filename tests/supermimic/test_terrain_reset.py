"""Height placement, local-reset isolation, and compiled collision queries."""
import ast
import json
import math
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import pytest
import torch

from supermimic.terrain.reset import (
  ResetFootprintCfg, reference_point_index, TerrainResetCfg, place_on_terrain, reset_height,
)
from test_env_lifecycle import ROOT, methods
from supermimic.terrain.heightmap import TerrainHeightMap, terrain_bounds


def test_pbfm_grid_and_height_rule():
  pattern = ResetFootprintCfg()
  offsets = pattern.sample_offsets('cpu')
  center = reference_point_index(offsets)
  assert pattern.shape == (21, 13) and len(offsets) == 273
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
  lo, hi = terrain_bounds(model, data)
  lower, upper = lo[2], hi[2]
  assert lower == pytest.approx(-.2) and upper == pytest.approx(.3)
  pattern = ResetFootprintCfg()
  offsets = pattern.sample_offsets('cpu')
  ground = []
  mask = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)
  for xy in offsets[:, :2].numpy():
    pos = np.array([*xy, upper + 1.5], dtype=float)
    distance = mujoco.mj_ray(model, data, pos, np.array([0., 0., -1.]), mask,
                             True, -1, np.zeros(1, dtype=np.int32))
    ground.append(pos[2] - distance)
  ground = torch.tensor([ground], dtype=torch.float32)
  result = reset_height(torch.tensor([.8]), ground, torch.ones_like(ground, dtype=torch.bool),
                        offsets, reference_point_index(offsets), TerrainResetCfg())
  # Adjacent step raises the placement to top(.3) + .8 + margin(.05).
  assert result.item() == pytest.approx(1.15)


def test_heightfield_bounds_include_elevation_and_static_body_transform():
  spec = mujoco.MjSpec()
  spec.add_hfield(name='height', size=(1., 1., .7, .1), nrow=2, ncol=2,
                  userdata=[0., .5, .8, 1.])
  body = spec.worldbody.add_body(name='terrain', pos=(3., 2., -2.))
  body.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname='height', group=5)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  lo, hi = terrain_bounds(model, data)
  lower, upper = lo[2], hi[2]
  assert lower == pytest.approx(-2.1) and upper == pytest.approx(-1.3)


def test_lookup_changes_only_selected_height_without_writing_simulator(capsys):
  ids = torch.tensor([0, 2])
  origins = torch.tensor([[10., 20., -1.], [10., 20., 0.], [10., 20., 2.]])
  root = torch.randn(2, 13)
  root[:, :2] = torch.tensor([[-3., 0.], [3., 0.]])
  root[:, 2] = origins[ids, 2] + torch.tensor([.81, .79])
  root[:, 3:7] = torch.tensor([1., 0., 0., 0.])
  heightmap = TerrainHeightMap(torch.tensor([[.2], [1.2]]), (-10., -10.), (10., 10.))
  env = NS(scene=NS(env_origins=origins), terrain_height_map=heightmap)
  rng = torch.random.get_rng_state()
  placed = place_on_terrain(env, object(), ids, root, TerrainResetCfg())
  torch.testing.assert_close(placed[:, 2], torch.tensor([1.01, 1.99]))
  torch.testing.assert_close(placed[:, :2], root[:, :2])
  torch.testing.assert_close(placed[:, 3:], root[:, 3:])
  assert torch.equal(rng, torch.random.get_rng_state())
  assert capsys.readouterr().err == ''


def test_out_of_map_reset_diagnostic_contains_environment_and_reference(capsys):
  ids = torch.tensor([3, 1, 4])
  root = torch.zeros(3, 13)
  root[:, :3] = torch.tensor([[12., -7., .8], [1., 2., .9], [-4., 8., 1.]])
  root[:, 3] = 1
  motion = NS(clip_id=torch.cat((torch.zeros(50), torch.ones(10))).long(),
              clip_start=torch.tensor([0, 50]), motion_names=('dance', 'walk'),
              motion_files=('/motions/dance.npz', '/motions/walk.npz'))
  command = NS(motion=motion, _initialization_frames=torch.tensor([0, 7, 3, 53, 59]),
               time_steps=torch.tensor([1, 8, 4, 54, 59]))
  heightmap = TerrainHeightMap(torch.zeros(2, 2), (-3., -3.), (3., 3.))
  env = NS(scene=NS(env_origins=torch.zeros(5, 3)), terrain_height_map=heightmap,
           command_manager=NS(get_term=lambda _: command))
  before = root.clone()
  with pytest.raises(RuntimeError, match='Reset XY is outside'):
    place_on_terrain(env, object(), ids, root, TerrainResetCfg())
  records = [json.loads(line.split(': ', 1)[1]) for line in capsys.readouterr().err.splitlines()]
  assert [r['env_id'] for r in records] == [3, 4]
  assert records[0]['motion_name'] == 'walk'
  assert records[0]['init_frame_global_k'] == 53 and records[0]['init_frame_local_k'] == 3
  assert records[0]['target_frame_global_f'] == 54
  assert records[0]['reset_xy_w'] == [12., -7.]
  torch.testing.assert_close(root, before)


@pytest.mark.parametrize('pitch', [0., math.pi/2, -math.pi/2])
def test_reset_footprint_uses_heading_and_vertical_pitch_fallback(pitch):
  from supermimic.terrain.reset import _sample_ground
  from supermimic.tools.rotations import axis_angle_to_mat, mat_to_quat
  # 90 degree heading: a local-forward sample must go world +Y, even at vertical pitch.
  yaw = axis_angle_to_mat(torch.tensor([0., 0., 1.]), torch.tensor(math.pi/2))
  tilt = axis_angle_to_mat(torch.tensor([0., 1., 0.]), torch.tensor(pitch))
  root = torch.zeros(1, 13); root[:, :2] = torch.tensor([2., 3.])
  root[:, 3:7] = mat_to_quat(yaw @ tilt)
  class Map:
    def sample(self, xy):
      torch.testing.assert_close(xy, torch.tensor([[[2., 4.], [1., 3.]]]), atol=1e-6, rtol=0)
      return torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool)
  _sample_ground(NS(terrain_height_map=Map()), root, torch.tensor([[1.,0.,0.],[0.,1.,0.]]))


@pytest.mark.parametrize('terrain', [False, True])
def test_command_writes_clipped_joints_before_placement_and_final_state_after(terrain):
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
