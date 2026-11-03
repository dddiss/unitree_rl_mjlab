"""Full-body placement, native FK agreement, and local-reset isolation."""
import json
import math
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import pytest
import torch

from supermimic.terrain.reset import RobotCollisionBounds, TerrainResetCfg, place_on_terrain
from supermimic.terrain.heightmap import TerrainHeightMap, terrain_bounds
from supermimic.tools.rotations import axis_angle_to_mat, mat_to_quat
from test_env_lifecycle import ROOT, methods


def sphere_geometry():
  model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body name="root"><freejoint/><geom name="body" size=".1"/></body>
  </worldbody></mujoco>''')
  return RobotCollisionBounds(model, [], [0], model.body('root').id)


def test_g1_collision_aabbs_match_native_fk_including_lying_poses(robot):
  model, joints, _, _ = robot
  bounds = RobotCollisionBounds(model, joints, np.arange(model.ngeom), model.body('pelvis').id)
  assert len(bounds.geom_ids) == 33
  assert 'head_collision' in bounds.geom_names
  rng = np.random.default_rng(42)
  count = 12
  p = torch.from_numpy(rng.normal(size=(count, 3))).float()
  quat = torch.from_numpy(rng.normal(size=(count, 4))).float()
  quat /= quat.norm(dim=-1, keepdim=True)
  quat[0] = torch.tensor([math.sqrt(.5), 0., math.sqrt(.5), 0.])
  q = torch.from_numpy(rng.uniform(-.5, .5, size=(count, len(joints)))).float()
  lo, hi = bounds.aabbs(p, quat, q)
  for n in range(count):
    data = mujoco.MjData(model)
    data.qpos[:3], data.qpos[3:7] = p[n], quat[n]
    data.qpos[model.jnt_qposadr[joints]] = q[n]
    mujoco.mj_forward(model, data)
    expected_lo, expected_hi = [], []
    for gid in bounds.geom_ids:
      center = data.geom_xpos[gid]
      if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CAPSULE:
        axis = data.geom_xmat[gid].reshape(3, 3)[:, 2]
        extent = model.geom_size[gid, 0] + np.abs(axis) * model.geom_size[gid, 1]
      else:
        assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE
        extent = np.full(3, model.geom_size[gid, 0])
      expected_lo.append(center - extent)
      expected_hi.append(center + extent)
    np.testing.assert_allclose(lo[n], expected_lo, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(hi[n], expected_hi, rtol=1e-4, atol=1e-5)


def test_fixed_bodies_joint_pivots_geom_rotations_and_collision_filtering():
  model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><worldbody>
    <geom name="ground" type="box" size="5 5 .1" pos="0 0 -.1"/>
    <body name="root"><freejoint/><geom name="root_visual" size=".5" contype="0" conaffinity="0"/>
      <body name="fixed" pos=".1 .2 .3" euler=".1 .2 .3">
        <geom name="box" type="box" pos=".05 -.03 .01" euler=".3 -.2 .1" size=".1 .2 .3"/>
        <body name="moving" pos=".2 0 .1" euler=".2 .1 .3">
          <joint name="hinge" type="hinge" axis="0 1 0" pos=".03 .04 .05"/>
          <geom name="ball" type="sphere" pos="0 0 .2" size=".04" contype="0" conaffinity="1"/>
        </body>
      </body>
    </body>
  </worldbody></mujoco>''')
  geom_ids = [model.geom(n).id for n in ('root_visual', 'box', 'ball')]
  bounds = RobotCollisionBounds(model, [model.joint('hinge').id], geom_ids, model.body('root').id)
  assert bounds.geom_names == ('box', 'ball')
  p, quat, q = torch.tensor([[2., -1., .8]]), torch.tensor([[.5, .5, -.5, .5]]), torch.tensor([[.7]])
  lo, hi = bounds.aabbs(p, quat, q)
  data = mujoco.MjData(model)
  data.qpos[:] = torch.cat((p, quat, q), -1)[0]
  mujoco.mj_forward(model, data)
  for i, gid in enumerate(bounds.geom_ids):
    rot = data.geom_xmat[gid].reshape(3, 3)
    if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
      center, ext = data.geom_xpos[gid], model.geom_size[gid, 0]
    else:
      center = data.geom_xpos[gid] + rot @ model.geom_aabb[gid, :3]
      ext = np.abs(rot) @ model.geom_aabb[gid, 3:]
    np.testing.assert_allclose(lo[0, i], center - ext, atol=1e-5)
    np.testing.assert_allclose(hi[0, i], center + ext, atol=1e-5)


@pytest.mark.parametrize('pitch', [0., math.pi/2, -math.pi/2, math.pi])
def test_g1_full_body_clearance_on_steps_is_batched_and_raises_only(robot, pitch, monkeypatch):
  model, joints, _, _ = robot
  bounds = RobotCollisionBounds(model, joints, np.arange(model.ngeom), model.body('pelvis').id)
  root = torch.zeros(3, 13)
  root[:, :3] = torch.tensor([[0., 0., .12], [0., 0., 3.], [1., 0., .8]])
  root[:, 3:7] = mat_to_quat(axis_angle_to_mat(torch.tensor([0., 1., 0.]), torch.tensor(pitch)))
  root[:, 7:] = torch.arange(6)
  q = torch.zeros(3, 29)
  ground = TerrainHeightMap(torch.zeros(240, 240), (-3., -3.), (3., 3.))
  ground.heights[130:150, 110:130] = .45
  env = NS(scene=NS(env_origins=torch.zeros(3, 3)), terrain_height_map=ground)
  original = root.clone()
  rng = torch.random.get_rng_state()
  def forbidden(*args, **kwargs):
    raise AssertionError('Placement must not run simulation or cast rays')
  monkeypatch.setattr(mujoco, 'mj_forward', forbidden)
  monkeypatch.setattr(mujoco, 'mj_ray', forbidden)
  placed = place_on_terrain(env, torch.arange(3), root, q, bounds, TerrainResetCfg())
  lo, hi = bounds.aabbs(placed[:, :3], placed[:, 3:7], q)
  heights, valid = ground.max_in_rect(lo[..., :2], hi[..., :2])
  assert valid.all()
  assert (lo[..., 2] >= heights + .01 - 1e-6).all()
  assert placed[1, 2] == pytest.approx(3.01)  # Airborne reference is not pulled down.
  assert (placed[:, 2] >= root[:, 2]).all()
  torch.testing.assert_close(placed[:, :2], root[:, :2])
  torch.testing.assert_close(placed[:, 3:], root[:, 3:])
  torch.testing.assert_close(original, root)
  assert torch.equal(rng, torch.random.get_rng_state())


def test_interior_obstacle_raises_fixed_upper_body_but_unrelated_obstacle_does_not():
  # A horizontal, fixed torso can cover a step missed by all four corners and
  # the centre. Checking feet alone would also miss this obstacle.
  model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="root">
    <freejoint/><geom name="foot" type="sphere" size=".05" pos="0 0 -.4"/>
    <body name="torso" pos=".6 0 -.2"><geom name="torso" type="box" size=".4 .1 .1"/></body>
  </body></worldbody></mujoco>''')
  bounds = RobotCollisionBounds(model, [], [0, 1], model.body('root').id)
  grid = TerrainHeightMap(torch.zeros(160, 160), (-2., -2.), (2., 2.))
  grid.heights[94, 80] = .9  # x=.35, between the torso corner and centre.
  root = torch.zeros(1, 13); root[:, 2] = .8; root[:, 3] = 1
  env = NS(scene=NS(env_origins=torch.zeros(1, 3)), terrain_height_map=grid)
  placed = place_on_terrain(env, torch.tensor([0]), root, torch.zeros(1, 0), bounds, TerrainResetCfg())
  assert placed[0, 2] == pytest.approx(1.21)  # .9 ground - (-.3 torso bottom) + .01
  grid.heights[94, 80] = 0.
  grid.heights[80, 96] = 2.  # Nearby y=.4, outside both geom footprints.
  placed = place_on_terrain(env, torch.tensor([0]), root, torch.zeros(1, 0), bounds, TerrainResetCfg())
  assert placed[0, 2] == pytest.approx(.81)


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
  assert lo[2] == pytest.approx(-2.1) and hi[2] == pytest.approx(-1.3)


def test_lookup_changes_only_selected_height_and_keeps_z_perturbation(capsys):
  ids = torch.tensor([0, 2])
  origins = torch.tensor([[10., 20., -1.], [10., 20., 0.], [10., 20., 2.]])
  root = torch.randn(2, 13)
  root[:, :2] = torch.tensor([[-3., 0.], [3., 0.]])
  root[:, 2] = origins[ids, 2] + torch.tensor([.81, .79])
  root[:, 3:7] = torch.tensor([1., 0., 0., 0.])
  heightmap = TerrainHeightMap(torch.tensor([[.2], [1.2]]), (-10., -10.), (10., 10.))
  env = NS(scene=NS(env_origins=origins), terrain_height_map=heightmap)
  geometry = sphere_geometry()
  rng = torch.random.get_rng_state()
  placed = place_on_terrain(env, ids, root, torch.zeros(2, 0), geometry, TerrainResetCfg())
  torch.testing.assert_close(placed[:, 2], torch.tensor([1.02, 2.0]))
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
    place_on_terrain(env, ids, root, torch.zeros(3, 0), sphere_geometry(), TerrainResetCfg())
  records = [json.loads(line.split(': ', 1)[1]) for line in capsys.readouterr().err.splitlines()]
  assert [r['env_id'] for r in records] == [3, 4]
  assert records[0]['motion_name'] == 'walk'
  assert records[0]['init_frame_global_k'] == 53 and records[0]['init_frame_local_k'] == 3
  assert records[0]['target_frame_global_f'] == 54
  assert records[0]['reset_xy_w'] == [12., -7.]
  assert records[0]['invalid_collision_geoms'] == ['body']
  torch.testing.assert_close(root, before)


@pytest.mark.parametrize('terrain', [False, True])
def test_command_writes_clipped_joints_before_placement_and_final_state_after(terrain):
  events = []
  geometry = object()
  def place(env, ids, state, joint_pos, bounds, cfg):
    events.append('probe')
    assert bounds is geometry and joint_pos.eq(1).all()
    assert state[0, 2] == pytest.approx(.8)
    result = state.clone(); result[:, 2] += .4
    return result
  ns = methods(ROOT / 'src/tasks/supermimic/mdp/commands.py', 'MotionCommand',
               {'_write_reference_state_to_sim'}, {'torch': torch, 'place_on_terrain': place})
  cmd = NS(cfg=NS(terrain_reset=TerrainResetCfg() if terrain else None), _env=NS(),
           _reset_collision_bounds=geometry)
  limits = torch.tensor([[[-1., 1.]]]).expand(3, 29, 2)
  def joints(q, v, env_ids):
    events.append('joints'); assert q.eq(1).all()
  def root(state, env_ids): events.append('root'); cmd.written = state
  cmd.robot = NS(data=NS(soft_joint_pos_limits=limits), write_joint_state_to_sim=joints,
                 write_root_state_to_sim=root, reset=lambda env_ids: events.append('reset'))
  ns['_write_reference_state_to_sim'](
    cmd, torch.tensor([1]), torch.tensor([[3., 4., .8]]), torch.tensor([[1., 0., 0., 0.]]),
    torch.ones(1, 3), torch.ones(1, 3)*2, torch.ones(1, 29)*3, torch.zeros(1, 29),
  )
  assert events == (['joints', 'probe', 'root', 'reset'] if terrain else ['joints', 'root', 'reset'])
  assert cmd.written[0, 2] == pytest.approx(1.2 if terrain else .8)
  torch.testing.assert_close(cmd.written[0, 7:], torch.tensor([1., 1., 1., 2., 2., 2.]))
