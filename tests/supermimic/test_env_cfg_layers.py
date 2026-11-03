"""Three configuration layers, exercised without constructing the simulator."""
import ast
from dataclasses import dataclass, field, replace
import math
from types import SimpleNamespace as NS

import pytest

from conftest import BODIES, ROOT
from supermimic.mdp import rewards, terminations
from supermimic.perception.camera import DepthCameraCfg, DepthPatternCfg
from supermimic.terrain.reset import TerrainResetCfg


FLAT_REWARDS = (
  'motion_global_root_pos', 'motion_global_root_ori', 'motion_body_pos',
  'motion_body_ori', 'motion_body_lin_vel', 'motion_body_ang_vel',
  'action_rate_l2', 'joint_limit',
)
TERRAIN_REWARDS = (
  'motion_global_root_pos', 'motion_global_root_ori', 'motion_upper_body_pos',
  'motion_lower_body_pos', 'motion_upper_body_ori', 'motion_lower_body_ori',
  'motion_body_lin_vel', 'motion_body_ang_vel', 'action_rate_l2', 'joint_limit',
  'feet_stumble', 'feet_at_plane',
)


@dataclass
class TermCfg:
  func: object
  weight: float = 1.0
  params: dict = field(default_factory=dict)
  time_out: bool = False


@dataclass
class GeometryCfg:
  geom_names_expr: tuple
  friction: tuple


@dataclass
class TerrainCfg:
  terrain_type: str
  geoms: tuple
  terrain_generator: object = None


class MotionCfg(NS):
  def __init__(self, **kwargs):
    super().__init__(terrain_reset=None, sampling_mode='return_based', **kwargs)


def load_config_factories(root=ROOT):
  """Execute real factory bodies using lightweight equivalents of mjlab configs.

  Physics construction is outside these tests; reward/failure callbacks and
  terrain-reset settings use the real project definitions.
  """
  def terrain(mode):
    return TerrainCfg('plane' if mode == 'plane' else 'generator',
                      (GeometryCfg(('.*',), (.3,)),), NS(border_width=17.))
  # Execute the camera's real configuration builder without importing mjlab.
  camera_path = root / 'src/tasks/supermimic/perception/camera.py'
  camera_class = next(n for n in ast.parse(camera_path.read_text()).body
                      if isinstance(n, ast.ClassDef) and n.name == 'DepthCameraCfg')
  builder = next(n for n in camera_class.body
                 if isinstance(n, ast.FunctionDef) and n.name == 'sensor_cfg')
  builder.body = [n for n in builder.body if not isinstance(n, ast.ImportFrom)]
  camera_namespace = dict(ObjRef=NS, DepthRayCastSensorCfg=NS, DepthPatternCfg=DepthPatternCfg)
  exec(compile(ast.Module(body=[builder], type_ignores=[]), str(camera_path), 'exec'), camera_namespace)
  class CameraCfg(DepthCameraCfg):
    sensor_cfg = camera_namespace['sensor_cfg']
  namespace = {name: NS for name in (
    'ManagerBasedRlEnvCfg', 'JointPositionActionCfg', 'ActionTermCfg', 'CommandTermCfg',
    'EventTermCfg', 'ObservationGroupCfg', 'ObservationTermCfg', 'SimulationCfg',
    'MujocoCfg', 'ContactMatch', 'ContactSensorCfg', 'ObjRef', 'GridPatternCfg',
    'WorldUpTerrainHeightSensorCfg',
  )}
  class ViewerConfig(NS):
    OriginType = NS(ASSET_BODY='asset_body')
  namespace.update(
    math=math, replace=replace, MotionCommandCfg=MotionCfg, ViewerConfig=ViewerConfig,
    SceneCfg=lambda **kw: NS(entities={}, sensors=(), spec_fn=None, **kw),
    SceneEntityCfg=lambda name, **kw: NS(name=name, **kw),
    RewardTermCfg=TermCfg, TerminationTermCfg=TermCfg,
    GeomCfg=GeometryCfg, TerrainEntityCfg=TerrainCfg, TerrainResetCfg=TerrainResetCfg,
    mdp=NS(**{**vars(rewards), **vars(terminations),
               'supermimic_observation': 'supermimic_observation'}),
    dr=NS(body_com_offset='body_com_offset', encoder_bias='encoder_bias',
          geom_friction='geom_friction'),
    G1_ACTION_SCALE={'joint': .25}, get_g1_robot_cfg=lambda: NS(model='g1'),
    unitree_g1_terrain_cfg=terrain, mark_terrain_padding_ray_group='mark_padding',
    DepthCameraCfg=CameraCfg,
  )
  for relative in ('tracking_env_cfg.py', 'config/g1/env_cfgs.py'):
    path = root / 'src/tasks/supermimic' / relative
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.Assign))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
  return namespace


def test_common_tracking_template_retains_main_defaults():
  cfg = load_config_factories()['make_tracking_env_cfg']()
  assert tuple(cfg.rewards) == FLAT_REWARDS
  assert [term.weight for term in cfg.rewards.values()] == [.5, .5, 1., 1., 1., 1., -.1, -2.]
  assert cfg.rewards['motion_global_root_pos'].func is rewards.motion_global_anchor_position_error_exp
  assert cfg.rewards['motion_body_pos'].func is rewards.motion_relative_body_position_error_exp
  assert cfg.rewards['motion_body_ori'].func is rewards.motion_relative_body_orientation_error_exp
  assert cfg.scene.terrain.terrain_type == 'plane'
  assert cfg.scene.entities == {} and cfg.scene.sensors == ()
  assert (cfg.sim.nconmax, cfg.sim.njmax) == (35, 250)
  assert cfg.commands['motion'].body_names == ()
  assert cfg.commands['motion'].anchor_body_name == ''
  assert cfg.commands['motion'].terrain_reset is None
  assert not cfg.auto_reset and cfg.decimation == 4 and cfg.sim.mujoco.timestep == .005


def test_g1_base_only_applies_robot_configuration():
  cfg = load_config_factories()['_make_g1_env_cfg']()
  assert tuple(cfg.rewards) == FLAT_REWARDS
  assert set(cfg.scene.entities) == {'robot'}
  assert cfg.commands['motion'].body_names == BODIES
  assert cfg.commands['motion'].anchor_body_name == 'pelvis'
  assert cfg.events['base_com'].params['asset_cfg'].body_names == ('torso_link',)
  assert cfg.events['foot_friction'].params['asset_cfg'].geom_names == r'.*_collision$'
  assert cfg.viewer.body_name == 'torso_link'
  assert cfg.scene.terrain.terrain_type == 'plane' and cfg.commands['motion'].terrain_reset is None


@pytest.mark.parametrize('task,kind', [('flat', None), ('terrain', None), ('terrain', 'plane')])
@pytest.mark.parametrize('play', [False, True])
@pytest.mark.parametrize('collect', [False, True])
def test_task_overrides_preserve_play_settings_and_explicit_sensors(task, kind, play, collect):
  factories = load_config_factories()
  factory = 'unitree_g1_env_cfg' if task == 'flat' else 'unitree_g1_terrain_env_cfg'
  cfg = factories[factory](
    play=play, terrain_type=kind, collect_foot_rewards=collect)
  names = FLAT_REWARDS if task == 'flat' else TERRAIN_REWARDS
  assert cfg.scene.terrain.geoms[0].friction == ((1.,) if play else (.3,))
  assert cfg.commands['motion'].sampling_mode == ('start' if play else 'return_based')
  assert tuple(cfg.rewards) == names
  anchor = cfg.rewards['motion_global_root_pos']
  assert anchor.func is (rewards.motion_global_anchor_position_error_exp if task == 'flat'
                         else rewards.motion_terrain_anchor_position_error_exp)
  assert anchor.weight == .5 and anchor.params == {'command_name': 'motion', 'std': .3}
  assert anchor.func.supermimic_reward.requires_height is (task == 'terrain')
  sensor_names = ['self_collision']
  if task == 'terrain':
    sensor_names += ['supermimic_depth', 'foot_collision_contacts', 'foot_support_contacts', 'foot_height_scanners']
  assert [s.name for s in cfg.scene.sensors] == sensor_names
  assert (cfg.sim.nconmax, cfg.sim.njmax) == ((35, 250) if task == 'flat' else (128, 512))
  assert tuple(cfg.terminations) == ('time_out', 'motion_end', 'anchor_pos', 'anchor_ori')
  assert cfg.terminations['time_out'].func is terminations.episode_time_out
  assert cfg.terminations['time_out'].func.supermimic_truncation == 'time_limit'
  assert cfg.terminations['motion_end'].time_out and cfg.terminations['time_out'].time_out
  assert cfg.terminations['anchor_pos'].params['threshold'] == .25
  assert cfg.terminations['anchor_ori'].params['threshold'] == .8
  cmd = cfg.commands['motion']
  if task == 'terrain' and kind != 'plane':
    assert cfg.scene.terrain.terrain_type == 'generator'
    assert cmd.terrain_reset == TerrainResetCfg(randomize_xy=not play)
    assert cfg.scene.spec_fn == 'mark_padding'
  else:
    assert cfg.scene.terrain.terrain_type == 'plane' and cmd.terrain_reset is None
  if play:
    assert cfg.episode_length_s == int(1e9)
    assert cmd.pose_range == cmd.velocity_range == {}
    assert cmd.joint_position_range == (0., 0.)
    assert tuple(cfg.events) == ('foot_friction',)
  else:
    assert cfg.episode_length_s == 10.
    assert cmd.pose_range['z'] == (-.01, .01) and cmd.velocity_range['x'] == (-.5, .5)
    assert cmd.joint_position_range == (-.1, .1)
    assert all(event.mode == 'startup' for event in cfg.events.values())


def test_task_and_common_configuration_instances_do_not_share_mutable_terms():
  factories = load_config_factories()
  terrain = factories['unitree_g1_terrain_env_cfg']()
  flat = factories['unitree_g1_env_cfg']()
  terrain.rewards['motion_global_root_pos'].params['std'] = 123.
  terrain.commands['motion'].pose_range['x'] = (0., 0.)
  for cfg in (flat, factories['make_tracking_env_cfg'](), factories['unitree_g1_terrain_env_cfg']()):
    assert cfg.rewards['motion_global_root_pos'].params['std'] == .3
    assert cfg.commands['motion'].pose_range['x'] == (-.05, .05)


def test_explicit_sensor_configuration_is_independent_of_reward_edits():
  cfg = load_config_factories()['unitree_g1_terrain_env_cfg'](collect_foot_rewards=False)
  sensors = cfg.scene.sensors
  cfg.rewards['feet_stumble'].weight = 0.
  cfg.rewards['feet_at_plane'].weight = -1.
  cfg.rewards.clear()
  assert cfg.scene.sensors is sensors
  assert [sensor.name for sensor in sensors][-3:] == [
    'foot_collision_contacts', 'foot_support_contacts', 'foot_height_scanners',
  ]


def test_depth_camera_is_task_owned_and_keeps_calibration():
  from dataclasses import asdict
  factories = load_config_factories()
  for name in ('make_tracking_env_cfg', '_make_g1_env_cfg', 'unitree_g1_env_cfg'):
    assert not any(isinstance(getattr(s, 'pattern', None), DepthPatternCfg)
                   for s in factories[name]().scene.sensors)
  sensor = factories['unitree_g1_terrain_env_cfg']().scene.sensors[1]
  calibration = sensor.pattern.camera
  assert asdict(calibration) == asdict(DepthCameraCfg())
  assert sensor.name == calibration.name
  assert vars(sensor.frame) == dict(type='body', name='torso_link', entity='robot')
  assert sensor.ray_alignment == 'base' and not sensor.exclude_parent_body
  assert sensor.include_geom_groups == (0, 2, 5)
  assert sensor.excluded_housing_meshes == ('head_link',)
  assert sensor.max_distance == calibration.max_distance / float(calibration.optical_rays()[..., 2].min()) + 1e-4
