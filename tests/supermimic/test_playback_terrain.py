"""Run the real reset hook, playback wrapper and GUI callbacks with fake physics."""
import ast
import runpy
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

from supermimic.rl.evaluation import PlaybackEnv
from test_env_lifecycle import ROOT
from test_playback_controls import make_playback_env, open_gui


def terrain_playback_env(motion, *, enabled=True, num_envs=1):
  original = make_playback_env(motion, num_envs=num_envs)

  class ResetBase(type(original)):
    def reset(self, *, seed=None, env_ids=None, options=None):
      # mjlab 1.6 seeds before _reset_idx, then forwards/senses after it.
      if seed is not None:
        torch.manual_seed(seed)
      self._reset_idx(self.command._resolve_env_ids(env_ids))
      self.forwarded_origin = self.scene.env_origins.clone()
      return {'frame': self.command.time_steps.clone()}, {'reset': True}

    def _reset_idx(self, env_ids=None):
      super().reset(env_ids=env_ids)

  path = ROOT / 'src/tasks/supermimic/env.py'
  cls = next(n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.ClassDef) and n.name == 'SuperMimicEnv')
  cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name in {'reset', '_reset_idx', 'depth_image'}]
  namespace = {'torch': torch, 'ManagerBasedRlEnv': ResetBase}
  # Keep the class intact so zero-argument super() follows the real reset path.
  exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
  env = object.__new__(namespace['SuperMimicEnv'])
  env.__dict__.update(original.__dict__)
  env.command._env = env
  env.command._write_reference_state_to_sim = env.write_state
  env._randomize_terrain_on_reset = enabled
  env.reset_version = torch.zeros(num_envs, dtype=torch.long)
  env._depth_image = None
  env._depth_camera = NS(read=lambda: env.forwarded_origin[:, :1, None].expand(-1, 18, 32).clone())

  class Terrain:
    def __init__(self):
      self.calls = []

    def randomize_env_origins(self, ids):
      self.calls.append(ids.clone())
      # Deliberately move every invocation, exposing writes that use stale origins.
      env.scene.env_origins[ids] = torch.tensor([8., -8., .1]) * len(self.calls)

  env.scene.terrain = Terrain()
  return env


@pytest.mark.parametrize('path', ['first', 'manual', 'terminated', 'truncated', 'start_here'])
def test_each_play_reset_selects_terrain_before_writing_k_and_refreshing_depth(mdp, path):
  env = terrain_playback_env(mdp.motion)
  cmd = env.command
  playback = PlaybackEnv(env)
  assert env.num_envs == 1
  if path == 'start_here':
    (selector, frame, _, button), _, _ = open_gui(env, selected=0)
    cmd.on_viewer_pause(True)
    selector.value = '1: motion1'
    frame.value = 2
  for i in range(1, 4):
    if path == 'first':
      cmd.request_frames(torch.tensor([12]))
      env.reset()
      k = 12
    elif path == 'manual':
      playback.reset()
      k = 0
    elif path in {'terminated', 'truncated'}:
      getattr(env, path)[0] = True
      playback.step(torch.ones(1, 29))
      k = 0
    else:
      button.click()  # Dispatches env.reset directly, without PlaybackEnv.reset.
      k = 53
    assert len(env.scene.terrain.calls) == i
    assert cmd._initialization_frames.item() == k
    assert cmd.time_steps.item() == k + 1
    torch.testing.assert_close(env.root_pos[:, :2],
      mdp.motion.body_pos_w[k:k+1, 0, :2] + env.scene.env_origins[:, :2])
    torch.testing.assert_close(env.joint_pos, mdp.motion.joint_pos[k:k+1])
    assert env.reset_version.item() == i
    assert env.episode_length_buf.item() == 0
    assert env.actions.eq(0).all() and env.history.eq(0).all()
    assert env.depth_image.eq(env.scene.env_origins[0, 0]).all()


def test_subset_reset_only_moves_selected_environment(mdp):
  env = terrain_playback_env(mdp.motion, num_envs=2)
  env.reset()
  env.actions.fill_(3)
  env.history.fill_(7)
  before = [t[0].clone() for t in (env.command.time_steps, env.root_pos,
    env.scene.env_origins, env.actions, env.history, env.depth_image, env.reset_version)]
  PlaybackEnv(env).reset(env_ids=torch.tensor([1]))
  assert env.scene.terrain.calls[-1].tolist() == [1]
  for tensor, expected in zip((env.command.time_steps, env.root_pos,
      env.scene.env_origins, env.actions, env.history, env.depth_image, env.reset_version), before):
    torch.testing.assert_close(tensor[0], expected)
  assert env.command._initialization_frames[1] == 51
  assert env.command.time_steps[1] == 52


@pytest.mark.parametrize('enabled,has_terrain', [(False, True), (True, False)])
def test_disabled_or_missing_terrain_keeps_origins_and_random_stream(mdp, enabled, has_terrain):
  env = terrain_playback_env(mdp.motion, enabled=enabled)
  if not has_terrain:
    env.scene.terrain = None
  # Isolate the hook from the command sampler's own random draws.
  env.command._resample_command = lambda ids: None
  rng = torch.get_rng_state().clone()
  env.reset()
  torch.testing.assert_close(torch.get_rng_state(), rng)
  assert env.scene.env_origins.eq(0).all()
  if has_terrain:
    assert not env.scene.terrain.calls


def test_empty_reset_does_not_select_terrain(mdp):
  env = terrain_playback_env(mdp.motion)
  env.reset(env_ids=torch.empty(0, dtype=torch.long))
  assert not env.scene.terrain.calls
  assert env.reset_version.eq(0).all()


def test_reset_seed_precedes_terrain_selection(mdp):
  env = terrain_playback_env(mdp.motion)
  def randomize(ids):
    env.scene.env_origins[ids] = torch.rand(len(ids), 3)
  env.scene.terrain.randomize_env_origins = randomize
  env.reset(seed=17)
  before = env.scene.env_origins.clone()
  env.reset(seed=17)
  torch.testing.assert_close(env.scene.env_origins, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires Linux NVIDIA GPU and mjlab 1.6.0')
def test_real_play_terrain_uses_uniform_rows_columns_and_preserves_xy(mdp):
  from src.tasks.supermimic.config import make_env
  env = make_env(mdp.motion.motion_files, evaluation=True, num_envs=1,
                 randomize_terrain_on_reset=True)
  try:
    terrain = env.scene.terrain
    rows, cols = terrain.terrain_origins.shape[:2]
    cmd = env.command_manager.get_term('motion')
    for seed in (3, 17, 42):
      # These are the first two draws after mjlab reset(seed=...), independent
      # of the training mixture proportions and the initial difficulty limit.
      generator = torch.Generator(device=env.device).manual_seed(seed)
      row = torch.randint(rows, (1,), device=env.device, generator=generator)
      col = torch.randint(cols, (1,), device=env.device, generator=generator)
      k = torch.tensor([12], device=env.device)
      cmd.request_frames(k)
      env.reset(seed=seed)
      torch.testing.assert_close(terrain.terrain_levels, row)
      torch.testing.assert_close(terrain.terrain_types, col)
      torch.testing.assert_close(env.scene.env_origins, terrain.terrain_origins[row, col])
      torch.testing.assert_close(cmd.robot.data.body_link_pose_w[:, cmd.root_body_index, :2],
        cmd.motion.body_pos_w[k, cmd.root_body_index, :2] + env.scene.env_origins[:, :2])
      assert cmd.time_steps.item() == 13
  finally:
    env.close()


@pytest.mark.parametrize('evaluation,opt_in', [(False, False), (True, False), (True, True)])
def test_factory_keeps_training_and_evaluation_fixed_unless_play_opts_in(monkeypatch, evaluation, opt_in):
  from supermimic.config import make_env
  cfg = NS(scene=NS(num_envs=0), commands={'motion': NS()})
  env_cfgs = ModuleType('supermimic.config.g1.env_cfgs')
  env_cfgs.unitree_g1_flat_tracking_env_cfg = lambda **kwargs: cfg
  environment = ModuleType('supermimic.env')
  environment.SuperMimicEnv = lambda cfg, **kwargs: NS(cfg=cfg, **kwargs)
  camera = ModuleType('supermimic.perception.camera')
  camera.DepthCameraCfg = NS
  for module in (env_cfgs, environment, camera):
    monkeypatch.setitem(sys.modules, module.__name__, module)
  kwargs = {'randomize_terrain_on_reset': True} if opt_in else {}
  env = make_env('motion.npz', evaluation=evaluation, device='cpu', **kwargs)
  assert env.randomize_terrain_on_reset is opt_in
  assert env.cfg.scene.num_envs == 1


@pytest.mark.parametrize('mode', ['native', 'viser', 'video'])
def test_play_cli_enables_random_terrain_for_single_environment(monkeypatch, mode):
  events = []
  cmd = NS(motion=NS(motion_names=('dance',), clip_start=torch.tensor([0])),
           request_frames=lambda frames: events.append(('request', frames.tolist())))
  env = NS(command_manager=NS(get_term=lambda _: cmd),
           reset=lambda: events.append(('reset',)),
           close=lambda: events.append(('close',)))
  options = {}
  config = ModuleType('src.tasks.supermimic.config')
  def make_env(*args, **kwargs):
    options.update(kwargs)
    return env
  config.make_env = make_env
  evaluation = ModuleType('src.tasks.supermimic.rl.evaluation')
  evaluation.read_checkpoint_config = lambda _: NS(depth_enabled=True)
  evaluation.load_policy = lambda *args: (object(), NS(), {})
  evaluation.PlaybackEnv = PlaybackEnv
  def record_clips(*args):
    events.append(('video',))
    return []
  evaluation.record_clips = record_clips
  runner = ModuleType('src.tasks.supermimic.rl.runner')
  runner.PolicyController = lambda *args: object()
  viewers = ModuleType('mjlab.viewer')
  class Viewer:
    def __init__(self, wrapper, controller):
      assert wrapper.env is env
    def run(self):
      events.append(('viewer',))
  viewers.NativeMujocoViewer = viewers.ViserPlayViewer = Viewer
  for module in (config, evaluation, runner, viewers):
    monkeypatch.setitem(sys.modules, module.__name__, module)
  script = ROOT / 'scripts/play_supermimic.py'
  args = [str(script), '--motion-file', 'motions', '--checkpoint-file', 'last.pt', '--device', 'cpu']
  args += ['--video-dir', 'videos'] if mode == 'video' else ['--viewer', mode]
  monkeypatch.setattr(sys, 'argv', args)
  monkeypatch.setattr(sys, 'path', list(sys.path))
  runpy.run_path(str(script))['main']()
  assert options['num_envs'] == 1
  assert options['evaluation'] and options['randomize_terrain_on_reset']
  assert options['depth_enabled']
  assert events == ([('video',), ('close',)] if mode == 'video' else
                    [('request', [0]), ('reset',), ('viewer',), ('close',)])
