"""Exercise real playback/reset and command GUI logic without a display or GPU.

Only the simulator writes and Viser controls are replaced. The command methods
are extracted just as in test_env_lifecycle, so CPU tests do not import mjlab.
"""
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from supermimic.rl.evaluation import PlaybackEnv
from test_env_lifecycle import ROOT, methods


class GuiHandle:
  """Small synchronous Viser double, including programmatic update callbacks."""
  def __init__(self, initial_value=None, **kwargs):
    self.__dict__.update(kwargs)
    self.disabled = False
    self._value = initial_value
    self._updates = []
    self._clicks = []

  @property
  def value(self):
    return self._value

  @value.setter
  def value(self, value):
    if hasattr(self, 'min'):
      assert self.min <= value <= self.max
    if value == self._value:
      return
    self._value = value
    for callback in self._updates:
      callback(NS(target=self))

  def on_update(self, callback):
    self._updates.append(callback)
    return callback

  def on_click(self, callback):
    self._clicks.append(callback)
    return callback

  def click(self):
    assert not self.disabled
    for callback in self._clicks:
      callback(NS(target=self))


class Gui:
  def add_folder(self, name):
    return nullcontext()

  def add_dropdown(self, name, **kwargs):
    return GuiHandle(**kwargs)

  def add_slider(self, name, **kwargs):
    handle = GuiHandle(**kwargs)
    assert handle.min <= handle.value <= handle.max
    return handle

  def add_checkbox(self, name, **kwargs):
    return GuiHandle(**kwargs)

  def add_button(self, name):
    return GuiHandle()


def make_playback_env(motion, num_envs=2):
  # All reset noise ranges are zero for evaluation. The real motion reference
  # and resample method still select k and write its complete robot state.
  def identity_quaternion(roll, pitch, yaw):
    return torch.stack((torch.ones_like(roll), roll, pitch, yaw), dim=-1)

  namespace = methods(
    ROOT / 'src/tasks/supermimic/mdp/commands.py', 'MotionCommand',
    {'motion_ids', '_resolve_env_ids', '_resample_command', 'request_frames',
     'compute', '_get_gui_frame', 'create_gui', '_set_scrubber_disabled',
     'on_viewer_pause'},
    {'torch': torch,
     'sample_uniform': lambda lower, upper, size, device: torch.zeros(size, device=device),
     'quat_from_euler_xyz': identity_quaternion,
     'quat_mul': lambda identity, reference: reference},
  )
  command_type = type('Command', (), {
    name: value for name, value in namespace.items()
    if name in {'motion_ids', '_resolve_env_ids', '_resample_command',
                'request_frames', 'compute', '_get_gui_frame', 'create_gui',
                '_set_scrubber_disabled', 'on_viewer_pause'}
  })

  class Env:
    device = 'cpu'

    def __init__(self):
      self.num_envs = num_envs
      self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
      self.history = torch.zeros(num_envs, 10, 3)
      self.actions = torch.zeros(num_envs, 29)
      self.scene = NS(env_origins=torch.zeros(num_envs, 3))
      self.command = command_type()
      cmd = self.command
      cmd._env = self
      cmd.device = self.device
      cmd.motion = motion
      cmd._all_env_ids = torch.arange(num_envs)
      cmd.time_steps = torch.zeros(num_envs, dtype=torch.long)
      cmd._initialization_frames = torch.zeros_like(cmd.time_steps)
      cmd._requested_frames = torch.full_like(cmd.time_steps, -1)
      cmd.start_bin = torch.zeros_like(cmd.time_steps)
      cmd.root_body_index = 0
      cmd.cfg = NS(sampling_mode='start', pose_range={}, velocity_range={},
                   joint_position_range=(0., 0.))
      cmd.metrics = {name: torch.zeros(num_envs) for name in
                     ('sampling_entropy', 'sampling_top1_prob', 'sampling_top1_bin')}
      cmd._update_command = lambda ids: None
      cmd._update_metrics = lambda ids: None
      cmd._write_reference_state_to_sim = self.write_state
      self.command_manager = NS(get_term=lambda _: cmd)
      self.root_pos = torch.zeros(num_envs, 3)
      self.joint_pos = torch.zeros(num_envs, 29)
      self.terminated = torch.zeros(num_envs, dtype=torch.bool)
      self.truncated = torch.zeros(num_envs, dtype=torch.bool)
      self.reset_calls = []

    def write_state(self, ids, pos, quat, lin_vel, ang_vel, joint_pos, joint_vel):
      self.root_pos[ids] = pos
      self.joint_pos[ids] = joint_pos

    def reset(self, *, seed=None, env_ids=None, options=None):
      ids = self.command._resolve_env_ids(env_ids)
      self.reset_calls.append((seed, ids.clone(), options))
      self.command._resample_command(ids)
      self.episode_length_buf[ids] = 0
      self.history[ids] = 0
      self.actions[ids] = 0
      return {'frame': self.command.time_steps.clone()}, {'reset': True}

    def step(self, action):
      self.actions.copy_(action)
      self.episode_length_buf += 1
      done = self.terminated | self.truncated
      self.command.compute(torch.where(done, 0., .02))
      return {}, torch.ones(num_envs), self.terminated, self.truncated, {'step': True}

  env = Env()
  # Deliberately begin in different clips, away from their first frames.
  env.command.request_frames(torch.tensor([12, 54])[:num_envs])
  env.reset()
  env.reset_calls.clear()
  return env


def open_gui(env, selected=1):
  cmd = env.command
  changes = []
  requests = []

  def dispatch(action, payload):
    # mjlab's CUSTOM gui_reset path operates on the unwrapped environment:
    # it consumes the command's pending exact frame without a viewer Reset.
    requests.append((action, payload))
    ids = cmd._all_env_ids if payload['all_envs'] else torch.tensor([selected])
    env.reset(env_ids=ids)
    cmd.on_viewer_pause(True)

  cmd.create_gui('motion', NS(gui=Gui()), lambda: selected,
                 on_change=lambda: changes.append(True), request_action=dispatch)
  return cmd._scrubber_handles, changes, requests


def test_manual_reset_restarts_each_current_clip_and_forwards_reset_options(mdp, monkeypatch):
  env = make_playback_env(mdp.motion)
  # The default sampler would choose the wrong clip for row 1 without a request.
  monkeypatch.setattr(torch, 'randint', lambda high, size, device:
                      torch.zeros(size, dtype=torch.long, device=device))
  playback = PlaybackEnv(env)
  obs, info = playback.reset(seed=17, options={'example': 'forwarded'})
  assert env.command.motion_ids.tolist() == [0, 1]
  assert env.command._initialization_frames.tolist() == [0, 51]
  assert env.command.time_steps.tolist() == [1, 52]
  torch.testing.assert_close(env.root_pos, mdp.motion.body_pos_w[[0, 51], 0])
  assert obs['frame'].tolist() == [1, 52] and info == {'reset': True}
  seed, ids, options = env.reset_calls[-1]
  assert seed == 17 and ids.tolist() == [0, 1]
  assert options == {'example': 'forwarded'}


def test_manual_subset_reset_preserves_other_environment_buffers_and_target(mdp, monkeypatch):
  env = make_playback_env(mdp.motion)
  monkeypatch.setattr(torch, 'randint', lambda high, size, device:
                      torch.zeros(size, dtype=torch.long, device=device))
  env.episode_length_buf[:] = torch.tensor([8, 2])
  env.history[:] = 7
  env.actions[:] = 3
  before = (env.command.time_steps[0].clone(), env.root_pos[0].clone(),
            env.history[0].clone(), env.actions[0].clone())
  PlaybackEnv(env).reset(env_ids=torch.tensor([1]))
  assert env.command._initialization_frames.tolist() == [12, 51]
  assert env.command.time_steps[1] == 52
  assert env.episode_length_buf.tolist() == [8, 0]
  assert env.history[1].eq(0).all() and env.actions[1].eq(0).all()
  for actual, expected in zip(
      (env.command.time_steps[0], env.root_pos[0], env.history[0], env.actions[0]), before):
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('reason', ['terminated', 'truncated'])
def test_automatic_done_still_replays_current_clip_only_for_finished_rows(mdp, reason):
  env = make_playback_env(mdp.motion)
  getattr(env, reason)[1] = True
  previous_target = int(env.command.time_steps[0])
  obs, reward, done, info = PlaybackEnv(env).step(torch.ones(2, 29))
  assert done.tolist() == [False, True] and info == {'step': True}
  assert env.command.time_steps.tolist() == [previous_target + 1, 52]
  assert env.command._initialization_frames.tolist() == [12, 51]
  assert env.episode_length_buf.tolist() == [1, 0]
  assert env.actions[0].eq(1).all() and env.actions[1].eq(0).all()
  assert obs['frame'][1] == 52 and reward.tolist() == [1., 1.]


def test_initial_gui_reads_selected_environment_actual_clip_and_local_frame(mdp):
  env = make_playback_env(mdp.motion)
  env.episode_length_buf[1] = 2
  env.command.compute(.02, torch.tensor([1]))
  env.command.compute(.02, torch.tensor([1]))
  (selector, frame, _, _), _, _ = open_gui(env, selected=1)
  assert selector.value == '1: motion1'
  assert frame.max == 6 and frame.value == 5  # global 54 + 2, clip starts at 51
  assert frame.disabled
  env.command.on_viewer_pause(True)
  assert frame.value == 5 and not frame.disabled


@pytest.mark.parametrize('local_k', [2, 6])
def test_start_here_uses_k_without_repeated_click_drift_including_last_frame(mdp, local_k):
  env = make_playback_env(mdp.motion)
  (selector, frame, all_envs, button), _, requests = open_gui(env)
  env.command.on_viewer_pause(True)
  all_envs.value = False
  frame.value = local_k
  untouched_target = env.command.time_steps[0].clone()
  for _ in range(3):
    button.click()
    assert selector.value == '1: motion1'
    assert frame.value == local_k
    assert env.command._initialization_frames[1] == 51 + local_k
    assert env.command.time_steps[1] == min(52 + local_k, 57)
    assert env.episode_length_buf[1] == 0
    torch.testing.assert_close(env.root_pos[1], mdp.motion.body_pos_w[51 + local_k, 0])
    torch.testing.assert_close(env.command.time_steps[0], untouched_target)
  assert len(requests) == 3


def test_pause_tracks_current_restart_frame_without_confusing_final_target(mdp):
  env = make_playback_env(mdp.motion)
  env.command.request_frames(torch.tensor([53]), torch.tensor([1]))
  env.reset(env_ids=torch.tensor([1]))  # clip 1: k=2, f=3
  (_, frame, _, _), _, _ = open_gui(env)
  env.command.on_viewer_pause(True)
  assert frame.value == 2
  for expected_k in (3, 4, 5, 6, 6):
    env.step(torch.zeros(2, 29))
    env.command.on_viewer_pause(True)
    assert frame.value == expected_k
    assert env.command.time_steps[1] == min(51 + expected_k + 1, 57)


def test_selecting_clip_or_frame_changes_only_gui_until_start_here(mdp):
  env = make_playback_env(mdp.motion)
  (selector, frame, all_envs, button), changes, requests = open_gui(env)
  env.command.on_viewer_pause(True)
  before_target = env.command.time_steps.clone()
  before_state = env.root_pos.clone()
  selector.value = '0: motion0'
  assert frame.value == 0 and frame.max == 50
  frame.value = 8
  all_envs.value = False
  torch.testing.assert_close(env.command.time_steps, before_target)
  torch.testing.assert_close(env.root_pos, before_state)
  assert env.command._requested_frames.eq(-1).all()
  assert changes and not requests and not env.reset_calls
  button.click()
  assert env.command.motion_ids.tolist() == [0, 0]
  assert env.command._initialization_frames.tolist() == [12, 8]
  assert env.command.time_steps.tolist() == [13, 9]
  assert selector.value == '0: motion0' and frame.value == 8
