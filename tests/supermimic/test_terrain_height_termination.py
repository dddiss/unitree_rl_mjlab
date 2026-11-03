"""Reference-ground targets agree in real steps and counterfactual rollouts."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import pytest
import torch

from supermimic.core.state import unflatten_state
from supermimic.rl.algorithm import SuperMimicAlgorithm
from supermimic.rl.buffer import ReplayBuffer
from supermimic.terrain.heightmap import TerrainHeightMap
from test_algorithm import filled_buffer, small_config
from test_buffer import transition

ROOT = Path(__file__).resolve().parents[2]


class SlopedGround:
  def __init__(self):
    self.queries = []

  def __call__(self, xy):
    self.queries.append(xy.clone())
    return .01 * xy[..., 0] + .02 * xy[..., 1]


def real_height_term(env):
  # Load the actual manager adapter without importing GPU-only mjlab modules.
  path = ROOT / 'src/tasks/supermimic/mdp/terminations.py'
  fn = next(n for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == 'bad_anchor_pos_z_only')
  module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn],
                      type_ignores=[])
  ns = {'__package__': 'supermimic.mdp'}
  exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), ns)
  return ns['bad_anchor_pos_z_only'](env, 'motion', .25)


def test_real_and_imagined_height_match_with_origins_offsets_and_wrong_robot_xy(mdp):
  ground = mdp.termination_ground = SlopedGround()
  frames = torch.tensor([2, 3, 4, 5])
  origins = torch.tensor([[10., 20., .4], [-20., 5., -.2], [3., 4., 1.], [2., 6., .3]])
  offsets = torch.tensor([[1., 2.], [3., -4.], [-2., 3.], [4., 1.]])
  before = tuple(t.clone() for t in mdp.reference_root(frames))
  lift = mdp.termination_height_offset(frames, origins, offsets)
  expected_xy = before[0][:, :2] + origins[:, :2] + offsets
  torch.testing.assert_close(ground.queries[-1], expected_xy)
  expected = .01 * expected_xy[:, 0] + .02 * expected_xy[:, 1] - origins[:, 2]
  torch.testing.assert_close(lift, expected)
  p, r, v, w = before
  p = p.clone()
  p[:, :2] += offsets + 8.  # Deliberately far from reference XY.
  p[:, 2] += lift + torch.tensor([.24, .26, -.24, -.26])
  p.requires_grad_()
  state = (p, r, v, w, mdp.motion.joint_pos[frames], mdp.motion.joint_vel[frames])
  cmd = NS(time_steps=frames, reference_xy_offset=offsets, root_body_index=0)
  cmd.robot = NS(data=NS(body_link_pose_w=torch.cat((p.detach() + origins,
                                                    mdp.motion.body_quat_w[frames, mdp.root_body_index]), -1)[:, None],
                         body_link_vel_w=torch.cat((v, w), -1)[:, None],
                         joint_pos=state[4], joint_vel=state[5]))
  env = NS(functional_mdp=mdp, scene=NS(env_origins=origins),
           command_manager=NS(get_term=lambda _: cmd))
  assert real_height_term(env).tolist() == [False, True, False, True]
  assert mdp.termination(state, frames, height_offset=lift).tolist() == [False, True, False, True]
  z, tilt = mdp.termination_errors(state, frames, height_offset=lift)
  probability = mdp.soft_termination(state, frames, height_offset=lift)
  expected_p = 1 - (1-torch.sigmoid((z-.25)/.02)) * (1-torch.sigmoid((tilt-.8)/.04))
  torch.testing.assert_close(probability, expected_p)
  probability.sum().backward()
  assert p.grad[:, 2].abs().min() > 1
  assert p.grad[:2, 2].gt(0).all() and p.grad[2:, 2].lt(0).all()
  assert p.grad[:, :2].eq(0).all()  # Query is a reference target, not predicted XY.
  for actual, original in zip(mdp.reference_root(frames), before):
    torch.testing.assert_close(actual, original)


def test_ground_lift_affects_only_termination_not_reward_goal_or_velocity(mdp):
  cfg = small_config()
  algorithm = SuperMimicAlgorithm(mdp, cfg, 'cpu')
  buffer, transitions = filled_buffer(mdp, cfg)
  tr = transitions[0]
  state, f = unflatten_state(tr['s1']), tr['f']
  original_motion = {k: getattr(mdp.motion, k).clone() for k in ('body_pos_w', 'body_lin_vel_w', 'body_ang_vel_w')}
  goal = mdp.goal(state, f).clone()
  first_r, first_d = algorithm.reward_term(state, f, tr['a'], tr['a_prev'])
  lifted_r, lifted_d = algorithm.reward_term(state, f, tr['a'], tr['a_prev'], height_offset=torch.full((2,), .5))
  torch.testing.assert_close(first_r, lifted_r, rtol=0, atol=0)
  assert (lifted_d > first_d + .9).all()
  torch.testing.assert_close(mdp.goal(state, f), goal, rtol=0, atol=0)
  for key, value in original_motion.items():
    torch.testing.assert_close(getattr(mdp.motion, key), value, rtol=0, atol=0)
  mdp.cfg.term_height = None
  torch.testing.assert_close(mdp.soft_termination(state, f),
                             mdp.soft_termination(state, f, height_offset=torch.full((2,), 100.)))


def test_current_target_not_advanced_target_and_default_threshold(mdp):
  f = torch.tensor([1])
  state = (*mdp.reference_root(f), mdp.motion.joint_pos[f], mdp.motion.joint_vel[f])
  # Reference crosses a step between f and f+1, so a premature query is visible.
  mid = (mdp.motion.body_pos_w[1, 0, 0] + mdp.motion.body_pos_w[2, 0, 0]) / 2
  heightmap = TerrainHeightMap(torch.tensor([[0.], [.3]]),
                               (float(mid)-.1, -1.), (float(mid)+.1, 1.))
  mdp.termination_ground = heightmap.height
  origin, offset = torch.zeros(1, 3), torch.zeros(1, 2)
  now = mdp.termination_height_offset(f, origin, offset)
  later = mdp.termination_height_offset(f+1, origin, offset)
  assert not mdp.termination(state, f, height_offset=now).item()
  assert mdp.termination(state, f, height_offset=later).item()
  tree = ast.parse((ROOT/'src/tasks/supermimic/tracking_env_cfg.py').read_text())
  value = next(n.value for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)
               and isinstance(n.target, ast.Name) and n.target.id == 'terminations')
  class Mdp:
    def __getattr__(self, name):
      return name
  terms = eval(compile(ast.Expression(value), '<terminations>', 'eval'),
               {'mdp': Mdp(), 'TerminationTermCfg': NS, 'SceneEntityCfg': lambda name: name})
  assert terms['anchor_pos'].func == 'bad_anchor_pos_z_only'
  assert terms['anchor_pos'].params['threshold'] == .25


def test_compiled_reward_termination_preserves_values_and_state_gradient(mdp):
  alg = SuperMimicAlgorithm(mdp, small_config(), 'cpu')
  compiled = torch.compile(alg.reward_term, backend='eager', fullgraph=True)
  f = torch.tensor([1, 2])
  p, r, v, w = mdp.reference_root(f)
  p = p + torch.tensor([[0., 0., .54], [0., 0., .56]])
  action = torch.zeros(2, 29)
  results = []
  for fn in (alg.reward_term, compiled):
    pos = p.clone().requires_grad_()
    state = (pos, r, v, w, mdp.motion.joint_pos[f], mdp.motion.joint_vel[f])
    reward, done = fn(state, f, action, action, height_offset=torch.full((2,), .3))
    gradient, = torch.autograd.grad((reward * (1-done)).sum(), pos)
    results.append((reward, done, gradient))
  for eager, traced in zip(*results):
    torch.testing.assert_close(eager, traced)


def test_replay_origin_is_snapshot_across_resets_and_ring_overwrite(mdp):
  b = ReplayBuffer(3, 1, torch.zeros(1, 33), mdp.motion.clip_end, mdp.motion.clip_id, 32)
  for step in range(7):
    tr = transition([step+1], 0, done=True)
    tr['env_origin'][:] = torch.tensor([step, -step, step / 10])
    tr['reference_xy_offset'][:] = torch.tensor([step*2, -step*2])
    b.add(tr)
    tr['env_origin'].zero_()
  start, _ = b.sample_starts(128, 10)
  step = start['f'] - 1
  assert set(step.tolist()) == {4, 5, 6}
  torch.testing.assert_close(start['env_origin'], torch.stack((step, -step, step / 10), -1).float())
  torch.testing.assert_close(start['reference_xy_offset'], torch.stack((step*2, -step*2), -1).float())


@pytest.mark.parametrize('visual', [False, True])
@pytest.mark.parametrize('height_reward', [False, True])
def test_imagination_queries_future_targets_from_sampled_placement_without_future_rows(mdp, visual, height_reward):
  from test_visual_algorithm import visual_buffer, visual_config
  from supermimic.core.components import RewardBinding
  from supermimic.mdp.rewards import motion_terrain_anchor_position_error_exp
  torch.set_num_threads(2)
  if height_reward:
    mdp.cfg.reward_components = (RewardBinding('motion_global_root_pos',
      motion_terrain_anchor_position_error_exp.supermimic_reward, {'std': .3}),)
    mdp.cfg.term_height = None  # The reward must request the lookup on its own.
    assert mdp.requires_height_offset
  cfg = visual_config() if visual else small_config()
  cfg.horizon = 4
  alg = SuperMimicAlgorithm(mdp, cfg, 'cpu')
  b, transitions = (visual_buffer if visual else filled_buffer)(mdp, cfg)
  for tr in transitions:
    alg.observe(tr, b.xi)
  # Only one logged transition remains, with 1/2 valid future target frames.
  b.size = b.ptr = 1
  b.eligible.zero_(); b.eligible[0] = True
  b.data['f'][0] = torch.tensor([50, 56])
  b.data['ep_step'][0] = 0
  b.data['env_origin'][0] = torch.tensor([[4., 7., .2], [8., -2., .4]])
  b.data['reference_xy_offset'][0] = torch.tensor([[.5, .7], [-.2, .1]])
  ground = mdp.termination_ground = SlopedGround()
  sampled = {}
  sample = b.sample_starts
  def record_start(*args):
    start, history = sample(*args)
    sampled.update(start)
    return start, history
  b.sample_starts = record_start
  reward_calls = []
  reward = alg._reward_term
  def capture(state, f, a, prev, *args, **kwargs):
    reward_calls.append(f.clone())
    env_origin = sampled['env_origin'].repeat(cfg.horizon, 1)
    xy = mdp.motion.body_pos_w[f, mdp.root_body_index, :2] + env_origin[:, :2] + sampled['reference_xy_offset'].repeat(cfg.horizon, 1)
    expected = .01*xy[:, 0] + .02*xy[:, 1] - env_origin[:, 2]
    torch.testing.assert_close(kwargs['height_offset'], expected)
    assert not kwargs['height_offset'].requires_grad
    if height_reward:
      terms = mdp.reward_terms(state, f, a, prev, **kwargs)
      target = mdp.reference_root(f)[0].clone()
      target[:, 2] += expected
      torch.testing.assert_close(terms['motion_global_root_pos'],
                                 torch.exp(-(target-state[0]).square().sum(-1)/.3**2))
    return reward(state, f, a, prev, *args, **kwargs)
  alg._reward_term = capture
  metrics = alg.policy_update(b)
  assert metrics['policy_skipped'] == 0 and metrics['pi_gradnorm'] > 0
  assert len(ground.queries) == len(reward_calls) == 1
  frames = reward_calls[0].reshape(cfg.horizon, cfg.pi_batch)
  expected_frames = torch.minimum(sampled['f'][None] + torch.arange(cfg.horizon)[:, None],
                                  mdp.motion.clip_end[mdp.motion.clip_id[sampled['f']]][None])
  torch.testing.assert_close(frames, expected_frames)
