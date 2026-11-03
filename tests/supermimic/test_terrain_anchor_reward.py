"""Terrain anchor targets share height context without altering flat tracking."""
from types import SimpleNamespace as NS

import pytest
import torch

from conftest import ROOT, BODIES
from supermimic.core.components import bind_analytic_manager_terms
from supermimic.core.functional import FunctionalMDP, MdpConfig
from supermimic.core.state import read_state, state_in_reference_frame
from supermimic.rl.algorithm import SuperMimicAlgorithm
from supermimic.terrain.heightmap import TerrainHeightMap
from test_algorithm import small_config
from test_env_cfg_layers import load_config_factories
from test_env_lifecycle import methods
from test_terrain_height_termination import SlopedGround


def terrain_config(height_failure=False):
  cfg = load_config_factories()['unitree_g1_terrain_env_cfg']()
  # Isolate this analytic reward; keep the real task's parameters/metadata.
  cfg.rewards = {'motion_global_root_pos': cfg.rewards['motion_global_root_pos']}
  if not height_failure:
    cfg.terminations.pop('anchor_pos')
  bind_analytic_manager_terms(cfg)
  return cfg


def apply_config(mdp, cfg):
  mdp.cfg = MdpConfig.from_env_cfg(cfg.rewards, cfg.terminations, .02, body_names=BODIES)


def reference_state(mdp, frames):
  return (*mdp.reference_root(frames), mdp.motion.joint_pos[frames], mdp.motion.joint_vel[frames])


def real_reward_env(mdp, cfg, frames, origins, offsets, state, ground):
  # Run actual environment assembly/readout/cache methods. Only physical FK
  # construction is supplied by the already compiled CPU G1 test fixture.
  ns = methods(ROOT/'src/tasks/supermimic/env.py', 'SuperMimicEnv',
    {'functional_mdp', 'shared_reward_terms'},
    dict(torch=torch, MdpConfig=MdpConfig, FunctionalMDP=FunctionalMDP,
         G1Kinematics=lambda *args: mdp.kin, read_state=read_state,
         state_in_reference_frame=state_in_reference_frame))
  class Env:
    functional_mdp = ns['functional_mdp']
    shared_reward_terms = ns['shared_reward_terms']
  env = Env()
  count = len(frames)
  # state uses environment-relative coordinates, including the episode XY shift.
  p, _, v, w, q, qd = state
  data = NS(body_link_pose_w=torch.cat((p+origins,
    mdp.motion.body_quat_w[frames, mdp.root_body_index]), -1)[:, None],
    body_link_vel_w=torch.cat((v, w), -1)[:, None], joint_pos=q, joint_vel=qd,
    default_joint_pos=mdp.default_q[None].expand(count, -1),
    soft_joint_pos_limits=torch.stack((mdp.soft_lo, mdp.soft_hi), -1)[None])
  robot = NS(data=data, indexing=NS(joint_ids=torch.arange(29),
    body_ids=torch.arange(mdp.motion.body_pos_w.shape[1])))
  cmd = NS(robot=robot, motion=mdp.motion, body_indexes=mdp.body_indexes,
           root_body_index=0, cfg=NS(body_names=BODIES),
           time_steps=frames, reference_xy_offset=offsets)
  env.cfg, env.device, env.step_dt = cfg, 'cpu', .02
  env._functional_mdp = env._reward_terms = None
  env.sim = NS(mj_model=None)
  env.scene = NS(env_origins=origins, terrain=NS(cfg=NS(terrain_type='generator')))
  env.terrain_height_map = NS(height=ground)
  env.command_manager = NS(get_term=lambda _: cmd)
  env.action_manager = NS(action=torch.zeros(count, 29), prev_action=torch.zeros(count, 29))
  env.terrain_tolerance = torch.ones(count, dtype=torch.bool)
  return env


@pytest.mark.parametrize('height_failure', [False, True])
def test_measured_and_imagined_reward_use_reference_xy_and_origin_z(mdp, height_failure):
  cfg = terrain_config(height_failure)
  f = torch.tensor([1, 2, 3, 4])
  origin = torch.tensor([[10., 20., .4], [-20., 5., -.2], [3., 4., 1.], [2., 6., .3]])
  offset = torch.tensor([[1., 2.], [3., -4.], [-2., 3.], [4., 1.]])
  ref = reference_state(mdp, f)
  query_xy = ref[0][:, :2] + origin[:, :2] + offset
  lift = .01*query_xy[:, 0] + .02*query_xy[:, 1] - origin[:, 2]
  error = torch.tensor([[.10, .05, -.1], [-.05, .10, .1], [.10, -.05, 0.], [-.10, -.05, .2]])
  p = ref[0] + torch.cat((offset, lift[:, None]), -1) + error
  ground = SlopedGround()
  env = real_reward_env(mdp, cfg, f, origin, offset, (p, *ref[1:]), ground)
  shared = env.functional_mdp
  assert shared.requires_reward_height_offset and shared.requires_height_offset
  assert shared.termination_ground is ground  # Wired even with height termination removed.
  term = cfg.rewards['motion_global_root_pos']
  measured = term.func(env, **term.params)
  expected = torch.exp(-error.square().sum(-1)/.3**2)
  torch.testing.assert_close(measured, expected, rtol=1e-4, atol=1e-5)
  torch.testing.assert_close(ground.queries[0], query_xy)
  term.func(env, **term.params)
  assert len(ground.queries) == 1  # The normal measured reward cache still applies.
  state = state_in_reference_frame(read_state(env), offset)
  action = env.action_manager.action
  alg = SuperMimicAlgorithm(shared, small_config(), 'cpu')
  imagined, _ = alg.reward_term(state, f, action, action, height_offset=lift)
  torch.testing.assert_close(imagined, measured * .5 * .02, rtol=1e-4, atol=1e-6)
  assert not shared.termination(state, f, height_offset=lift).any()


def test_height_request_survives_failure_removal_and_ends_with_reward_removal(mdp):
  cfg = terrain_config()
  apply_config(mdp, cfg)
  assert mdp.cfg.term_height is None and mdp.requires_height_offset
  assert mdp.requires_reward_height_offset
  cfg.rewards.clear()
  apply_config(mdp, cfg)
  assert not mdp.requires_reward_height_offset and not mdp.requires_height_offset


def test_flat_reward_is_unchanged_and_zero_lift_matches_legacy_formula(mdp):
  f = torch.tensor([1, 4])
  ref = reference_state(mdp, f)
  state = (ref[0] + .1, *ref[1:])
  a = torch.zeros(2, 29)
  original = mdp.reward_terms(state, f, a, a)
  # An available height offset must not affect the ordinary task's reward.
  supplied = mdp.reward_terms(state, f, a, a, height_offset=torch.tensor([.8, -.5]))
  for name in original:
    torch.testing.assert_close(supplied[name], original[name], rtol=0, atol=0)
  apply_config(mdp, terrain_config())
  zero = mdp.termination_height_offset(f, torch.zeros(2, 3), torch.zeros(2, 2))
  actual = mdp.reward_terms(state, f, a, a, height_offset=zero)['motion_global_root_pos']
  torch.testing.assert_close(actual, original['motion_global_root_pos'], rtol=0, atol=0)
  with pytest.raises(ValueError, match='reference height_offset'):
    mdp.reward_terms(state, f, a, a)


def test_reward_uses_current_target_across_height_jump_and_preserves_reference(mdp):
  apply_config(mdp, terrain_config())
  f = torch.tensor([1])
  state = reference_state(mdp, f)
  position = mdp.motion.body_pos_w.clone()
  linear = mdp.motion.body_lin_vel_w.clone()
  angular = mdp.motion.body_ang_vel_w.clone()
  goal = mdp.goal(state, f).clone()
  action = torch.zeros(1, 29)
  critic = mdp.critic_features(state, f, torch.zeros(1, 33), action).clone()
  boundary = float(mdp.motion.body_pos_w[1:3, mdp.root_body_index, 0].mean())
  heightmap = TerrainHeightMap(torch.tensor([[0.], [.3]]), (boundary-.1, -1.), (boundary+.1, 1.))
  mdp.termination_ground = heightmap.height
  origin, offset = torch.zeros(1, 3), torch.zeros(1, 2)
  current = mdp.termination_height_offset(f, origin, offset)
  advanced = mdp.termination_height_offset(f+1, origin, offset)
  reward = mdp.reward_terms(state, f, action, action, height_offset=current)['motion_global_root_pos']
  wrong = mdp.reward_terms(state, f, action, action, height_offset=advanced)['motion_global_root_pos']
  assert reward.item() == 1. and wrong.item() == pytest.approx(float(torch.exp(torch.tensor(-1.))))
  for actual, saved in ((mdp.motion.body_pos_w, position), (mdp.motion.body_lin_vel_w, linear),
                        (mdp.motion.body_ang_vel_w, angular), (mdp.goal(state, f), goal),
                        (mdp.critic_features(state, f, torch.zeros(1, 33), action), critic)):
    torch.testing.assert_close(actual, saved, rtol=0, atol=0)


def test_compiled_terrain_reward_keeps_z_gradient_without_height_failure(mdp):
  apply_config(mdp, terrain_config())
  alg = SuperMimicAlgorithm(mdp, small_config(), 'cpu')
  compiled = torch.compile(alg.reward_term, backend='aot_eager', fullgraph=True)
  f = torch.tensor([1, 2])
  ref = reference_state(mdp, f)
  lift = torch.tensor([.3, -.2])
  error = torch.tensor([-.05, .05])
  a = torch.zeros(2, 29)
  try:
    for fn in (alg.reward_term, compiled):
      p = ref[0].clone()
      p[:, 2] += lift + error
      p.requires_grad_()
      reward, _ = fn((p, *ref[1:]), f, a, a, height_offset=lift)
      expected = torch.exp(-error.square()/.3**2) * .5 * .02
      torch.testing.assert_close(reward, expected)
      gradient, = torch.autograd.grad(reward.sum(), p)
      torch.testing.assert_close(gradient[:, 2], -2*error/.3**2*expected)
      assert gradient[:, :2].eq(0).all()
      assert gradient[0, 2] > 0 and gradient[1, 2] < 0
  finally:
    torch.compiler.reset()
