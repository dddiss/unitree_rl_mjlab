"""Root-relative, split-body rewards agree in simulation and imagination."""
import math
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import pytest
import torch

from conftest import BODIES
from supermimic.mdp import rewards
from supermimic.core.functional import MdpConfig
from supermimic.tools.rotations import so3_exp, mat_to_quat, quat_to_mat, yaw_mat
from test_env_cfg_layers import load_config_factories


def configured_rewards():
  return load_config_factories()['unitree_g1_terrain_env_cfg']().rewards


def apply_config(mdp):
  terms = configured_rewards()
  mdp.cfg = MdpConfig.from_env_cfg(terms, {}, .02, body_names=BODIES)
  mdp.validate_imagination_rewards(('feet_stumble', 'feet_at_plane'))
  return terms


def test_split_config_partitions_bodies_and_preserves_total_weights(mdp):
  terms = apply_config(mdp)
  upper = set(terms['motion_upper_body_pos'].params['body_names'])
  lower = set(terms['motion_lower_body_pos'].params['body_names'])
  assert not upper & lower
  assert upper | lower == set(BODIES) - {'pelvis'}
  assert 'torso_link' in upper and 'left_ankle_roll_link' in lower
  assert 'motion_body_pos' not in terms and 'motion_body_ori' not in terms
  for suffix in ('pos', 'ori'):
    up, low = terms[f'motion_upper_body_{suffix}'], terms[f'motion_lower_body_{suffix}']
    assert up.weight == low.weight == 1.0
    assert up.params['tolerance'] == 0
    assert up.params['std'] == low.params['std']
  assert terms['motion_lower_body_pos'].params['tolerance'] == .05
  assert terms['motion_lower_body_ori'].params['tolerance'] == math.radians(10)
  assert mdp.max_step_reward == pytest.approx(.14)
  assert mdp.cfg.weights['feet_stumble'] == -10
  assert mdp.cfg.weights['feet_at_plane'] == 0


def test_xyz_translation_and_yaw_alignment_do_not_change_body_reward():
  root = torch.tensor([[2., -3., 5.]])
  ref_root = torch.tensor([[-1., 4., .8]])
  ref_rot = so3_exp(torch.tensor([[.2, -.1, .4]]))
  yaw = so3_exp(torch.tensor([[0., 0., 1.2]]))
  rot = yaw @ ref_rot
  relative = torch.tensor([[[.3, .2, -.6], [-.4, .1, .5]]])
  ref_pos = ref_root[:, None] + relative
  body_pos = root[:, None] + (yaw[:, None] @ relative[..., None]).squeeze(-1)
  args = (body_pos, root, rot, ref_pos, ref_root, ref_rot)
  value = rewards.root_relative_body_position_reward(*args, std=.3)
  torch.testing.assert_close(value, torch.ones(1))
  # Independent XYZ shifts of robot and reference leave this reward invariant.
  shifted = rewards.root_relative_body_position_reward(
    body_pos + 7, root + 7, rot, ref_pos - 9, ref_root - 9, ref_rot, std=.3)
  torch.testing.assert_close(shifted, value)
  # A relative Z deformation still matters; only whole-root translation cancels.
  body_pos[:, 0, 2] += .2
  changed = rewards.root_relative_body_position_reward(
    body_pos, root, rot, ref_pos, ref_root, ref_rot, std=.3)
  torch.testing.assert_close(changed, torch.tensor([math.exp(-.2**2 / 2 / .3**2)]))


def test_tolerance_applies_per_body_and_upper_body_remains_strict():
  root = torch.zeros(4, 3)
  rot = torch.eye(3).expand(4, 3, 3)
  ref = torch.zeros(4, 2, 3)
  pos = ref.clone()
  pos[:, 0, 2] = torch.tensor([0., .04, .05, .15])
  pos.requires_grad_()
  lower = rewards.root_relative_body_position_reward(pos, root, rot, ref, root, rot, std=.3, tolerance=.05)
  upper = rewards.root_relative_body_position_reward(pos, root, rot, ref, root, rot, std=.3)
  torch.testing.assert_close(lower[:3], torch.ones(3))
  assert upper[1] < 1 and upper[2] < 1
  torch.testing.assert_close(lower[3], torch.tensor(math.exp(-.1**2 / 2 / .3**2)))
  lower.sum().backward()
  assert torch.isfinite(pos.grad).all() and pos.grad[:3].eq(0).all()
  assert pos.grad[3, 0, 2] < 0


def test_orientation_tolerance_preserves_yaw_alignment_and_has_finite_gradients():
  angles = torch.tensor([[0., 0., 0.], [math.radians(5), 0., 0.],
                         [math.radians(20), 0., 0.]], requires_grad=True)
  identity = torch.eye(3).expand(3, 3, 3)
  body = so3_exp(angles)[:, None]
  ref = identity[:, None]
  lower = rewards.body_orientation_tolerance_reward(
    body, identity, ref, identity, std=.4, tolerance=math.radians(10))
  torch.testing.assert_close(lower[:2], torch.ones(2))
  torch.testing.assert_close(lower[2], torch.tensor(math.exp(-math.radians(10)**2/.4**2)))
  upper = rewards.body_orientation_tolerance_reward(body, identity, ref, identity, std=.4)
  assert upper[1] < 1
  yaw = so3_exp(torch.tensor([[0., 0., .8]]).expand(3, -1))
  torch.testing.assert_close(rewards.body_orientation_tolerance_reward(
    yaw[:, None] @ body, yaw, ref, identity, std=.4, tolerance=math.radians(10)), lower)
  lower.sum().backward()
  assert torch.isfinite(angles.grad).all() and angles.grad[:2].eq(0).all()
  assert angles.grad[2, 0] < 0


def test_new_manager_rewards_match_fk_with_nonzero_origin_and_root_rotation(mdp, robot):
  terms = apply_config(mdp)
  model, _, bodies, _ = robot
  root = model.body('pelvis').id
  d = mujoco.MjData(model)
  d.qpos[:3] = [.4, -.2, 1.2]
  d.qpos[3:7] = mat_to_quat(so3_exp(torch.tensor([.2, -.3, .7]))).numpy()
  d.qpos[7:] = np.linspace(-.2, .3, 29)
  d.qvel[:] = np.linspace(-.1, .2, model.nv)
  mujoco.mj_forward(model, d)
  velocity = np.zeros(6)
  mujoco.mj_objectVelocity(model, d, mujoco.mjtObj.mjOBJ_XBODY, root, velocity, 0)
  ten = lambda x: torch.tensor(np.array(x), dtype=torch.float32)[None]
  state = (ten(d.xpos[root]), quat_to_mat(ten(d.xquat[root])), ten(velocity[3:]),
           ten(velocity[:3]), ten(d.qpos[7:]), ten(d.qvel[6:]))
  f = torch.tensor([3])
  origin = torch.tensor([[14., -20., 3.]])
  cmd = NS(cfg=NS(body_names=BODIES), root_body_index=root-1,
    robot=NS(data=NS(body_link_pos_w=ten(d.xpos[1:])+origin[:, None],
                    body_link_quat_w=ten(d.xquat[1:]))),
    robot_body_pos_w=ten(d.xpos[bodies])+origin[:, None],
    robot_body_quat_w=ten(d.xquat[bodies]),
    body_pos_w=mdp.motion.body_pos_w[f][:, mdp.body_indexes]+origin[:, None],
    body_quat_w=mdp.motion.body_quat_w[f][:, mdp.body_indexes])
  env = NS(command_manager=NS(get_term=lambda name: cmd), terrain_tolerance=torch.ones(1, dtype=torch.bool))
  a = torch.zeros(1, 29)
  imagined = mdp.reward_terms(state, f, a, a, terrain_tolerance=env.terrain_tolerance,
                              height_offset=torch.zeros_like(f, dtype=torch.float32))
  for name in (*mdp.cfg.body_position_rewards, *mdp.cfg.body_orientation_rewards):
    actual = terms[name].func(env, **terms[name].params)
    torch.testing.assert_close(actual, imagined[name], rtol=1e-4, atol=1e-5)
  # The current target is used, not a cached relative pose or reference f+1.
  cmd.body_pos_w[:, 7, 2] += .4
  assert not torch.allclose(terms['motion_upper_body_pos'].func(env, **terms['motion_upper_body_pos'].params),
                            imagined['motion_upper_body_pos'])


def test_batched_compiled_reward_and_backward_match_eager_and_legacy_terms(mdp):
  f = torch.tensor([[1, 4], [5, 6]])
  p, r, v, w = mdp.reference_root(f)
  q = (mdp.motion.joint_pos[f] + .2).requires_grad_()
  state = (p + torch.tensor([.1, -.2, .4]), r, v, w, q, mdp.motion.joint_vel[f])
  a = torch.zeros(2, 2, 29)
  old_terms = mdp.reward_terms(state, f, a, a)
  apply_config(mdp)
  # Zero terrain lift isolates the existing body-reward/gradient comparison.
  context = {'terrain_tolerance': torch.tensor([[False, True], [True, False]]),
             'height_offset': torch.zeros_like(f, dtype=torch.float32)}
  new_terms = mdp.reward_terms(state, f, a, a, **context)
  for name, old in old_terms.items():
    torch.testing.assert_close(new_terms[name], old, rtol=0, atol=0)
  eager = mdp.reward(state, f, a, a, **context)
  expected = sum(mdp.cfg.weights.get(name, 0) * value for name, value in new_terms.items()) * .02
  torch.testing.assert_close(eager, expected)
  compiled = torch.compile(mdp.reward, backend='aot_eager', fullgraph=True)
  output = compiled(state, f, a, a, **context)
  torch.testing.assert_close(output, eager)
  grads = [torch.autograd.grad(value.sum(), q, retain_graph=True)[0] for value in (eager, output)]
  assert torch.isfinite(grads[0]).all() and grads[0].abs().sum() > 0
  torch.testing.assert_close(*grads)
  torch.compiler.reset()


def test_new_terms_can_be_removed_or_retuned_from_configuration(mdp):
  terms = configured_rewards()
  del terms['motion_lower_body_pos']
  terms['motion_lower_body_ori'].params['tolerance'] = .3
  mdp.cfg = MdpConfig.from_env_cfg(terms, {}, .02, body_names=BODIES)
  assert 'motion_lower_body_pos' not in mdp.cfg.body_position_rewards
  assert mdp.cfg.body_orientation_rewards['motion_lower_body_ori']['tolerance'] == .3
  terms['motion_upper_body_pos'].params['body_names'] = ('typo',)
  with pytest.raises(ValueError, match='tracked robot bodies'):
    MdpConfig.from_env_cfg(terms, {}, .02, body_names=BODIES)
