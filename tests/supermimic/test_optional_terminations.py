"""Manager configuration and imagination use the same optional failure terms."""
from types import SimpleNamespace as NS

import pytest
import torch

from supermimic.mdp.functional import MdpConfig
from supermimic.mdp.rotations import so3_exp


@pytest.mark.parametrize('height', [True, False])
@pytest.mark.parametrize('tilt', [True, False])
@pytest.mark.parametrize('disabled_value', ['absent', 'none'])
def test_disabled_failure_terms_match_hard_and_soft_termination(mdp, height, tilt, disabled_value):
  terms = {}
  for name, enabled, threshold in [('anchor_pos', height, .25), ('anchor_ori', tilt, .8)]:
    if enabled:
      terms[name] = NS(params={'threshold': threshold})
    elif disabled_value == 'none':
      terms[name] = None
  mdp.cfg = MdpConfig.from_env_cfg({}, terms, .02)
  f = torch.tensor([1, 1, 1])
  p, r, v, w = mdp.reference_root(f)
  # Only height fails, only tilt fails, then both fail.
  p = p + torch.tensor([[0., 0., 1.], [0., 0., 0.], [0., 0., 1.]])
  r = so3_exp(torch.tensor([[0., 0., 0.], [torch.pi, 0., 0.], [torch.pi, 0., 0.]]))
  state = (p.requires_grad_(), r.requires_grad_(), v, w, mdp.motion.joint_pos[f], mdp.motion.joint_vel[f])
  assert mdp.termination(state, f).tolist() == [height, tilt, height or tilt]
  z, angle = mdp.termination_errors(state, f)
  expected_alive = torch.ones(3)
  if height:
    expected_alive *= 1 - torch.sigmoid((z.detach() - .25) / .02)
  if tilt:
    expected_alive *= 1 - torch.sigmoid((angle.detach() - .8) / .04)
  actual = mdp.soft_termination(state, f)
  torch.testing.assert_close(actual, 1 - expected_alive)
  if height or tilt:
    actual.sum().backward()
    for tensor in state[:2]:
      if tensor.grad is not None:
        assert torch.isfinite(tensor.grad).all()
  else:
    assert actual.eq(0).all()


def test_disabled_rewards_and_custom_thresholds_are_read_from_configuration(mdp):
  rewards = {'motion_global_root_pos': None,
             'motion_global_root_ori': NS(weight=.7, params={'std': .6}),
             'action_rate_l2': NS(weight=-.2, params={}),
             'feet_stumble': NS(weight=0., params={}),
             'feet_at_plane': NS(weight=0., params={})}
  mdp.cfg = MdpConfig.from_env_cfg(rewards, {'anchor_ori': NS(params={'threshold': .4})}, .01)
  assert mdp.cfg.term_height is None and mdp.cfg.term_tilt == .4
  assert mdp.cfg.root_ori_std == .6
  mdp.validate_imagination_rewards()
  f = torch.tensor([1, 3])
  state = (*mdp.reference_root(f), mdp.motion.joint_pos[f], mdp.motion.joint_vel[f])
  action = torch.zeros(2, 29)
  previous = torch.ones_like(action)
  raw = mdp.reward_terms(state, f, action, previous)
  torch.testing.assert_close(mdp.reward(state, f, action, previous),
                             (.7*raw['motion_global_root_ori']-.2*raw['action_rate_l2'])*.01)


def test_staged_foot_costs_leave_imagined_reward_unchanged(mdp):
  f = torch.tensor([1, 4])
  state = (*mdp.reference_root(f), mdp.motion.joint_pos[f], mdp.motion.joint_vel[f])
  action = torch.ones(2, 29) * .2
  baseline = mdp.reward(state, f, action, torch.zeros_like(action))
  mdp.cfg.weights.update(feet_stumble=0., feet_at_plane=0.)
  mdp.validate_imagination_rewards()
  torch.testing.assert_close(mdp.reward(state, f, action, torch.zeros_like(action)), baseline, rtol=0, atol=0)
  mdp.cfg.weights['feet_stumble'] = -2.
  with pytest.raises(ValueError, match='WM reward heads'):
    mdp.validate_imagination_rewards()
