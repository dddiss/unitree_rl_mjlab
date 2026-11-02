"""Family-conditioned rewards retain their collected context through replay."""
import ast
from types import SimpleNamespace as NS

import pytest
import torch

from supermimic.tools.rotations import so3_exp, mat_to_quat, quat_to_mat
from supermimic.mdp.rewards import terrain_tolerance_radius
from supermimic.rl.buffer import ReplayBuffer
from conftest import ROOT, BODIES
from test_body_tracking_rewards import apply_config
from test_env_lifecycle import methods
from test_buffer import transition
from test_foot_reward_heads import training_case


def terrain_env(names=None):
  # Execute the real helper/property without importing GPU-only mjlab modules.
  path = ROOT / 'src/tasks/supermimic/env.py'
  helper = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'tolerance_by_terrain_column')
  ns = {'torch': torch}
  exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), 'exec'), ns)
  ns = methods(path, 'SuperMimicEnv', {'terrain_tolerance'}, ns)
  class Env:
    terrain_tolerance = ns['terrain_tolerance']
  env = Env()
  env.device, env.num_envs, env._terrain_tolerance_by_type = 'cpu', 3, None
  cfg = NS(terrain_type='plane' if names is None else 'generator',
           terrain_generator=None if names is None else
           NS(curriculum=True, sub_terrains=dict.fromkeys(names)))
  env.scene = NS(terrain=NS(cfg=cfg, terrain_types=torch.zeros(3, dtype=torch.long)))
  return env


@pytest.mark.parametrize('names', [None, ('flat',), ('slope',), ('stairs_up',), ('stairs_down',), ('rough',), ('platforms',)])
def test_plane_flat_and_single_terrain_family(names):
  env = terrain_env(names)
  expected = names not in (None, ('flat',))
  assert env.terrain_tolerance.tolist() == [expected] * 3


def test_mixed_columns_follow_config_order_and_refresh_after_subset_reset():
  env = terrain_env(('stairs_up', 'flat', 'rough'))  # Flat need not be column zero.
  terrain = env.scene.terrain
  terrain.terrain_types[:] = torch.tensor([0, 1, 2])
  terrain.terrain_levels = torch.tensor([0, 7, 9])
  rng = torch.get_rng_state().clone()
  previous = env.terrain_tolerance
  assert previous.tolist() == [True, False, True]
  # The same operation as the terrain selector during a partial reset.
  terrain.terrain_types[0] = 1
  assert env.terrain_tolerance.tolist() == [False, False, True]
  assert previous.tolist() == [True, False, True]
  terrain.terrain_levels[:] = torch.tensor([9, 0, 2])
  assert env.terrain_tolerance.tolist() == [False, False, True]
  assert torch.equal(torch.get_rng_state(), rng)


def test_random_layout_cannot_silently_treat_column_ids_as_family_ids():
  env = terrain_env(('flat', 'stairs_up'))
  env.scene.terrain.cfg.terrain_generator.curriculum = False
  with pytest.raises(ValueError, match='one family per column'):
    _ = env.terrain_tolerance


def test_same_error_is_strict_on_flat_and_tolerated_elsewhere_in_both_paths(mdp):
  terms = apply_config(mdp)
  f = torch.tensor([1, 1])
  state = (*mdp.reference_root(f), mdp.motion.joint_pos[f], mdp.motion.joint_vel[f])
  bp, br, _, _ = mdp.kin.forward(*state)
  # Identical robot/reference states in two environments. Deform just the
  # reference lower bodies by 4 cm and 5 degrees (within non-flat tolerance).
  lower = mdp.cfg.body_position_rewards['motion_lower_body_pos']['indices']
  motion_ids = mdp.body_indexes[list(lower)]
  mdp.motion.body_pos_w[1, motion_ids, 2] += .04
  ref_rot = quat_to_mat(mdp.motion.body_quat_w[1, motion_ids])
  mdp.motion.body_quat_w[1, motion_ids] = mat_to_quat(
    so3_exp(torch.tensor([torch.pi/36, 0., 0.])) @ ref_rot)
  origin = torch.tensor([[10., -5., 2.], [-20., 4., .3]])
  cmd = NS(cfg=NS(body_names=BODIES), root_body_index=0,
    robot_body_pos_w=bp+origin[:, None], robot_body_quat_w=mat_to_quat(br),
    body_pos_w=mdp.motion.body_pos_w[f][:, mdp.body_indexes]+origin[:, None],
    body_quat_w=mdp.motion.body_quat_w[f][:, mdp.body_indexes],
    robot=NS(data=NS(body_link_pos_w=bp+origin[:, None], body_link_quat_w=mat_to_quat(br))))
  env = NS(command_manager=NS(get_term=lambda _: cmd), terrain_tolerance=torch.tensor([False, True]))
  a = torch.zeros(2, 29)
  imagined = mdp.reward_terms(state, f, a, a, terrain_tolerance=env.terrain_tolerance)
  for name in (*mdp.cfg.body_position_rewards, *mdp.cfg.body_orientation_rewards):
    measured = terms[name].func(env, **terms[name].params)
    torch.testing.assert_close(measured, imagined[name], rtol=1e-4, atol=1e-5)
    if 'lower' in name:
      assert measured[0] < 1 and measured[1] == 1
    else:
      torch.testing.assert_close(measured[0], measured[1])
  # A fresh flat reset must disable the lower-body tolerance immediately.
  env.terrain_tolerance[:] = False
  for name in ('motion_lower_body_pos', 'motion_lower_body_ori'):
    measured = terms[name].func(env, **terms[name].params)
    torch.testing.assert_close(measured[0], measured[1])
  with pytest.raises(ValueError, match='sampled terrain_tolerance'):
    mdp.reward(state, f, a, a)


def test_gate_does_not_remove_gradient_outside_tolerance():
  # Test the hinge directly: on flat even an error inside the rough-terrain
  # tolerance remains penalized; outside both radii both rows have gradients.
  errors = torch.tensor([.04, .04, .10, .10], requires_grad=True)
  radius = terrain_tolerance_radius(.05, torch.tensor([False, True, False, True]))
  reward = torch.exp(-torch.relu(errors-radius).square()/.3**2)
  reward.sum().backward()
  assert errors.grad[0] < 0 and errors.grad[1] == 0
  assert errors.grad[2:].lt(0).all() and torch.isfinite(errors.grad).all()


def test_replay_preserves_collected_family_through_reset_and_ring_overwrite(mdp):
  m = mdp.motion
  buffer = ReplayBuffer(3, 2, torch.zeros(2, 33), m.clip_end, m.clip_id, 32)
  for t in range(7):
    tr = transition([50, 56], t % 2, done=bool(t % 2))
    tr['s'][:, 0] = t
    tr['s'][:, 1] = torch.arange(2)
    tr['terrain_tolerance'][:] = (t + torch.arange(2)) % 2 == 1
    buffer.add(tr)
    tr['terrain_tolerance'].logical_not_()  # Next reset cannot overwrite replay.
  start, history = buffer.sample_starts(100, 3)
  assert start['s'][:, 0].min() >= 4
  expected = ((start['s'][:, 0]+start['s'][:, 1]).long()%2) == 1
  torch.testing.assert_close(start['terrain_tolerance'], expected)
  assert start['terrain_tolerance'].any() and (~start['terrain_tolerance']).any()
  assert start['xi'].shape[-1] == 33 and start['s'].shape[-1] == 71
  assert 'terrain_tolerance' not in history  # Reward context is not policy input.
  assert 'terrain_tolerance' not in buffer.sample_windows(4, 2)
  tr.pop('terrain_tolerance')
  with pytest.raises(KeyError, match='terrain_tolerance'):
    buffer.add(tr)


@pytest.mark.parametrize('visual', [False, True])
def test_imagination_broadcasts_start_family_for_short_and_full_rollouts(mdp, monkeypatch, visual):
  apply_config(mdp)
  alg, replay, _ = training_case(mdp, visual, horizon=4, pi_batch=2)
  replay.data['terrain_tolerance'][:replay.size] = torch.tensor([False, True])
  start, history = replay.sample_starts(2, alg.cfg.hist)
  start['terrain_tolerance'] = torch.tensor([False, True])
  start['f'] = torch.tensor([50, 5])  # One valid step and a full horizon.
  monkeypatch.setattr(replay, 'sample_starts', lambda *args: (start, history))
  original = alg.reward_term
  seen = []
  compiled = torch.compile(original, backend='aot_eager', fullgraph=True)
  def capture(*args, **kwargs):
    flag = kwargs['terrain_tolerance']
    seen.append(flag.clone())
    torch.testing.assert_close(flag, torch.tensor([False, True]).repeat(4))
    eager = original(*args, **kwargs)
    result = compiled(*args, **kwargs)
    for a, b in zip(eager, result):
      torch.testing.assert_close(a, b)
    return result
  monkeypatch.setattr(alg, '_reward_term', capture)
  wm_before = {k: v.clone() for k, v in alg.wm.state_dict().items()}
  try:
    result = alg.policy_update(replay)
    assert len(seen) == 1 and result['policy_skipped'] == 0
    assert all(torch.isfinite(torch.tensor(v)) for v in result.values())
    assert all(torch.equal(v, wm_before[k]) for k, v in alg.wm.state_dict().items())
  finally:
    torch.compiler.reset()
