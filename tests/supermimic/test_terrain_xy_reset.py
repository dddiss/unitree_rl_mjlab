"""Training-only XY placement and replay-safe reference translation."""
import ast
import copy
from types import SimpleNamespace as NS

import pytest
import torch

from supermimic.mdp.motion_lib import MotionReference
from supermimic.mdp.state import state_in_reference_frame, unflatten_state
from supermimic.rl.algorithm import SuperMimicAlgorithm
from supermimic.rl.buffer import ReplayBuffer
from supermimic.terrain import reset as terrain_reset
from supermimic.terrain.reset import TerrainResetCfg, place_on_terrain, terrain_tile_xy_bounds
from test_algorithm import filled_buffer, small_config
from test_buffer import transition
from test_env_lifecycle import ROOT, methods
from test_playback_controls import make_playback_env


@pytest.mark.parametrize('play', [False, True])
@pytest.mark.parametrize('kind', ['generator', 'plane'])
def test_factory_enables_xy_only_for_training_on_generated_tiles(play, kind):
  # Execute the actual configuration branch without importing GPU-only mjlab.
  path = ROOT / 'src/tasks/supermimic/config/g1/env_cfgs.py'
  fn = next(n for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == 'unitree_g1_flat_tracking_env_cfg')
  branch = next(n for n in fn.body if isinstance(n, ast.If) and 'TerrainResetCfg(' in ast.unparse(n))
  cmd = NS(terrain_reset=None)
  cfg = NS(scene=NS(terrain=NS(terrain_type=kind, terrain_generator=NS(border_width=17.)), sensors=()))
  namespace = dict(play=play, motion_cmd=cmd, cfg=cfg, TerrainResetCfg=TerrainResetCfg,
                   mark_terrain_padding_ray_group=lambda spec: None)
  exec(compile(ast.Module(body=[branch], type_ignores=[]), str(path), 'exec'), namespace)
  if kind == 'plane':
    assert cmd.terrain_reset is None
  else:
    assert cmd.terrain_reset.randomize_xy is (not play)


def placement_fixture():
  terrain = NS(cfg=NS(terrain_generator=NS(size=(8., 8.))),
               terrain_origins=torch.zeros(2, 2, 3),
               terrain_levels=torch.tensor([1, 0, 1]), terrain_types=torch.tensor([0, 1, 1]))
  # Deliberately off-center origins: bounds must follow grid cells, not origins.
  origins = torch.tensor([[5., -5., .1], [-3., 6., .2], [6., 6., .3]])
  class Scene(dict):
    pass
  scene = Scene()
  scene.terrain, scene.env_origins = terrain, origins
  env = NS(scene=scene)
  ids = torch.tensor([2, 0])
  root = torch.randn(2, 13)
  root[:, :2] = torch.tensor([[100., -100.], [-100., 100.]])
  root[:, 2] = origins[ids, 2] + .8
  root[:, 3:7] = torch.tensor([1., 0., 0., 0.])
  return env, ids, root


def test_random_xy_retries_only_misses_and_preserves_height_rule_and_velocities(monkeypatch):
  env, ids, root = placement_fixture()
  calls = []
  def probe(env, state, offsets):
    selected = ids if not calls else ids[1:]
    calls.append((selected.clone(), state.clone()))
    height = torch.tensor([.2, .4, .6])[selected, None].expand(-1, 273)
    valid = torch.ones_like(height, dtype=torch.bool)
    if len(calls) == 1:
      valid[1] = False
    return height, valid
  monkeypatch.setattr(terrain_reset, '_sample_ground', probe)
  placed = place_on_terrain(env, None, ids, root, TerrainResetCfg(randomize_xy=True))
  assert [call[0].tolist() for call in calls] == [[2, 0], [0]]
  torch.testing.assert_close(placed[0, :2], calls[0][1][0, :2])
  torch.testing.assert_close(placed[:, 2], torch.tensor([1.4, 1.0]))
  torch.testing.assert_close(placed[:, 3:], root[:, 3:])
  low, high = terrain_tile_xy_bounds(env.scene.terrain, ids, .65)
  torch.testing.assert_close(low, torch.tensor([[.65, .65], [.65, -7.35]]))
  for selected, candidate in calls:
    lo, hi = terrain_tile_xy_bounds(env.scene.terrain, selected, .65)
    assert ((candidate[:, :2] >= lo) & (candidate[:, :2] <= hi)).all()
  assert ((placed[:, :2] >= low) & (placed[:, :2] <= high)).all()


def test_random_xy_exhausted_retries_still_raises(monkeypatch, capsys):
  env, ids, root = placement_fixture()
  calls = []
  def probe(env, state, offsets):
    calls.append(state.clone())
    return torch.full((len(state), 273), float('nan')), torch.zeros(len(state), 273, dtype=torch.bool)
  monkeypatch.setattr(terrain_reset, '_sample_ground', probe)
  with pytest.raises(RuntimeError, match='Reset XY is outside'):
    place_on_terrain(env, None, ids, root, TerrainResetCfg(randomize_xy=True, xy_max_attempts=3))
  assert len(calls) == 3
  # Even missing diagnostic metadata does not hide the original height error.
  assert 'Could not collect height-map diagnostics' in capsys.readouterr().err


def test_command_moves_whole_reference_but_not_velocity_or_other_environment(mdp):
  env = make_playback_env(mdp.motion)
  cmd = env.command
  env.scene.env_origins = torch.tensor([[8., 16., 0.], [-8., 24., 0.]])
  names = {'_write_reference_state_to_sim', 'reference_origin_w', 'get_reference',
           'body_pos_w', 'anchor_pos_w', 'body_lin_vel_w', 'body_ang_vel_w'}
  def place(env, robot, ids, state, cfg):
    moved = state.clone()
    moved[:, :2] += torch.tensor([-3., 5.])
    return moved
  ns = methods(ROOT / 'src/tasks/supermimic/mdp/commands.py', 'MotionCommand', names,
               {'torch': torch, 'MotionReference': MotionReference, 'place_on_terrain': place})
  for name in names:
    setattr(type(cmd), name, ns[name])
  del cmd._write_reference_state_to_sim  # Use the real class method, not the fixture's stub.
  cmd.body_indexes = list(range(mdp.motion.body_pos_w.shape[1]))
  cmd.robot_anchor_body_index = 0
  cmd.cfg.terrain_reset = TerrainResetCfg(randomize_xy=True)
  cmd.reference_xy_offset[:] = torch.tensor([[11., 13.], [17., 19.]])
  written = {}
  def write(state, env_ids): written.update(ids=env_ids.clone(), state=state.clone())
  cmd.robot = NS(data=NS(soft_joint_pos_limits=torch.tensor([-100., 100.]).expand(2, 29, 2)),
                 write_joint_state_to_sim=lambda *args, **kw: None,
                 write_root_state_to_sim=write, reset=lambda env_ids: None)
  raw = mdp.motion.body_pos_w.clone()
  before = (cmd.time_steps[0].clone(), cmd.reference_xy_offset[0].clone())
  cmd.request_frames(torch.tensor([54]), torch.tensor([1]))
  env.reset(env_ids=torch.tensor([1]))
  assert cmd.time_steps[1] == 55  # Robot at k=54; target f=55.
  torch.testing.assert_close(cmd.time_steps[0], before[0])
  torch.testing.assert_close(cmd.reference_xy_offset[0], before[1])
  torch.testing.assert_close(cmd.reference_xy_offset[1], torch.tensor([-3., 5.]))
  expected = raw[54, 0] + env.scene.env_origins[1] + torch.tensor([-3., 5., 0.])
  torch.testing.assert_close(written['state'][0, :3], expected)
  torch.testing.assert_close(written['state'][0, 7:10], mdp.motion.body_lin_vel_w[54, 0])
  torch.testing.assert_close(written['state'][0, 10:13], mdp.motion.body_ang_vel_w[54, 0])
  # The same translation applies to any frame, not just the initialization frame.
  for frames in (torch.tensor([2, 52]), cmd.time_steps, torch.tensor([50, 57])):
    ref = cmd.get_reference(frames)
    torch.testing.assert_close(ref.body_pos_w, raw[frames] + cmd.reference_origin_w[:, None])
    torch.testing.assert_close(ref.body_lin_vel_w, mdp.motion.body_lin_vel_w[frames])
    torch.testing.assert_close(ref.body_ang_vel_w, mdp.motion.body_ang_vel_w[frames])
    torch.testing.assert_close(ref.body_quat_w, mdp.motion.body_quat_w[frames])
  torch.testing.assert_close(cmd.body_pos_w, cmd.get_reference(cmd.time_steps).body_pos_w)
  torch.testing.assert_close(cmd.anchor_pos_w, cmd.body_pos_w[:, 0])
  torch.testing.assert_close(mdp.motion.body_pos_w, raw)
  # The existing reset position error remains an error after both are translated.
  pos = raw[54:55, 0] + env.scene.env_origins[1] + torch.tensor([.04, -.03, .01])
  cmd._write_reference_state_to_sim(torch.tensor([1]), pos, torch.tensor([[1.,0.,0.,0.]]),
                                    torch.ones(1,3), torch.ones(1,3)*2, torch.zeros(1,29), torch.zeros(1,29))
  ref_k = cmd.get_reference(torch.tensor([12, 54])).body_pos_w[1, 0]
  torch.testing.assert_close(written['state'][0, :3] - ref_k, torch.tensor([.04, -.03, .01]))


def test_replay_offsets_survive_reset_ring_overwrite_and_history_padding(mdp):
  b = ReplayBuffer(4, 1, torch.zeros(1, 33), mdp.motion.clip_end, mdp.motion.clip_id, 32)
  for t in range(7):
    tr = transition([t+1], t % 3, done=t % 3 == 2)
    tr['s'][:, 0] = t
    tr['reference_xy_offset'][:] = torch.tensor([t//3 * 4., t//3 * -2.])
    b.add(tr)
    tr['reference_xy_offset'].fill_(1000.)  # A later reset cannot mutate replay.
  start, history = b.sample_starts(256, 10)
  assert start['reference_xy_offset'][:, 0].unique().numel() == 2
  for sampled in (start, history):
    episode = sampled['s'][..., 0].long() // 3
    expected = torch.stack((episode * 4., episode * -2.), -1)
    torch.testing.assert_close(sampled['reference_xy_offset'], expected)
  torch.testing.assert_close(history['reference_xy_offset'],
                             start['reference_xy_offset'][:, None].expand(-1, 10, -1))


def test_real_shared_reward_cache_rebases_translation(mdp):
  cfg = small_config()
  _, transitions = filled_buffer(mdp, cfg)
  tr = transitions[2]
  state = unflatten_state(tr['s1'])
  xy = torch.tensor([[2., -4.], [-3., 5.]])
  shifted = (state[0] + torch.nn.functional.pad(xy, (0, 1)), *state[1:])
  cmd = NS(time_steps=tr['f'], reference_xy_offset=xy)
  ns = methods(ROOT/'src/tasks/supermimic/env.py', 'SuperMimicEnv', {'shared_reward_terms'},
               {'read_state': lambda env: shifted, 'state_in_reference_frame': state_in_reference_frame})
  env = NS(_reward_terms=None, functional_mdp=mdp, command_manager=NS(get_term=lambda _: cmd),
           action_manager=NS(action=tr['a'], prev_action=tr['a_prev']), terrain_tolerance=tr['terrain_tolerance'])
  actual = ns['shared_reward_terms'](env)
  expected = mdp.reward_terms(state, tr['f'], tr['a'], tr['a_prev'], terrain_tolerance=tr['terrain_tolerance'])
  for key in expected:
    torch.testing.assert_close(actual[key], expected[key])


@pytest.mark.parametrize('visual', [False, True])
def test_translated_replay_gives_same_imagination_update_and_wm_deltas(mdp, visual):
  from test_visual_algorithm import visual_config, visual_buffer
  torch.set_num_threads(2)
  cfg = visual_config() if visual else small_config()
  original = SuperMimicAlgorithm(mdp, cfg, 'cpu')
  b, transitions = (visual_buffer if visual else filled_buffer)(mdp, cfg)
  # Binary-exact XY isolates coordinate semantics from float32 cancellation;
  # freshly initialized normalization/Adam can magnify sub-ULP differences.
  for key in ('s', 's1'):
    b.data[key][:8, :, :2] = b.data[key][:8, :, :2].mul(8).round().div(8)
  for row in range(len(transitions)):
    original.observe({key: value[row] for key, value in b.data.items()}, b.xi)
  translated = copy.deepcopy(original)
  moved = copy.deepcopy(b)
  xy = torch.tensor([[2., -4.], [-3., 5.]])
  moved.data['s'][:8, :, :2] += xy
  moved.data['s1'][:8, :, :2] += xy
  moved.data['reference_xy_offset'][:8] = xy
  # WM receives unchanged physical states; local increments/input features are invariant.
  s = unflatten_state(b.data['s'][0]); sm = unflatten_state(moved.data['s'][0])
  a, xi = b.data['a'][0], b.xi
  torch.testing.assert_close(original.wm.inputs(s, a, xi), original.wm.inputs(sm, a, xi))
  torch.testing.assert_close(original.wm.targets(s, unflatten_state(b.data['s1'][0])),
                             original.wm.targets(sm, unflatten_state(moved.data['s1'][0])))
  if visual:
    h = torch.zeros(2, cfg.depth_latent_dim)
    predicted, h1 = original.wm.step_visual(s, a, xi, h)
    predicted_moved, h1_moved = original.wm.step_visual(sm, a, xi, h)
    torch.testing.assert_close(h1, h1_moved)
  else:
    predicted = original.wm.step(s, a, xi)
    predicted_moved = original.wm.step(sm, a, xi)
  torch.testing.assert_close(predicted[0], state_in_reference_frame(predicted_moved, xy)[0])
  # An actual BPTT/Critic update must use the sampled episode's translation.
  torch.manual_seed(77)
  expected = original.policy_update(b)
  torch.manual_seed(77)
  actual = translated.policy_update(moved)
  for key in expected:
    assert actual[key] == pytest.approx(expected[key], rel=1e-4, abs=1e-6)
  for first, second in zip(original.pi_parameters + original.v_parameters,
                            translated.pi_parameters + translated.v_parameters):
    torch.testing.assert_close(first, second, rtol=1e-4, atol=1e-6)


def test_reference_translation_keeps_position_gradients():
  p = torch.randn(2, 3, requires_grad=True)
  others = tuple(torch.randn(2, 3) for _ in range(5))
  shifted = state_in_reference_frame((p, *others), torch.tensor([[3., -7.], [2., 4.]]))
  shifted[0].sum().backward()
  assert p.grad.eq(1).all()
  assert all(a is b for a, b in zip(shifted[1:], others))
