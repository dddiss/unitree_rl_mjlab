"""Measured cost supervision and differentiable, transition-aligned imagination."""
import copy
import csv

import pytest
import torch

from supermimic.config.g1.rl_cfg import SuperMimicConfig
from supermimic.mdp.foot_rewards import FOOT_REWARD_NAMES
from supermimic.mdp.state import unflatten_state, normalize_xi
from supermimic.rl.algorithm import SuperMimicAlgorithm, frozen
from supermimic.rl import networks
from supermimic.rl.buffer import ReplayBuffer
from supermimic.rl.networks import WorldModel
from supermimic.rl.runner import SuperMimicRunner
from test_algorithm import small_config, filled_buffer
from test_train_resume_eval import TensorEnv
from test_visual_algorithm import visual_config, visual_buffer
from test_foot_rewards import sensor_configurator
from test_logging_backend import sdk
from types import SimpleNamespace as NS


def training_case(mdp, visual=False, **options):
  torch.set_num_threads(2)
  mdp.cfg.weights.update(feet_stumble=0., feet_at_plane=0.)
  cfg = (visual_config if visual else small_config)(**options)
  alg = SuperMimicAlgorithm(mdp, cfg, 'cpu')
  _, transitions = (visual_buffer if visual else filled_buffer)(mdp, cfg)
  motion = mdp.motion
  replay = ReplayBuffer(cfg.buffer_steps, 2, torch.zeros(2, 33), motion.clip_end,
                        motion.clip_id, cfg.horizon,
                        cfg.depth_history_length if visual else 0,
                        foot_reward_names=FOOT_REWARD_NAMES)
  for t, tr in enumerate(transitions):
    tr['foot_costs'] = torch.tensor([[.1 + t * .01, .8], [.4, 1.2 - t * .02]])
    replay.add(tr)
    alg.observe(tr, replay.xi)
  return alg, replay, transitions


@pytest.mark.parametrize('name,default', [('feet_stumble', 1.), ('feet_at_plane', 0.)])
def test_cost_loss_weights_configurable_and_separate_from_reward_weights(name, default):
  field = name + '_loss_weight'
  assert getattr(SuperMimicConfig(), field) == default
  assert getattr(SuperMimicConfig(**{field: 0.}), field) == 0.
  assert getattr(SuperMimicConfig(**{field: 1.}), field) == 1.
  for value in (-1., float('inf'), float('nan')):
    with pytest.raises(ValueError, match=field):
      SuperMimicConfig(**{field: value})


def test_shared_head_branches_from_2048_features_with_two_raw_outputs():
  wm = WorldModel(foot_reward_names=FOOT_REWARD_NAMES)
  assert wm.foot_reward_names == FOOT_REWARD_NAMES
  assert wm.foot_head[0][0].in_features == 2048
  assert wm.foot_head[0][0].out_features == 256
  assert wm.foot_head[0][-1].out_features == 2
  assert sum(isinstance(m, torch.nn.Linear) for m in wm.foot_head.modules()) == 2
  with torch.no_grad():
    wm.foot_head[0][-1].weight.zero_()
    wm.foot_head[0][-1].bias.copy_(torch.tensor([-1., 1.]))
    costs = wm.foot_head(torch.zeros(2, 2048))
  torch.testing.assert_close(costs, torch.nn.functional.softplus(
    torch.tensor([[-1., 1.]])).expand(2, -1))


@pytest.mark.parametrize('visual', [False, True])
def test_single_trunk_pass_and_cost_gradients_without_contact_inputs(mdp, visual):
  alg, replay, _ = training_case(mdp, visual)
  w = replay.sample_windows(2, alg.cfg.wm_window)
  state = unflatten_state(w['s'][:, 0])
  action = torch.zeros(2, 29, requires_grad=True)
  latent = torch.randn(2, alg.cfg.depth_latent_dim, requires_grad=True) if visual else None
  calls, features, inputs = [], [], []
  handles = [alg.wm.net[0].register_forward_hook(lambda *args: calls.append(1)),
             alg.wm.net[-2].register_forward_hook(lambda m, a, y: features.append(y)),
             alg.wm.foot_head[0][0].register_forward_pre_hook(
               lambda m, a: inputs.append(a[0]))]
  try:
    with frozen(alg.wm):
      state1, h1, costs = alg.wm.step_with_foot_costs(state, action, normalize_xi(replay.xi), latent)
      assert calls == [1]
      torch.testing.assert_close(inputs[0], features[0], rtol=0, atol=0)
      assert costs.shape == (2, 2) and costs.ge(0).all()
      costs.sum().backward()
      assert torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0
      if visual:
        assert torch.isfinite(latent.grad).all() and latent.grad.abs().sum() > 0
      assert all(p.grad is None for p in alg.wm.parameters())
  finally:
    for handle in handles:
      handle.remove()
  with torch.no_grad():
    plain = alg.wm._predict(state, action, normalize_xi(replay.xi), depth_latent=latent)
  for expected, actual in zip(plain[0], state1):
    torch.testing.assert_close(actual, expected)
  torch.testing.assert_close(h1, plain[1])


@pytest.mark.parametrize('visual', [False, True])
def test_wm_supervises_raw_costs_with_mask_and_independent_loss_weights(mdp, visual, monkeypatch):
  alg, replay, _ = training_case(mdp, visual, wm_window=4, state_loss_weight=0.,
    latent_loss_weight=0., depth_loss_weight=0., feet_stumble_loss_weight=3., feet_at_plane_loss_weight=.7)
  c = alg.cfg
  w = replay.sample_windows(c.wm_batch, c.wm_window)
  # Terminal row 1 stays valid. Costs in later episodes must have no influence.
  w['terminated'][:, 1] = True
  w['mask'][:, 2:] = 0
  w['foot_costs'][:, 2:] = 10000.
  monkeypatch.setattr(replay, 'sample_windows', lambda *args, **kwargs: w)
  with torch.no_grad():
    predicted_costs = alg._world_rollout(w['s'], w['a'], w['xi'], w.get('depth_history'))[-1]
    losses = ((predicted_costs - w['foot_costs']).square() * w['mask'][..., None]).sum((0, 1)) / c.wm_batch
  old_head = copy.deepcopy(alg.wm.foot_head.state_dict())
  old_policy = copy.deepcopy(alg.policy.dec.state_dict())
  old_critic = copy.deepcopy(alg.critic.net.state_dict())
  metrics = alg.world_model_update(replay)
  assert metrics['wm_feet_stumble_loss'] == pytest.approx(float(losses[0] / c.wm_window))
  assert metrics['wm_feet_at_plane_loss'] == pytest.approx(float(losses[1] / c.wm_window))
  assert metrics['wm_total_loss'] == pytest.approx(float((3*losses[0] + .7*losses[1]) / c.wm_window))
  assert any(not torch.equal(value, alg.wm.foot_head.state_dict()[key]) for key, value in old_head.items())
  for module, before in ((alg.policy.dec, old_policy), (alg.critic.net, old_critic)):
    for key, value in before.items():
      torch.testing.assert_close(module.state_dict()[key], value, rtol=0, atol=0)
  # Zero task reward weights do not disable label learning or shared perception.
  assert all(mdp.cfg.weights[name] == 0 for name in FOOT_REWARD_NAMES)
  if visual:
    grad = alg.wm.depth_encoder.frame_cnn[0].weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


@pytest.mark.parametrize('disabled', [0, 1])
def test_independent_loss_weights_gate_output_rows_but_share_hidden_features(mdp, disabled):
  weights = {name + '_loss_weight': float(i != disabled)
             for i, name in enumerate(FOOT_REWARD_NAMES)}
  alg, replay, _ = training_case(mdp, state_loss_weight=0.,
                                **weights)
  head = alg.wm.foot_head
  output = head[0][-1]
  row_before = output.weight[disabled].detach().clone()
  bias_before = output.bias[disabled].detach().clone()
  hidden_before = head[0][0].weight.detach().clone()
  alg.world_model_update(replay)
  # A fresh optimizer has no momentum for the unsupervised output row. The
  # shared hidden layer still learns from the other cost; neither prediction
  # should be assumed frozen merely because its own loss weight is zero.
  torch.testing.assert_close(output.weight[disabled], row_before, rtol=0, atol=0)
  torch.testing.assert_close(output.bias[disabled], bias_before, rtol=0, atol=0)
  assert output.weight.grad[disabled].eq(0).all()
  assert output.weight.grad[1-disabled].abs().sum() > 0
  assert not torch.equal(head[0][0].weight, hidden_before)


def test_replay_preserves_raw_terminal_costs_through_ring_wrap(mdp):
  m = mdp.motion
  _, transitions = filled_buffer(mdp, small_config())
  replay = ReplayBuffer(4, 2, torch.zeros(2, 33), m.clip_end, m.clip_id, 32,
                        foot_reward_names=FOOT_REWARD_NAMES)
  for t, tr in enumerate(transitions):
    tr['a'][:, 0] = t
    tr['foot_costs'] = torch.tensor([t + .25, t + .75]).expand(2, -1).clone()
    tr['terminated'][:] = t == 5
    replay.add(tr)
    tr['foot_costs'].zero_()  # Buffer must own pre-reset labels.
  w = replay.sample_windows(10, 4)
  torch.testing.assert_close(w['foot_costs'][..., 0], w['a'][..., 0] + .25)
  torch.testing.assert_close(w['foot_costs'][..., 1], w['a'][..., 0] + .75)
  assert w['mask'].tolist() == [[1, 1, 0, 0]] * 10
  assert w['foot_costs'][:, 1, 0].eq(5.25).all()
  with pytest.raises(KeyError, match='foot_costs'):
    replay.add({key: value for key, value in transitions[-1].items() if key != 'foot_costs'})


def test_imagined_costs_apply_reward_weights_and_dt_once(mdp):
  alg, replay, _ = training_case(mdp)
  start, _ = replay.sample_starts(2, 2)
  state = unflatten_state(start['s'])
  action = torch.zeros(2, 29)
  costs = torch.tensor([[.2, .8], [.7, 1.2]], requires_grad=True)
  original = mdp.reward(state, start['f'], action, action)
  reward, soft = alg.reward_term(state, start['f'], action, action, costs)
  torch.testing.assert_close(reward, original, rtol=0, atol=0)
  mdp.cfg.weights.update(feet_stumble=-2., feet_at_plane=-3.)
  reward, actual_soft = alg.reward_term(state, start['f'], action, action, costs)
  torch.testing.assert_close(reward, original + (-2*costs[:, 0]-3*costs[:, 1])*mdp.cfg.step_dt)
  torch.testing.assert_close(soft, actual_soft)
  reward.sum().backward()
  torch.testing.assert_close(costs.grad, torch.tensor([[-2., -3.]]).expand(2, -1)*mdp.cfg.step_dt)
  compiled = torch.compile(alg.reward_term, fullgraph=True, backend='aot_eager')
  try:
    compiled_reward, compiled_soft = compiled(state, start['f'], action, action, costs)
    torch.testing.assert_close(compiled_reward, reward)
    torch.testing.assert_close(compiled_soft, soft)
    gradient, = torch.autograd.grad(compiled_reward.sum(), (costs,))
    torch.testing.assert_close(gradient, costs.grad)
  finally:
    torch.compiler.reset()


@pytest.mark.parametrize('visual', [False, True])
def test_actor_cost_bptt_keeps_wm_frozen_and_short_motion_bootstrap(mdp, visual, monkeypatch):
  alg, replay, _ = training_case(mdp, visual, horizon=4, pi_batch=2)
  c = alg.cfg
  mdp.cfg.weights.update(feet_stumble=-2., feet_at_plane=-3.)
  # Isolate cost-head gradients; bootstrap is constant V=1, no analytic rewards.
  monkeypatch.setattr(mdp, 'reward', lambda s, f, a, prev: torch.zeros_like(f, dtype=torch.float32))
  monkeypatch.setattr(mdp, 'soft_termination', lambda s, f: torch.zeros_like(f, dtype=torch.float32))
  with torch.no_grad():
    for p in alg.critic.net.parameters():
      p.zero_()
    alg.critic.net[-1].bias.fill_(1.)
  start, history = replay.sample_starts(2, c.hist)
  start['f'] = torch.tensor([50, 55])  # Only 1 and 3 valid transitions.
  monkeypatch.setattr(replay, 'sample_starts', lambda *args: (start, history))
  predictions = []
  imagine_step = alg._imag_step
  def capture(*args):
    result = imagine_step(*args)
    result[-1].retain_grad()
    predictions.append(result[-1])
    return result
  alg._imag_step = capture
  before = copy.deepcopy(alg.wm.state_dict())
  actor_before = copy.deepcopy(alg.policy.dec.state_dict())
  result = alg.policy_update(replay)
  lengths = torch.tensor([1, 3])
  weights = torch.tensor([-2., -3.]) * mdp.cfg.step_dt
  objective = c.gamma ** lengths
  for t, costs in enumerate(predictions):
    valid = t < lengths
    objective += c.gamma**t * (costs.detach()*weights).sum(-1) * valid
    expected_grad = -c.gamma**t * weights[None] * valid[:, None] / c.horizon / 2
    torch.testing.assert_close(costs.grad, expected_grad)
  assert result['pi_loss'] == pytest.approx(float(-objective.mean()/c.horizon), abs=1e-7)
  assert result['pi_gradnorm'] > 0
  for key, value in before.items():
    torch.testing.assert_close(alg.wm.state_dict()[key], value, rtol=0, atol=0)
  assert any(not torch.equal(value, alg.policy.dec.state_dict()[key]) for key, value in actor_before.items())


@pytest.mark.parametrize('visual', [False, True])
def test_compiled_cost_rollout_matches_forward_and_parameter_gradients(mdp, visual):
  alg, replay, _ = training_case(mdp, visual)
  w = replay.sample_windows(2, alg.cfg.wm_window)
  compiled = torch.compile(alg._rollout, fullgraph=True, backend='aot_eager')
  # Constant fixture states normalize to zero; excite the action input so the
  # first trunk weight (not just its bias) has a nonzero derivative.
  w['a'] = torch.linspace(-.2, .3, w['a'].numel()).reshape_as(w['a'])
  try:
    args = (w['s'], w['a'], w['xi'], w.get('depth_history'))
    expected, actual = alg._rollout(*args)[-1], compiled(*args)[-1]
    torch.testing.assert_close(actual, expected)
    parameters = (alg.wm.net[0].weight, alg.wm.foot_head[0][0].weight)
    ref_grad = torch.autograd.grad(expected.sum(), parameters)
    got_grad = torch.autograd.grad(actual.sum(), parameters)
    for ref, got in zip(ref_grad, got_grad):
      torch.testing.assert_close(got, ref)
      assert torch.isfinite(got).all() and got.abs().sum() > 0
  finally:
    torch.compiler.reset()


def test_training_allocates_sensors_for_zero_weight_supervision():
  cfg = NS(decimation=4, scene=NS(sensors=(), terrain=NS(terrain_type='generator')), rewards={
    'feet_stumble': NS(weight=0., params={'sensor_name': 'stumble'}),
    'feet_at_plane': NS(weight=0., params={'sensor_name': 'sole', 'contact_sensor_name': 'support'})})
  sensor_configurator()(cfg, collect_foot_rewards=True)
  assert {sensor.name for sensor in cfg.scene.sensors} == {'stumble', 'sole', 'support'}
  cfg.scene.sensors = ()
  del cfg.rewards['feet_stumble']
  sensor_configurator()(cfg, collect_foot_rewards=True)
  assert {sensor.name for sensor in cfg.scene.sensors} == {'sole', 'support'}


class FootTensorEnv(TensorEnv):
  foot_reward_names = FOOT_REWARD_NAMES

  def step(self, action):
    result = super().step(action)
    self.last_transition['foot_costs'] = torch.stack((action[:, 0].square(), action[:, 1].square()), -1)
    for name in FOOT_REWARD_NAMES:
      self.last_step_metrics['reward'][name] = torch.zeros(2)
    return result


def test_cost_heads_train_resume_and_log_to_csv_and_wandb(mdp, tmp_path, sdk):
  mdp.cfg.weights.update(feet_stumble=0., feet_at_plane=0.)
  cfg = small_config(iterations=2, warmup_steps=4, wm_only_iters=1, log_every=1,
                     logger='wandb', wandb_mode='offline')
  runner = SuperMimicRunner(FootTensorEnv(mdp), cfg, tmp_path/'first')
  runner.learn()
  with (tmp_path/'first/train_log.csv').open() as file:
    rows = list(csv.DictReader(file))
  for row, logged in zip(rows, sdk.runs[0].rows):
    for name in FOOT_REWARD_NAMES:
      key = 'wm_' + name + '_loss'
      assert float(row[key]) == pytest.approx(logged['Loss/'+key])
      assert float(row['Reward/'+name]) == 0.
    assert float(row['wm_total_loss']) == pytest.approx(
      cfg.state_loss_weight * float(row['wm_loss'])
      + cfg.feet_stumble_loss_weight * float(row['wm_feet_stumble_loss'])
      + cfg.feet_at_plane_loss_weight * float(row['wm_feet_at_plane_loss']))
  saved = copy.deepcopy(runner.algorithm.wm.foot_head.state_dict())
  assert torch.load(tmp_path/'first/last.pt', weights_only=False)['format_version'] == 5
  cfg.iterations = 3
  resumed = SuperMimicRunner(FootTensorEnv(mdp), cfg, tmp_path/'resumed')
  resumed.load(tmp_path/'first/last.pt')
  assert resumed.buffer.size == 0 and resumed.iteration == 2
  for key, value in saved.items():
    torch.testing.assert_close(resumed.algorithm.wm.foot_head.state_dict()[key], value)
  resumed.learn()
  assert resumed.iteration == 3 and resumed.buffer.size >= cfg.warmup_steps


def test_old_checkpoint_is_playable_but_cannot_silently_resume_without_cost_heads(mdp, tmp_path):
  from supermimic.rl.evaluation import load_policy
  cfg = small_config()
  old = SuperMimicRunner(TensorEnv(mdp), cfg, tmp_path/'old')
  old.save(tmp_path/'old.pt')
  mdp.cfg.weights.update(feet_stumble=0., feet_at_plane=0.)
  new = SuperMimicRunner(FootTensorEnv(mdp), cfg, tmp_path/'new')
  with pytest.raises(ValueError, match='no WM foot-cost heads'):
    new.load(tmp_path/'old.pt')
  load_policy(tmp_path/'old.pt', new.env)


def test_independent_head_checkpoint_format_is_rejected_before_loading_weights(mdp, tmp_path):
  mdp.cfg.weights.update(feet_stumble=0., feet_at_plane=0.)
  runner = SuperMimicRunner(FootTensorEnv(mdp), small_config(), tmp_path/'run')
  runner.save(tmp_path/'checkpoint.pt')
  checkpoint = torch.load(tmp_path/'checkpoint.pt', weights_only=False)
  checkpoint['format_version'] = 4  # Version 4 declares two independent MLPs.
  torch.save(checkpoint, tmp_path/'checkpoint.pt')
  before = copy.deepcopy(runner.algorithm.wm.state_dict())
  with pytest.raises(ValueError, match='independent WM foot-cost MLPs'):
    runner.load(tmp_path/'checkpoint.pt')
  for key, value in before.items():
    torch.testing.assert_close(runner.algorithm.wm.state_dict()[key], value, rtol=0, atol=0)


def test_removed_cost_does_not_require_labels_or_create_head(mdp):
  mdp.cfg.weights['feet_at_plane'] = 0.
  alg = SuperMimicAlgorithm(mdp, small_config(), 'cpu')
  assert alg.foot_reward_names == ('feet_at_plane',)
  assert alg.wm.foot_reward_names == ('feet_at_plane',)
  assert alg.wm.foot_head[0][-1].out_features == 1


def test_nonzero_foot_rewards_are_supported_but_unknown_rewards_still_rejected(mdp):
  mdp.cfg.weights.update(feet_stumble=-.1, feet_at_plane=-.5)
  assert SuperMimicAlgorithm(mdp, small_config(), 'cpu').foot_reward_names == FOOT_REWARD_NAMES
  mdp.cfg.weights['unimplemented_reward'] = -1.
  with pytest.raises(ValueError, match='unimplemented_reward'):
    SuperMimicAlgorithm(mdp, small_config(), 'cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires NVIDIA GPU for BF16/Inductor')
def test_cuda_bf16_cost_heads_compile_and_backpropagate_to_action(monkeypatch):
  monkeypatch.setattr(networks, 'AMP_DTYPE', torch.bfloat16)
  wm = WorldModel(hidden=(24, 24), foot_reward_names=FOOT_REWARD_NAMES).cuda()
  zeros = torch.zeros(2, 3, device='cuda')
  state = (zeros, torch.eye(3, device='cuda').expand(2, -1, -1), zeros, zeros,
           torch.zeros(2, 29, device='cuda'), torch.zeros(2, 29, device='cuda'))
  action = torch.randn(2, 29, device='cuda', requires_grad=True)
  xi = torch.zeros(2, 33, device='cuda')
  compiled = torch.compile(wm.step_with_foot_costs, fullgraph=True)
  try:
    with frozen(wm):
      expected = wm.step_with_foot_costs(state, action, xi)[-1]
      actual = compiled(state, action, xi)[-1]
      assert actual.dtype == torch.float32
      torch.testing.assert_close(actual, expected, rtol=.02, atol=.002)
      ref, = torch.autograd.grad(expected.sum(), (action,))
      got, = torch.autograd.grad(actual.sum(), (action,))
      torch.testing.assert_close(got, ref, rtol=.05, atol=.002)
      assert torch.isfinite(got).all() and got.abs().sum() > 0
      assert all(p.grad is None for p in wm.parameters())
  finally:
    torch.compiler.reset()
