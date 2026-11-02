"""Composition without edits to Runner: new terms, heads, and replaceable networks."""
import copy
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace as NS

import pytest
import torch

from supermimic.config.g1.rl_cfg import SuperMimicConfig, unitree_g1_terrain_runner_cfg
from supermimic.core.components import Prediction, analytic_reward, analytic_failure, reward_component, bind_analytic_manager_terms, prediction_bindings
from supermimic.core.functional import MdpConfig
from supermimic.mdp import rewards
from supermimic.core.state import unflatten_state
from supermimic.rl.algorithm import SuperMimicAlgorithm
from supermimic.networks import Policy, Critic, WorldModel
from supermimic.rl.buffer import ReplayBuffer
from supermimic.playback.evaluation import load_policy, read_checkpoint_config
from supermimic.rl.runner import SuperMimicRunner
from test_algorithm import small_config, filled_buffer
from test_visual_algorithm import visual_config, visual_buffer
from test_train_resume_eval import TensorEnv
from test_visual_train_resume import VisualTensorEnv


@dataclass
class RewardCfg:
  func: object
  weight: float = 1.0
  params: dict = field(default_factory=dict)


@dataclass
class FailureCfg:
  func: object
  params: dict = field(default_factory=dict)
  time_out: bool = False


@analytic_reward
def quiet_joints(ctx, scale=1.0):
  return torch.exp(-scale * ctx.state[5].square().mean(-1))


@analytic_failure(soft_width=.05)
def joint_speed_error(ctx):
  return ctx.state[5].abs().amax(-1)


@reward_component(prediction=Prediction(loss_weight=.3, nonnegative=False))
def signed_sensor_reward(env, gain=1.0):
  return env.sensor_value * gain


def legacy_rewards():
  cfg = MdpConfig()
  return {name: RewardCfg(fn, cfg.weights[name], {} if attr is None else {'std': getattr(cfg, attr)})
          for name, (fn, _, attr) in rewards.LEGACY_REWARDS.items()}


def test_new_analytic_component_and_renamed_existing_term_match_measured_and_compiled(mdp):
  config = NS(rewards={'renamed_root': RewardCfg(rewards.motion_global_anchor_position_error_exp,
                                               .7, {'command_name': 'motion', 'std': .6}),
                       'quiet': RewardCfg(quiet_joints, .3, {'scale': 2.})}, terminations={})
  bind_analytic_manager_terms(config)
  mdp.cfg = MdpConfig.from_env_cfg(config.rewards, config.terminations, .02)
  mdp.validate_imagination_rewards()
  f = torch.tensor([1, 4])
  p, r, v, w = mdp.reference_root(f)
  p = (p + .1).requires_grad_()
  qd = torch.full((2, 29), .4, requires_grad=True)
  state = (p, r, v, w, mdp.motion.joint_pos[f], qd)
  a = torch.zeros(2, 29)
  raw = mdp.reward_terms(state, f, a, a)
  env = NS(shared_reward_terms=lambda: raw)
  for name, cfg in config.rewards.items():
    torch.testing.assert_close(cfg.func(env, **cfg.params), raw[name], rtol=0, atol=0)
  expected = (.7 * torch.exp(-torch.full((2,), .03) / .6**2)
              + .3 * torch.exp(-2 * qd.square().mean(-1))) * .02
  torch.testing.assert_close(mdp.reward(state, f, a, a), expected)
  compiled = torch.compile(mdp.reward, backend='aot_eager', fullgraph=True)
  try:
    actual = compiled(state, f, a, a)
    torch.testing.assert_close(actual, expected)
    gradients = torch.autograd.grad(actual.sum(), (p, qd))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
  finally:
    torch.compiler.reset()


def test_new_failure_uses_same_error_and_can_be_removed_without_algorithm_changes(mdp):
  cfg = NS(rewards={}, terminations={'joint_velocity': FailureCfg(joint_speed_error, {'threshold': .7})})
  bind_analytic_manager_terms(cfg)
  mdp.cfg = MdpConfig.from_env_cfg(cfg.rewards, cfg.terminations, .02)
  assert not mdp.requires_height_offset
  f = torch.tensor([1, 2])
  qd = torch.tensor([.6, .8])[:, None].expand(-1, 29).clone().requires_grad_()
  state = (*mdp.reference_root(f), mdp.motion.joint_pos[f], qd)
  env = NS(shared_failure_terms=lambda: mdp.failure_terms(state, f))
  term = cfg.terminations['joint_velocity']
  assert term.func(env, **term.params).tolist() == [False, True]
  expected = torch.sigmoid((qd.abs().amax(-1) - .7) / .05)
  actual = mdp.soft_termination(state, f)
  torch.testing.assert_close(actual, expected)
  actual.sum().backward()
  assert qd.grad.abs().sum() > 0
  cfg.terminations.clear()
  mdp.cfg = MdpConfig.from_env_cfg({}, cfg.terminations, .02)
  assert not mdp.requires_height_offset
  assert not mdp.termination(state, f).any()
  assert mdp.soft_termination(state, f).eq(0).all()


def test_unknown_failure_and_unknown_replacement_do_not_silently_use_old_formula(mdp):
  with pytest.raises(ValueError, match='differentiable SuperMimic component'):
    MdpConfig.from_env_cfg({}, {'unsupported': FailureCfg(lambda env: True, {'threshold': 1})}, .02)
  mdp.cfg = MdpConfig.from_env_cfg({'motion_global_root_pos': RewardCfg(lambda env: 1)}, {}, .02)
  with pytest.raises(ValueError, match='No imagined reward'):
    mdp.validate_imagination_rewards()


def test_ambiguous_analytic_and_predicted_reward_cannot_double_count():
  with pytest.raises(ValueError, match='exactly one imagined reward source'):
    reward_component(tensor=lambda ctx: 1.0, prediction=Prediction())(lambda env: 1.0)


@pytest.mark.parametrize('visual', [False, True])
def test_new_learned_reward_builds_labels_head_loss_and_imagination_without_name_dispatch(mdp, visual):
  terms = legacy_rewards()
  terms['new_signed_reward'] = RewardCfg(signed_sensor_reward, -.6, {'gain': 2.})
  mdp.cfg = MdpConfig.from_env_cfg(terms, {}, .02)
  cfg = (visual_config if visual else small_config)(reward_loss_weights={'new_signed_reward': .8})
  alg = SuperMimicAlgorithm(mdp, cfg, 'cpu')
  assert alg.reward_schema.names == ('new_signed_reward',)
  assert alg.reward_loss_weights == {'new_signed_reward': .8}
  assert alg.reward_schema.log_fields == ('wm_new_signed_reward_loss',)
  assert alg.wm.foot_head[0][-1].out_features == 1
  with torch.no_grad():
    alg.wm.foot_head[0][-1].weight.zero_()
    alg.wm.foot_head[0][-1].bias.fill_(-.2)
  assert alg.wm.foot_head(torch.zeros(2, cfg.wm_hidden[-1])).lt(0).all()
  # This component still supplies raw labels with task weight zero.
  terms['new_signed_reward'].weight = 0
  bindings = prediction_bindings(terms)
  assert tuple(name for name, _, _ in bindings) == alg.reward_schema.names
  fake_env = NS(sensor_value=torch.tensor([.1, -.2]))
  labels = torch.stack([term.func(fake_env, **term.params) for _, term, _ in bindings], -1)
  torch.testing.assert_close(labels[:, 0], torch.tensor([.2, -.4]))
  _, transitions = (visual_buffer if visual else filled_buffer)(mdp, cfg)
  m = mdp.motion
  buffer = ReplayBuffer(cfg.buffer_steps, 2, torch.zeros(2, 33), m.clip_end, m.clip_id,
                        cfg.horizon, cfg.depth_history_length if visual else 0,
                        reward_schema=alg.reward_schema)
  for tr in transitions:
    tr[alg.reward_schema.field] = labels
    buffer.add(tr)
    alg.observe(tr, buffer.xi)
  metrics = alg.world_model_update(buffer, strict=True)
  assert metrics['wm_new_signed_reward_loss'] > 0
  expected = cfg.state_loss_weight * metrics['wm_loss'] + .8 * metrics['wm_new_signed_reward_loss']
  if visual:
    expected += cfg.latent_loss_weight * metrics['wm_latent_loss'] + cfg.depth_loss_weight * metrics['wm_depth_loss']
  assert metrics['wm_total_loss'] == pytest.approx(expected, rel=1e-5)
  row = transitions[0]
  state = unflatten_state(row['s1'])
  predicted = labels.clone().requires_grad_()
  analytic = mdp.reward(state, row['f'], row['a'], row['a_prev'])
  actual, _ = alg.reward_term(state, row['f'], row['a'], row['a_prev'], predicted)
  torch.testing.assert_close(actual, analytic - .6 * predicted[:, 0] * .02)
  actual.sum().backward()
  torch.testing.assert_close(predicted.grad, torch.full_like(predicted, -.012))
  wm_before = copy.deepcopy(alg.wm.state_dict())
  assert alg.policy_update(buffer)['policy_skipped'] == 0
  for key, value in wm_before.items():
    torch.testing.assert_close(alg.wm.state_dict()[key], value, rtol=0, atol=0)


def test_explicit_sensor_keeps_zero_weight_supervision_and_refresh_declaration():
  @reward_component(prediction=Prediction(), sense_before_reward=True)
  def arbitrary_sensor(env, sensor):
    return env.scene[sensor].force
  sensor = NS(name='impact', force=torch.tensor([.25, .75]))
  cfg = NS(scene=NS(sensors=(sensor,)), rewards={
    'something_new': RewardCfg(arbitrary_sensor, 0., {'sensor': 'impact'})})
  bindings = prediction_bindings(cfg.rewards)
  assert len(bindings) == 1 and bindings[0][0] == 'something_new'
  _, term, component = bindings[0]
  assert component.sense_before_reward and component.prediction.loss_weight == 1.
  torch.testing.assert_close(term.func(NS(scene={'impact': sensor}), **term.params), sensor.force)
  cfg.rewards.clear()
  assert prediction_bindings(cfg.rewards) == ()
  assert cfg.scene.sensors == (sensor,)


class AlternativePolicy(Policy):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.alternative_architecture = True


class AlternativeCritic(Critic):
  pass


class AlternativeWorldModel(WorldModel):
  pass


@pytest.mark.parametrize('visual', [False, True])
def test_factory_selection_is_shared_with_checkpoint_play_and_resume(mdp, tmp_path, visual):
  cfg = (visual_config if visual else small_config)(
    actor_class=f'{__name__}:AlternativePolicy', actor_kwargs={'z_dim': 8},
    critic_class=f'{__name__}:AlternativeCritic',
    world_model_class=f'{__name__}:AlternativeWorldModel',
    iterations=1, warmup_steps=4, wm_only_iters=0, log_every=1, ckpt_every=1)
  env_type = VisualTensorEnv if visual else TensorEnv
  runner = SuperMimicRunner(env_type(mdp), cfg, tmp_path/'train')
  assert isinstance(runner.algorithm.policy, AlternativePolicy)
  assert isinstance(runner.algorithm.critic, AlternativeCritic)
  assert isinstance(runner.algorithm.wm, AlternativeWorldModel)
  assert runner.algorithm.policy.enc[-1].out_features == 8
  runner.learn()
  policy, loaded, ck = load_policy(tmp_path/'train/last.pt', env_type(mdp))
  assert isinstance(policy, AlternativePolicy) and loaded.actor_kwargs == {'z_dim': 8}
  assert ck['network_spec']['classes']['actor_class'] == cfg.actor_class
  resumed = SuperMimicRunner(env_type(mdp), cfg, tmp_path/'resume')
  resumed.load(tmp_path/'train/last.pt')
  assert resumed.iteration == 1 and resumed.buffer.size == 0
  if visual:
    alg = resumed.algorithm
    encoder = alg.wm.depth_encoder
    assert alg.policy.depth_encoder is encoder and alg.critic.depth_encoder is encoder
    shared = {id(p) for p in encoder.parameters()}
    assert not shared.intersection(id(p) for p in alg.pi_parameters + alg.v_parameters)


def test_base_task_retains_legacy_rewards_and_no_vision():
  from supermimic.tools.registry import load_runner_cfg
  cfg = load_runner_cfg('Unitree-G1-SuperMimic')
  assert cfg.task_id == 'Unitree-G1-SuperMimic' and not cfg.depth_enabled
  assert unitree_g1_terrain_runner_cfg().depth_enabled
  from test_env_cfg_layers import load_config_factories
  env_cfg = load_config_factories()['unitree_g1_env_cfg']()
  assert (env_cfg.sim.nconmax, env_cfg.sim.njmax) == (35, 250)
  tensor = MdpConfig.from_env_cfg(env_cfg.rewards, {}, .02)
  assert tensor.weights == MdpConfig().weights
  assert not any(b.component.prediction for b in tensor.resolved_rewards())


@pytest.mark.parametrize('task', ['Unitree-G1-SuperMimic', 'Unitree-G1-SuperMimic-Terrain'])
def test_environment_factory_consumes_task_before_simulator_constructor(monkeypatch, task):
  import sys
  from types import ModuleType
  from supermimic.config import make_env
  captured = {}
  cmd = NS()
  cfg = NS(scene=NS(), commands={'motion': cmd})
  config_module = ModuleType('supermimic.config.g1.env_cfgs')
  def factory(**kwargs):
    captured.update(kwargs)
    return cfg
  def chosen_factory(kind):
    def create(**kwargs):
      captured['task'] = kind
      return factory(**kwargs)
    return create
  config_module.unitree_g1_env_cfg = chosen_factory('Unitree-G1-SuperMimic')
  config_module.unitree_g1_terrain_env_cfg = chosen_factory('Unitree-G1-SuperMimic-Terrain')
  env_module = ModuleType('supermimic.env')
  # A closed signature catches a task accidentally forwarded through **kwargs
  # into mjlab's ManagerBasedRlEnv constructor.
  def create(cfg, *, device, render_mode, collect_foot_rewards,
             randomize_terrain_on_reset):
    assert collect_foot_rewards and device == 'cpu'
    return cfg
  env_module.SuperMimicEnv = create
  for module in (config_module, env_module):
    monkeypatch.setitem(sys.modules, module.__name__, module)
  assert make_env('motion.npz', device='cpu', num_envs=4,
                  task_id=task) is cfg
  assert captured['task'] == task
  assert 'depth_enabled' not in captured and 'depth_camera' not in captured
  assert cmd.motion_file == 'motion.npz' and cfg.scene.num_envs == 4


@pytest.mark.parametrize('factory, visual', [
  ('unitree_g1_runner_cfg', False), ('unitree_g1_terrain_runner_cfg', True),
])
def test_runner_factory_preserves_shared_user_config(monkeypatch, factory, visual):
  from supermimic.config.g1 import rl_cfg
  class_name = 'TerrainSuperMimicConfig' if visual else 'SuperMimicConfig'
  configured = getattr(rl_cfg, class_name)(actor_kwargs={'z_dim': 8}, lr=3e-4, bin_ema=.2)
  monkeypatch.setattr(rl_cfg, class_name, lambda: configured)
  result = getattr(rl_cfg, factory)()
  assert result is configured and result.actor_kwargs == {'z_dim': 8}
  assert result.lr == 3e-4 and result.bin_ema == .2
  assert result.depth_enabled is visual
  assert result.task_id == ('Unitree-G1-SuperMimic-Terrain' if visual else 'Unitree-G1-SuperMimic')


def test_base_task_checkpoint_resumes_and_evaluates_without_depth_or_cost_heads(mdp, tmp_path):
  cfg = small_config(task_id='Unitree-G1-SuperMimic', iterations=1, warmup_steps=4, wm_only_iters=0)
  runner = SuperMimicRunner(TensorEnv(mdp), cfg, tmp_path/'run')
  runner.save(tmp_path/'main.pt')
  saved = read_checkpoint_config(tmp_path/'main.pt')
  assert saved.task_id == 'Unitree-G1-SuperMimic' and not saved.depth_enabled
  resumed = SuperMimicRunner(TensorEnv(mdp), saved, tmp_path/'resumed')
  resumed.load(tmp_path/'main.pt')
  assert resumed.algorithm.wm.foot_head is None and resumed.controller.depth_history is None
  resumed.learn()
  load_policy(tmp_path/'resumed/last.pt', TensorEnv(mdp))


@pytest.mark.parametrize('task', ['Unitree-G1-SuperMimic', 'Unitree-G1-SuperMimic-Terrain'])
def test_nonvisual_checkpoint_keeps_explicit_task_id(tmp_path, task):
  from supermimic.tools.registry import load_runner_cfg
  cfg = load_runner_cfg(task)
  cfg.depth_enabled = False
  path = tmp_path/'config.pt'
  torch.save({'config': asdict(cfg)}, path)
  saved = read_checkpoint_config(path)
  assert saved.task_id == task and not saved.depth_enabled and type(saved) is type(cfg)
