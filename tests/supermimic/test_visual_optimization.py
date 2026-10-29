"""Numerical and work-count regressions against uncached nn.GRU perception."""
import copy
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from supermimic.perception.preprocessing import depth_float
from supermimic.rl import networks
from supermimic.rl.algorithm import SuperMimicAlgorithm, frozen
from supermimic.rl.depth_encoder import DepthEncoder
from supermimic.mdp.state import normalize_xi, unflatten_state
from test_visual_algorithm import visual_config, visual_buffer


class UncachedEncoder(DepthEncoder):
  """Original CNN+nn.GRU forward, behind the new feature-window interface.

  Here a 'cached feature' is just the raw image. Every aggregation therefore
  recomputes all frame CNNs, exactly as the pre-optimization learner did.
  """
  def __init__(self, cfg):
    super().__init__(cfg.depth_latent_dim, cfg.depth_frame_dim, cfg.depth_checkpoint,
                     cfg.depth_batch_size, cfg.depth_frames)
    self.gru = nn.GRU(cfg.depth_frame_dim, cfg.depth_latent_dim, batch_first=True)

  def encode_frames(self, depth):
    return depth_float(depth)

  def _legacy_encode(self, depth):
    leading = depth.shape[:-3]
    frames = depth.shape[-3]
    x = self.frame_projection(self.frame_cnn(depth.reshape(-1, 1, 18, 32)))
    with torch.backends.cudnn.flags(enabled=False):
      _, hidden = self.gru(x.reshape(-1, frames, x.shape[-1]))
    return hidden[-1].reshape(*leading, -1)

  def aggregate(self, depth):
    leading = depth.shape[:-3]
    rows = depth.reshape(-1, *depth.shape[-3:])
    outputs = []
    for chunk in rows.split(self.batch_size):
      if self.checkpoint_activations and torch.is_grad_enabled():
        outputs.append(checkpoint(self._legacy_encode, chunk, use_reentrant=False))
      else:
        outputs.append(self._legacy_encode(chunk))
    return torch.cat(outputs).reshape(*leading, -1)


def assert_learners_close(actual, expected):
  for name in ('wm', 'policy', 'critic'):
    model, reference = getattr(actual, name), getattr(expected, name)
    assert model.state_dict().keys() == reference.state_dict().keys()
    for key, value in model.state_dict().items():
      torch.testing.assert_close(value, reference.state_dict()[key], rtol=2e-4, atol=2e-6)
    for p, q in zip(model.parameters(), reference.parameters()):
      if q.grad is None:
        assert p.grad is None
      else:
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-6)
  for name in ('opt_wm', 'opt_pi', 'opt_v'):
    actual_state, expected_state = getattr(actual, name).state_dict(), getattr(expected, name).state_dict()
    assert actual_state['param_groups'] == expected_state['param_groups']
    for index, state in actual_state['state'].items():
      for key, value in state.items():
        torch.testing.assert_close(value, expected_state['state'][index][key], rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize('recompute', [False, True])
def test_cached_training_matches_uncached_losses_gradients_updates_and_rng(mdp, recompute):
  torch.set_num_threads(2)
  cfg = visual_config(wm_window=8, horizon=8, depth_frames=8, depth_skip=5,
                      depth_checkpoint=recompute, critic_epochs=2, depth_batch_size=1)
  actual = SuperMimicAlgorithm(mdp, cfg, 'cpu')
  with patch.object(networks, 'make_depth_encoder', UncachedEncoder):
    expected = SuperMimicAlgorithm(mdp, cfg, 'cpu')
  expected.load_state_dict(copy.deepcopy(actual.state_dict()))
  buffer, transitions = visual_buffer(mdp, cfg)
  for tr in transitions:
    actual.observe(tr, buffer.xi); expected.observe(tr, buffer.xi)
  window = buffer.sample_windows(cfg.wm_batch, cfg.wm_window, strict=True)
  start, history = buffer.sample_starts(cfg.pi_batch, cfg.hist)
  # One trajectory ends immediately, the other has all eight valid steps.
  # Both still use their own endpoint images/target for the bootstrap.
  start['f'] = torch.tensor([50, 10])
  # Nonidentical initial frames expose stride/order mistakes hidden by padding.
  start['depth_history'] = torch.randint(256, start['depth_history'].shape, dtype=torch.uint8)
  window['depth_history'] = torch.randint(256, window['depth_history'].shape, dtype=torch.uint8)

  class Fixed:
    def sample_windows(self, *args, **kw): return window
    def sample_starts(self, *args): return start, history

  # A second iteration also checks that no cache survives a parameter update.
  for _ in range(2):
    for method in ('world_model_update', 'policy_update'):
      rng = torch.get_rng_state().clone()
      expected_metrics = getattr(expected, method)(Fixed())
      expected_rng = torch.get_rng_state().clone()
      torch.set_rng_state(rng)
      actual_metrics = getattr(actual, method)(Fixed())
      assert torch.equal(torch.get_rng_state(), expected_rng)
      assert actual_metrics.keys() == expected_metrics.keys()
      for key, value in actual_metrics.items():
        torch.testing.assert_close(value, expected_metrics[key], rtol=2e-4, atol=2e-6)
      assert_learners_close(actual, expected)


def test_frame_cnn_work_reuses_history_and_invalidates_critic_cache(mdp):
  cfg = visual_config(wm_window=8, horizon=8, depth_frames=8, depth_skip=5,
                      depth_checkpoint=False, critic_epochs=2)
  alg = SuperMimicAlgorithm(mdp, cfg, 'cpu')
  buffer, _ = visual_buffer(mdp, cfg)
  counts = {name: 0 for name in ('wm', 'policy', 'critic')}
  hooks = []
  for name in counts:
    def count(module, args, name=name): counts[name] += len(args[0])
    hooks.append(getattr(alg, name).depth_encoder.frame_cnn.register_forward_pre_hook(count))
  try:
    alg.world_model_update(buffer, strict=True)
    assert counts['wm'] == cfg.wm_batch * (cfg.depth_history_length + cfg.wm_window - 1)
    assert counts['policy'] == counts['critic'] == 0
    start, history = buffer.sample_starts(cfg.pi_batch, cfg.hist)
    start['f'] = torch.tensor([50, 10])
    class Fixed:
      def sample_starts(self, *args): return start, history
    counts.update(dict.fromkeys(counts, 0))
    alg.policy_update(Fixed())
    for name in ('policy', 'wm'):
      assert counts[name] == cfg.pi_batch * (cfg.depth_history_length + cfg.horizon - 1)
    # Endpoint bootstrap; once-only target frame bank; fresh histories for EVERY
    # critic minibatch in every epoch (weights change after each minibatch).
    expected = (cfg.pi_batch * cfg.depth_frames
                + cfg.pi_batch * (cfg.depth_history_length + cfg.horizon)
                + cfg.critic_epochs * (1 + cfg.horizon) * cfg.depth_frames)
    assert counts['critic'] == expected
  finally:
    for hook in hooks: hook.remove()


def test_imagination_step_and_critic_compile_fullgraph_with_gradients(mdp):
  cfg = visual_config(depth_checkpoint=True, depth_frames=8, depth_skip=5,
                      horizon=3, depth_batch_size=1)
  alg = SuperMimicAlgorithm(mdp, cfg, 'cpu')
  buffer, _ = visual_buffer(mdp, cfg)
  start, history = buffer.sample_starts(cfg.pi_batch, cfg.hist)
  state = unflatten_state(start['s'])
  f, xi = start['f'], start['xi']
  obs = mdp.obs(unflatten_state(history['s']), history['a_prev'], history['xi'], history['f'])
  goal = mdp.goal(state, f, cfg.goal_frames)
  image = torch.rand(cfg.pi_batch, cfg.depth_frames, 18, 32, requires_grad=True)
  compiled_step = torch.compile(alg._imagine_step, backend='aot_eager', fullgraph=True)
  compiled_value = torch.compile(alg.critic_forward, backend='aot_eager', fullgraph=True)
  compiled_error = torch.compile(alg.critic_error, backend='aot_eager', fullgraph=True)
  try:
    with frozen(alg.wm, alg.critic):
      pf = alg.policy.depth_encoder.encode_frames(image)
      wf = alg.wm.depth_encoder.encode_frames(image)
      rng = torch.get_rng_state().clone()
      expected = alg._imagine_step(state, obs, f, xi, goal,
                                  policy_depth_features=pf, wm_depth_features=wf)
      torch.set_rng_state(rng)
      got = compiled_step(state, obs, f, xi, goal,
                          policy_depth_features=pf, wm_depth_features=wf)
      for a, b in zip((*got[0], *got[1:]), (*expected[0], *expected[1:])):
        torch.testing.assert_close(a, b)
      expected_grad, = torch.autograd.grad(expected[-1].square().mean(), image, retain_graph=True)
      actual_grad, = torch.autograd.grad(got[-1].square().mean(), image)
      torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-6)
      assert actual_grad.abs().sum() > 0
      x = alg.critic_input(state, f, normalize_xi(xi), goal, start['a_prev'])
      a = image.detach().clone().requires_grad_(); b = a.detach().clone().requires_grad_()
      expected_value = alg.critic_forward(x, a); got_value = compiled_value(x, b)
      torch.testing.assert_close(got_value, expected_value)
      torch.testing.assert_close(torch.autograd.grad(got_value.sum(), b)[0],
                                 torch.autograd.grad(expected_value.sum(), a)[0], rtol=1e-4, atol=1e-6)
    # Trainable critic, compiled forward AND reduction, including parameter grads.
    actual_loss = compiled_error(compiled_value(x, image.detach()), torch.ones(cfg.pi_batch))
    expected_loss = alg.critic_error(alg.critic_forward(x, image.detach()), torch.ones(cfg.pi_batch))
    torch.testing.assert_close(actual_loss, expected_loss)
    ag = torch.autograd.grad(actual_loss, tuple(alg.critic.parameters()))
    eg = torch.autograd.grad(expected_loss, tuple(alg.critic.parameters()))
    for a, b in zip(ag, eg): torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
  finally:
    torch.compiler.reset()
