"""Numerical, compilation and work-count regressions for shared latent dynamics."""
import copy
from contextlib import nullcontext
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
  # Both still use their own endpoint latent/target for the bootstrap.
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


def test_imagination_encodes_once_and_never_decodes_even_during_critic_epochs(mdp):
  cfg = visual_config(wm_window=8, horizon=8, depth_frames=8, depth_skip=5,
                      depth_checkpoint=False, critic_epochs=2)
  alg = SuperMimicAlgorithm(mdp, cfg, 'cpu')
  buffer, _ = visual_buffer(mdp, cfg)
  frames=[];decoded=[]
  hook=alg.wm.depth_encoder.frame_cnn.register_forward_pre_hook(lambda m,a:frames.append(len(a[0])))
  decode_hook=alg.wm.depth_decoder.register_forward_pre_hook(lambda m,a:decoded.append(len(a[0])))
  try:
    alg.world_model_update(buffer, strict=True)
    # Initial differentiable window + dense no-grad real frame bank for targets.
    assert sum(frames)==cfg.wm_batch*(cfg.depth_frames+cfg.depth_history_length+cfg.wm_window)
    assert sum(decoded)==cfg.wm_batch*cfg.wm_window
    frames.clear();decoded.clear()
    alg.policy_update(buffer)
    assert sum(frames)==cfg.pi_batch*cfg.depth_frames
    assert not decoded
  finally:hook.remove();decode_hook.remove()


def test_imagination_step_and_critic_compile_fullgraph_with_latent_gradients(mdp):
  cfg = visual_config(depth_checkpoint=True, depth_frames=8, depth_skip=5,
                      horizon=3, depth_batch_size=1)
  alg = SuperMimicAlgorithm(mdp, cfg, 'cpu')
  buffer, _ = visual_buffer(mdp, cfg)
  start, history = buffer.sample_starts(cfg.pi_batch, cfg.hist)
  state = unflatten_state(start['s'])
  f, xi = start['f'], start['xi']
  obs = mdp.obs(unflatten_state(history['s']), history['a_prev'], history['xi'], history['f'])
  goal = mdp.goal(state, f, cfg.goal_frames)
  h=torch.randn(cfg.pi_batch,cfg.depth_latent_dim,requires_grad=True)
  compiled_step = torch.compile(alg._imagine_step, backend='aot_eager', fullgraph=True)
  compiled_value = torch.compile(alg.critic_forward, backend='aot_eager', fullgraph=True)
  compiled_error = torch.compile(alg.critic_error, backend='aot_eager', fullgraph=True)
  try:
    with frozen(alg.wm, alg.critic):
      rng = torch.get_rng_state().clone()
      expected = alg._imagine_step(state, obs, f, xi, goal, h)
      torch.set_rng_state(rng)
      got = compiled_step(state, obs, f, xi, goal, h)
      for a, b in zip((*got[0], *got[1:]), (*expected[0], *expected[1:])):
        torch.testing.assert_close(a, b)
      expected_grad, = torch.autograd.grad(expected[-1].square().mean(), h)
      actual_grad, = torch.autograd.grad(got[-1].square().mean(), h)
      torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-6)
      assert actual_grad.abs().sum() > 0
      x = alg.critic_input(state, f, normalize_xi(xi), goal, start['a_prev'])
      a = h.detach().clone().requires_grad_(); b = a.detach().clone().requires_grad_()
      expected_value = alg.critic_forward(x, a); got_value = compiled_value(x, b)
      torch.testing.assert_close(got_value, expected_value)
      torch.testing.assert_close(torch.autograd.grad(got_value.sum(), b)[0],
                                 torch.autograd.grad(expected_value.sum(), a)[0], rtol=1e-4, atol=1e-6)
    actual_loss = compiled_error(compiled_value(x, h.detach()), torch.ones(cfg.pi_batch))
    expected_loss = alg.critic_error(alg.critic_forward(x, h.detach()), torch.ones(cfg.pi_batch))
    ag = torch.autograd.grad(actual_loss, tuple(alg.v_parameters))
    eg = torch.autograd.grad(expected_loss, tuple(alg.v_parameters))
    for a, b in zip(ag, eg): torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
  finally:
    torch.compiler.reset()


def test_batched_decoder_matches_stepwise_outputs_and_gradients(mdp):
  cfg=visual_config(depth_frames=2,depth_batch_size=1)
  alg=SuperMimicAlgorithm(mdp,cfg,'cpu')
  h=torch.randn(3,5,cfg.depth_latent_dim,requires_grad=True)
  other=h.detach().clone().requires_grad_()
  expected=torch.stack([alg.wm.depth_decoder(h[:,t]) for t in range(5)],1)
  actual=alg.wm.decode_depth(other)
  torch.testing.assert_close(actual,expected)
  eg=torch.autograd.grad(expected.square().mean(),(h,*alg.wm.depth_decoder.parameters()))
  ag=torch.autograd.grad(actual.square().mean(),(other,*alg.wm.depth_decoder.parameters()))
  for a,b in zip(ag,eg):torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-6)


def legacy_critic_forward(alg, x, h):
  """Reference execution: one time step at a time, then image-sized chunks."""
  if x.ndim == 3:
    return torch.stack([legacy_critic_forward(alg, row, latent)
                        for row, latent in zip(x, h)])
  size = alg.cfg.depth_batch_size
  return torch.cat([alg.critic_forward(x[i:i+size], h[i:i+size])
                    for i in range(0, len(x), size)])


@pytest.mark.parametrize('short_clips', [False, True])
def test_batched_critic_matches_chunked_updates_and_preserves_gradient_boundaries(mdp, short_clips):
  cfg = visual_config(horizon=4, pi_batch=3, critic_epochs=2,
                      critic_minibatches=2, depth_batch_size=1)
  actual = SuperMimicAlgorithm(mdp, cfg, 'cpu')
  expected = SuperMimicAlgorithm(mdp, cfg, 'cpu')
  expected.load_state_dict(copy.deepcopy(actual.state_dict()))
  expected._critic_forward = lambda x, h=None: legacy_critic_forward(expected, x, h)
  buffer, _ = visual_buffer(mdp, cfg)
  start, history = buffer.sample_starts(cfg.pi_batch, cfg.hist)
  start['f'] = torch.tensor([50, 56, 10] if short_clips else [10, 20, 30])
  start['depth_history'] = torch.randint(256, start['depth_history'].shape, dtype=torch.uint8)

  class Fixed:
    def sample_starts(self, *args): return start, history

  # Keep stale WM perception gradients: batching must not include them in
  # Actor/Critic clipping or assign these parameters to another optimizer.
  for alg in (actual, expected):
    for p in alg.wm.depth_encoder.parameters():
      p.grad = torch.full_like(p, 100.)
  wm_before = copy.deepcopy(actual.wm.state_dict())
  calls = []

  def capture(module, args, kwargs):
    x, h = args[0], kwargs['depth_latent']
    calls.append((tuple(x.shape[:-1]), tuple(h.shape[:-1]),
                  torch.is_grad_enabled(), x.requires_grad, h.requires_grad))

  hook = actual.critic.register_forward_pre_hook(capture, with_kwargs=True)
  valid_steps = 7 if short_clips else cfg.pi_batch * cfg.horizon
  minibatch_sizes = [len(ids) for ids in torch.arange(valid_steps).tensor_split(cfg.critic_minibatches)]
  try:
    # Repeating the update also catches stale values reused after optimizer steps.
    for _ in range(2):
      calls.clear()
      rng = torch.get_rng_state().clone()
      expected_metrics = expected.policy_update(Fixed())
      expected_rng = torch.get_rng_state().clone()
      torch.set_rng_state(rng)
      with patch.object(actual.opt_pi, 'step', wraps=actual.opt_pi.step) as pi_step, \
           patch.object(actual.opt_v, 'step', wraps=actual.opt_v.step) as v_step:
        actual_metrics = actual.policy_update(Fixed())
      assert pi_step.call_count == 1
      assert v_step.call_count == cfg.critic_epochs * cfg.critic_minibatches
      assert torch.equal(torch.get_rng_state(), expected_rng)
      for key, value in actual_metrics.items():
        torch.testing.assert_close(value, expected_metrics[key], rtol=2e-4, atol=2e-6)
      assert_learners_close(actual, expected)
      # One differentiable endpoint batch, one no-grad trajectory batch,
      # then one forward per original minibatch (including uneven sizes).
      assert calls == [
        ((3,), (3,), True, True, True),
        ((5, 3), (5, 3), False, False, False),
        *[((n,), (n,), True, False, False)
          for n in minibatch_sizes * cfg.critic_epochs],
      ]
      for key, value in wm_before.items():
        torch.testing.assert_close(actual.wm.state_dict()[key], value, rtol=0, atol=0)
      for p in actual.wm.depth_encoder.parameters():
        torch.testing.assert_close(p.grad, torch.full_like(p, 100.), rtol=0, atol=0)
  finally:
    hook.remove()


@pytest.mark.parametrize('device', [
  'cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='Requires NVIDIA GPU for Inductor')),
])
def test_batched_critic_compiles_with_time_axis_and_variable_minibatches(mdp, device):
  cfg = visual_config(depth_batch_size=1)
  alg = SuperMimicAlgorithm(mdp, cfg, device)
  backend = 'inductor' if device == 'cuda' else 'aot_eager'
  value = torch.compile(alg.critic_forward, backend=backend, fullgraph=True, dynamic=True)
  error = torch.compile(alg.critic_error, backend=backend, fullgraph=True, dynamic=True)
  try:
    for frozen_parameters in (True, False):
      with frozen(alg.critic) if frozen_parameters else nullcontext():
        # Both a time-major trajectory and differently sized minibatches.
        for shape in ((7,), (3, 2), (5,)):
          x = torch.randn(*shape, alg.critic.norm.mean.numel(), device=device, requires_grad=True)
          h = torch.randn(*shape, cfg.depth_latent_dim, device=device, requires_grad=True)
          other_x = x.detach().clone().requires_grad_()
          other_h = h.detach().clone().requires_grad_()
          target = torch.randn(*shape, device=device)
          expected = legacy_critic_forward(alg, x, h)
          actual = value(other_x, other_h)
          torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
          expected_loss = alg.critic_error(expected, target)
          actual_loss = error(actual, target)
          parameters = () if frozen_parameters else tuple(alg.v_parameters)
          eg = torch.autograd.grad(expected_loss, (x, h, *parameters))
          ag = torch.autograd.grad(actual_loss, (other_x, other_h, *parameters))
          assert ag[0].abs().sum() > 0 and ag[1].abs().sum() > 0
          for a, b in zip(ag, eg):
            torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-5)
    assert all(p.grad is None for p in alg.wm.depth_encoder.parameters())
  finally:
    torch.compiler.reset()
