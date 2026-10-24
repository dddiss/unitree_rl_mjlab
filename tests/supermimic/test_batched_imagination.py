"""Compare batched imagination heads against the original stepwise learner."""

import copy

import torch
from torch import nn

from supermimic.config.g1.rl_cfg import SuperMimicConfig
from supermimic.mdp.rotations import so3_exp
from supermimic.mdp.state import flatten_state, normalize_xi, unflatten_state
from supermimic.rl.algorithm import SuperMimicAlgorithm, frozen


class StepwiseAlgorithm(SuperMimicAlgorithm):
  """The pre-batching policy update, retained as a numerical regression oracle.

  In particular, rewards use each post-action state and its pre-advance target,
  and critic features are evaluated separately for all H+1 trajectory states.
  """

  def policy_update(self, buffer):
    c = self.cfg
    sampled = buffer.sample_starts(c.pi_batch, c.hist)
    if sampled is None:
      return {"policy_skipped": 1.0}
    start, hist = sampled
    state = unflatten_state(start["s"])
    xi, f, a_prev = start["xi"], start["f"], start["a_prev"]
    xi_n = normalize_xi(xi)
    with torch.no_grad():
      history = self.mdp.add_noise(self.mdp.obs(
        unflatten_state(hist["s"]), hist["a_prev"], hist["xi"], hist["f"]))
    goal = self.mdp.goal(state, f, c.goal_frames)
    states, frames, goals, previous_actions = [state], [f], [goal], [a_prev]
    rewards, probabilities = [], []
    with frozen(self.wm, self.critic):
      for _ in range(c.horizon):
        next_state, action, history, next_f, next_goal = self._imag_step(
          state, history, f, xi, goal)
        rewards.append(self.mdp.reward(next_state, f, action, a_prev))
        probabilities.append(self.mdp.soft_termination(next_state, f))
        state, f, goal, a_prev = next_state, next_f, next_goal, action
        states.append(state)
        frames.append(f)
        goals.append(goal)
        previous_actions.append(a_prev)
      r, d = torch.stack(rewards), torch.stack(probabilities)
      alive = (1 - d).cumprod(0)
      prior_alive = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
      discount = c.gamma ** torch.arange(c.horizon, device=self.device)[:, None]
      objective = (discount * prior_alive * r).sum(0)
      objective += c.gamma ** c.horizon * alive[-1] * self.critic(
        self.critic_input(state, f, xi_n, goal, a_prev))
      loss = -objective.mean()
      self.opt_pi.zero_grad(set_to_none=True)
      loss.backward()
      norm = nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
      nn.utils.clip_grad_value_(self.policy.parameters(), 0.1)
      if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise FloatingPointError("Non-finite policy loss/gradient")
      self.opt_pi.step()
    with torch.no_grad():
      x = torch.stack([
        self.critic_input(tuple(t.detach() for t in s), ft, xi_n, g.detach(), a.detach())
        for s, ft, g, a in zip(states, frames, goals, previous_actions)
      ])
      self.critic.norm.update(x)
      v = self.critic(x)
      ret, targets = v[-1], []
      for t in range(c.horizon - 1, -1, -1):
        ret = r[t].detach() + c.gamma * (1 - d[t].detach()) * (
          (1 - c.lam) * v[t+1] + c.lam * ret)
        targets.append(ret)
      y = torch.stack(targets[::-1]).flatten()
      x = x[:-1].flatten(0, 1)
    losses = []
    for _ in range(c.critic_epochs):
      permutation = torch.randperm(len(y), device=self.device)
      for ids in permutation.tensor_split(c.critic_minibatches):
        if ids.numel() == 0:
          continue
        value_loss = (self.critic(x[ids]) - y[ids]).square().mean()
        self.opt_v.zero_grad(set_to_none=True)
        value_loss.backward()
        value_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        if not torch.isfinite(value_loss) or not torch.isfinite(value_norm):
          raise FloatingPointError("Non-finite critic loss/gradient")
        self.opt_v.step()
        losses.append(value_loss.detach())
    return {"policy_skipped": 0.0, "pi_loss": float(loss.detach()),
            "v_loss": float(torch.stack(losses).mean()), "pi_gradnorm": float(norm),
            "imag_survival": float(alive[-1].detach().mean()),
            "imag_reward": float((prior_alive * r).detach().mean() / self.mdp.max_step_reward)}


class FixedStarts:
  def __init__(self, mdp, cfg):
    # Rows 0 and 1 reach the last frames of different clips. Row 2 does not.
    f = torch.tensor([48, 55, 7])
    xi = torch.linspace(-0.01, 0.01, 3 * 33).reshape(3, 33)
    xi[:, 0] = torch.tensor([0.4, 0.75, 1.1])
    previous = torch.linspace(-0.2, 0.25, 3 * 29).reshape(3, 29)

    def make(frame):
      p, r, v, w = mdp.reference_root(frame)
      p = p + torch.tensor([0.07, -0.04, 0.15])
      r = r @ so3_exp(torch.tensor([0.1, -0.08, 0.2]))
      v = v + torch.tensor([0.04, -0.03, 0.02])
      w = w + torch.tensor([0.1, -0.2, 0.05])
      q = mdp.motion.joint_pos[frame] + torch.linspace(-0.07, 0.09, 29)
      qd = mdp.motion.joint_vel[frame] + torch.linspace(-0.1, 0.1, 29)
      return flatten_state((p, r, v, w, q, qd))

    hist_f = f[:, None] + torch.arange(1 - cfg.hist, 1)
    self.start = dict(s=make(f-1), a_prev=previous, f=f, xi=xi)
    self.history = dict(s=make(hist_f-1),
                        a_prev=previous[:, None].expand(-1, cfg.hist, -1),
                        f=hist_f, xi=xi[:, None].expand(-1, cfg.hist, -1))

  def sample_starts(self, batch, hist):
    assert batch == 3 and hist == self.history["s"].shape[1]
    return self.start, self.history


def test_batched_update_matches_original_stepwise_update(mdp):
  """The complete learner update, including noise and minibatches, is preserved."""
  torch.set_num_threads(2)
  torch.manual_seed(13)
  cfg = SuperMimicConfig(
    pi_batch=3, horizon=3, hist=2, goal_frames=3,
    critic_epochs=2, critic_minibatches=2,
    wm_hidden=(24, 24), policy_hidden=(24, 24), critic_hidden=(24,),
    amp="none", compile=False, logger="csv")
  expected = StepwiseAlgorithm(mdp, cfg, "cpu")
  actual = SuperMimicAlgorithm(mdp, cfg, "cpu")
  with torch.no_grad():
    # The default WM has a zero last layer: make dynamics action-dependent so
    # that this comparison exercises BPTT through the model, not just action cost.
    expected.wm.net[-1].weight.normal_(std=0.04)
    expected.wm.net[-1].bias.normal_(std=0.005)
    expected.wm.out_norm.var.fill_(0.01)
    expected.wm.out_norm.mean[2] = 0.025
    expected.wm.out_norm.mean[3:6] = torch.tensor([0.05, -0.03, 0.04])
  initial = copy.deepcopy(expected.state_dict())
  actual.load_state_dict(copy.deepcopy(initial))
  buffer = FixedStarts(mdp, cfg)

  reward_calls, critic_calls, wm_outputs = [], [], []
  reward_term = actual._reward_term
  critic_input = actual._critic_input

  def capture_reward(state, f, action, a_prev):
    reward_calls.append((f.detach().clone(), action.detach().clone(), a_prev.detach().clone()))
    assert state[0].shape == (cfg.horizon * cfg.pi_batch, 3)
    assert all(t.requires_grad for t in state)
    return reward_term(state, f, action, a_prev)

  def capture_critic(state, f, xi_n, goal, a_prev):
    critic_calls.append((len(f), torch.is_grad_enabled()))
    assert a_prev.requires_grad == torch.is_grad_enabled()
    return critic_input(state, f, xi_n, goal, a_prev)

  def retain_model_output(module, args, output):
    output.retain_grad()
    wm_outputs.append(output)

  actual._reward_term, actual._critic_input = capture_reward, capture_critic
  hook = actual.wm.net[-1].register_forward_hook(retain_model_output)
  rng_before = torch.get_rng_state().clone()
  expected_metrics = expected.policy_update(buffer)
  expected_rng = torch.get_rng_state().clone()
  torch.set_rng_state(rng_before)
  try:
    actual_metrics = actual.policy_update(buffer)
  finally:
    hook.remove()

  # Moving deterministic heads out of the rollout must not change noise draws
  # or the critic's minibatch order.
  assert torch.equal(torch.get_rng_state(), expected_rng)
  assert actual_metrics.keys() == expected_metrics.keys()
  for key in expected_metrics:
    torch.testing.assert_close(actual_metrics[key], expected_metrics[key], rtol=2e-5, atol=2e-6)
  for model_name in ("wm", "policy", "critic"):
    actual_model, expected_model = getattr(actual, model_name), getattr(expected, model_name)
    for name, value in actual_model.state_dict().items():
      torch.testing.assert_close(value, expected_model.state_dict()[name], rtol=2e-5, atol=2e-6)
    for actual_parameter, expected_parameter in zip(actual_model.parameters(), expected_model.parameters()):
      if expected_parameter.grad is None:
        assert actual_parameter.grad is None
      else:
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=3e-5, atol=3e-6)
    if model_name == "wm":
      assert all(torch.equal(value, initial["wm"][name]) for name, value in actual_model.state_dict().items())
      assert all(parameter.requires_grad for parameter in actual_model.parameters())
    else:
      assert any(not torch.equal(value, initial[model_name][name]) for name, value in actual_model.named_parameters())
  for optimizer_name in ("opt_pi", "opt_v"):
    actual_optimizer = getattr(actual, optimizer_name).state_dict()
    expected_optimizer = getattr(expected, optimizer_name).state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for index, state in actual_optimizer["state"].items():
      for key, value in state.items():
        torch.testing.assert_close(value, expected_optimizer["state"][index][key], rtol=3e-5, atol=3e-6)
  assert len(wm_outputs) == cfg.horizon
  assert all(output.grad is not None and torch.isfinite(output.grad).all()
             and output.grad.abs().sum() > 0 for output in wm_outputs)

  # One H*B reward/termination call, one terminal bootstrap and one (H+1)*B
  # critic call replace the per-step heads without losing the first a_prev.
  assert len(reward_calls) == 1
  assert critic_calls == [(3, True), (12, False)]
  frames, actions, previous = reward_calls[0]
  assert frames.reshape(3, 3).tolist() == [[48, 55, 7], [49, 56, 8], [50, 57, 9]]
  torch.testing.assert_close(previous[:3], buffer.start["a_prev"], rtol=0, atol=0)
  torch.testing.assert_close(previous[3:], actions[:-3], rtol=0, atol=0)
