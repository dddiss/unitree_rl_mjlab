"""World-model regression, differentiable policy optimization and lambda returns.

Adapted from supermimic_vibed (THIRD_PARTY_NOTICES.md). No PPO updates.
"""
import math
from contextlib import contextmanager

import torch
from torch import nn

from ..mdp.state import GOAL_FRAME_DIM, INV_DIM, OBS_DIM, normalize_xi, unflatten_state
from . import networks
from .networks import WorldModel, Policy, Critic


@contextmanager
def frozen(*models):
  flags = [[p.requires_grad for p in m.parameters()] for m in models]
  try:
    for model in models:
      model.requires_grad_(False)
    yield
  finally:
    for model, previous in zip(models, flags):
      for p, flag in zip(model.parameters(), previous):
        p.requires_grad_(flag)


class SuperMimicAlgorithm:
  def __init__(self, mdp, cfg, device):
    self.mdp, self.cfg, self.device = mdp, cfg, torch.device(device)
    networks.AMP_DTYPE = torch.bfloat16 if cfg.amp == "bf16" else None
    goal_dim = GOAL_FRAME_DIM * cfg.goal_frames
    self.wm = WorldModel(cfg.wm_hidden).to(device)
    self.policy = Policy(OBS_DIM, goal_dim, cfg.hist, hidden=cfg.policy_hidden).to(device)
    self.critic = Critic(INV_DIM + 29 + 14 * 3 + 3 + 6 + 33 + goal_dim, cfg.critic_hidden).to(device)
    self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.lr)
    self.opt_pi = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
    self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
    self.error_floor = torch.tensor([3e-3]*3 + [5e-3]*3 + [2e-2]*3 + [5e-2]*3
                                    + [3e-3]*29 + [5e-2]*29, device=device)
    compile_fn = torch.compile if cfg.compile and self.device.type == "cuda" else lambda f: f
    self._world_rollout = compile_fn(self._rollout)
    self._imag_step = compile_fn(self._imagine_step)
    self._reward = compile_fn(mdp.reward)
    self._soft_term = compile_fn(mdp.soft_termination)

  def set_lr(self, iteration):
    c = self.cfg
    phase = min(iteration / c.lr_decay_iters, 1.0)
    lr = c.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * phase)))
    for optimizer in (self.opt_wm, self.opt_pi, self.opt_v):
      for group in optimizer.param_groups:
        group["lr"] = lr
    return lr

  @torch.no_grad()
  def observe(self, transition, xi):
    s, s1 = unflatten_state(transition["s"]), unflatten_state(transition["s1"])
    self.wm.in_norm.update(self.wm.inputs(s, transition["a"], normalize_xi(xi)))
    self.wm.out_norm.update(self.wm.targets(s, s1))

  def _rollout(self, states, actions, xi):
    state = unflatten_state(states[:, 0])
    predictions = []
    for j in range(self.cfg.wm_window):
      state = self.wm.step(state, actions[:, j], normalize_xi(xi[:, j]))
      predictions.append(state)
    return tuple(torch.stack([s[i] for s in predictions], 1) for i in range(6))

  def world_model_update(self, buffer):
    c = self.cfg
    w = buffer.sample_windows(c.wm_batch, c.wm_window)
    predicted = self._world_rollout(w["s"], w["a"], w["xi"])
    scale = torch.maximum(self.wm.out_norm.std(), self.error_floor)
    error = (self.wm.state_error(predicted, unflatten_state(w["s1"])) / scale).square().mean(-1)
    loss = (error * w["mask"]).sum() / c.wm_batch
    self.opt_wm.zero_grad(set_to_none=True)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
    if not torch.isfinite(loss) or not torch.isfinite(norm):
      raise FloatingPointError("Non-finite world-model loss/gradient")
    self.opt_wm.step()
    last = (error[:, -1] * w["mask"][:, -1]).sum() / w["mask"][:, -1].sum().clamp_min(1)
    return {"wm_loss": float(loss.detach() / c.wm_window), "wm_err_last": float(last.detach())}

  def critic_input(self, state, f, xi_n, goal):
    return torch.cat((self.mdp.critic_features(state, f, xi_n), goal), -1)

  def next_target(self, f):
    m = self.mdp.motion
    return torch.minimum(f + 1, m.clip_end[m.clip_id[f]])

  def _imagine_step(self, state, history, f, xi, goal):
    action = self.policy(history, self.mdp.add_goal_noise(goal))
    next_state = self.wm.step(state, action, normalize_xi(xi))
    # Reward evaluates next_state against f, BEFORE changing the target.
    next_f = self.next_target(f)
    obs = self.mdp.add_noise(self.mdp.obs(next_state, action, xi, next_f))
    history = torch.cat((history[:, 1:], obs[:, None]), 1)
    # Reference orientations are relative to the predicted robot at this step.
    # Keep a clean goal and its gradient for the critic; only the policy adds noise.
    next_goal = self.mdp.goal(next_state, next_f, self.cfg.goal_frames)
    return next_state, action, history, next_f, next_goal

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
      history = self.mdp.add_noise(self.mdp.obs(unflatten_state(hist["s"]), hist["a_prev"], hist["xi"], hist["f"]))
    goal = self.mdp.goal(state, f, c.goal_frames)
    states, frames, goals = [state], [f], [goal]
    rewards, probabilities = [], []
    with frozen(self.wm, self.critic):
      for _ in range(c.horizon):
        next_state, action, history, next_f, next_goal = self._imag_step(state, history, f, xi, goal)
        rewards.append(self._reward(next_state, f, action, a_prev))
        probabilities.append(self._soft_term(next_state, f))
        state, f, goal, a_prev = next_state, next_f, next_goal, action
        states.append(state); frames.append(f); goals.append(goal)
      r, d = torch.stack(rewards), torch.stack(probabilities)
      alive = (1 - d).cumprod(0)
      prior_alive = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
      discount = c.gamma ** torch.arange(c.horizon, device=self.device)[:, None]
      objective = (discount * prior_alive * r).sum(0)
      objective += c.gamma ** c.horizon * alive[-1] * self.critic(self.critic_input(state, f, xi_n, goal))
      loss = -objective.mean()
      self.opt_pi.zero_grad(set_to_none=True)
      loss.backward()
      norm = nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
      nn.utils.clip_grad_value_(self.policy.parameters(), 0.1)
      if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise FloatingPointError("Non-finite policy loss/gradient")
      self.opt_pi.step()
    # All targets are detached; critic training must not retain the BPTT graph.
    with torch.no_grad():
      x = torch.stack([self.critic_input(tuple(t.detach() for t in s), ft, xi_n, g.detach())
                       for s, ft, g in zip(states, frames, goals)])
      self.critic.norm.update(x)
      v = self.critic(x)
      ret, targets = v[-1], []
      for t in range(c.horizon - 1, -1, -1):
        ret = r[t].detach() + c.gamma * (1 - d[t].detach()) * ((1 - c.lam) * v[t+1] + c.lam * ret)
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

  def state_dict(self):
    return {key: getattr(self, key).state_dict() for key in
            ("wm", "policy", "critic", "opt_wm", "opt_pi", "opt_v")}

  def load_state_dict(self, state):
    for key, value in state.items():
      getattr(self, key).load_state_dict(value)
