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
from ..perception.history import select_history
from ..perception.preprocessing import depth_float


def parameter_groups(model, lr, visual_groups=(), excluded=()):
  """Keep each parameter in exactly one optimizer group with its own base LR."""
  groups, excluded = [], {id(p) for p in excluded}
  for module, rate in visual_groups:
    parameters = list(module.parameters())
    excluded.update(id(p) for p in parameters)
    groups.append({"params": parameters, "lr": rate, "base_lr": rate})
  return [{"params": [p for p in model.parameters() if id(p) not in excluded],
           "lr": lr, "base_lr": lr}, *groups]


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
    self.wm = WorldModel(cfg.wm_hidden, depth_cfg=cfg).to(device)
    encoder = self.wm.depth_encoder if cfg.depth_enabled else None
    self.policy = Policy(OBS_DIM, goal_dim, cfg.hist, hidden=cfg.policy_hidden,
                         depth_cfg=cfg, depth_encoder=encoder).to(device)
    self.critic = Critic(INV_DIM + 29 + 14 * 3 + 3 + 29 + 33 + goal_dim, cfg.critic_hidden,
                         depth_cfg=cfg, depth_encoder=encoder).to(device)
    wm_groups = ((self.wm.depth_encoder, cfg.wm_perception_lr),
                 (self.wm.depth_decoder, cfg.depth_decoder_lr)) if cfg.depth_enabled else ()
    shared = tuple(encoder.parameters()) if encoder is not None else ()
    self.opt_wm = torch.optim.Adam(parameter_groups(self.wm, cfg.lr, wm_groups))
    self.opt_pi = torch.optim.Adam(parameter_groups(self.policy, cfg.lr, excluded=shared), betas=(0.9, 0.95))
    self.opt_v = torch.optim.Adam(parameter_groups(self.critic, cfg.lr, excluded=shared))
    # Shared encoder gradients from the preceding WM step must never enter
    # Actor/Critic clipping, nor be stepped by either of their optimizers.
    self.error_floor = torch.tensor([3e-3]*3 + [5e-3]*3 + [2e-2]*3 + [5e-2]*3
                                    + [3e-3]*29 + [5e-2]*29, device=device)
    compile_fn = torch.compile if cfg.compile and self.device.type == "cuda" else lambda f, **kw: f
    self._world_rollout = compile_fn(self._rollout)
    self._imag_step = compile_fn(self._imagine_step)
    self._reward_term = compile_fn(self.reward_term)
    self._critic_input = compile_fn(self.critic_input)
    # Compile a bounded critic chunk, not a graph containing a variable number
    # of visual chunks. The final chunk may vary with motion-end masking.
    self._critic_forward_chunk = compile_fn(self.critic_forward, dynamic=True)
    self._critic_error = compile_fn(self.critic_error, dynamic=True)
    if cfg.depth_enabled:
      self._encode_depth = compile_fn(self.encode_depth)
      self._depth_targets = compile_fn(self.depth_targets)
      self._decode_depth = compile_fn(self.wm.decode_depth)

  @property
  def pi_parameters(self):
    return [p for group in self.opt_pi.param_groups for p in group["params"]]

  @property
  def v_parameters(self):
    return [p for group in self.opt_v.param_groups for p in group["params"]]

  def set_lr(self, iteration):
    c = self.cfg
    phase = min(iteration / c.lr_decay_iters, 1.0)
    lr = c.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * phase)))
    for optimizer in (self.opt_wm, self.opt_pi, self.opt_v):
      for group in optimizer.param_groups:
        group["lr"] = group.get("base_lr", c.lr) * (lr / c.lr)
    return lr

  @torch.no_grad()
  def observe(self, transition, xi):
    s, s1 = unflatten_state(transition["s"]), unflatten_state(transition["s1"])
    self.wm.in_norm.update(self.wm.inputs(s, transition["a"], normalize_xi(xi)))
    self.wm.out_norm.update(self.wm.targets(s, s1))

  def encode_depth(self, history):
    return self.wm.depth_encoder(select_history(history, self.cfg.depth_frames, self.cfg.depth_skip))

  @torch.no_grad()
  def depth_targets(self, history, next_images):
    """True h[t+1:t+K+1], including pre-reset terminal images, never predictions.

    Later cross-episode rows are masked by the existing WM mask. Encode each
    real frame once and aggregate the same strided windows used in collection.
    This target branch does not update perception; the initial rollout h does.
    """
    c = self.cfg
    images = torch.cat((depth_float(history), depth_float(next_images)), 1)
    features = self.wm.depth_encoder.encode_frames(images)
    times = history.shape[1] + torch.arange(next_images.shape[1], device=images.device)
    ages = torch.arange(c.depth_frames - 1, -1, -1, device=images.device) * c.depth_skip
    return self.wm.depth_encoder.aggregate(features[:, times[:, None] - ages[None]])

  def _rollout(self, states, actions, xi, depth_history=None):
    state = unflatten_state(states[:, 0])
    predictions = []
    latents = []
    if self.cfg.depth_enabled:
      # Only the initial h reads real images, with gradients to the shared CNN/GRU.
      latent = self.encode_depth(depth_history)
    for j in range(self.cfg.wm_window):
      if self.cfg.depth_enabled:
        state, latent = self.wm.step_visual(state, actions[:, j], normalize_xi(xi[:, j]), latent)
        latents.append(latent)
      else:
        state = self.wm.step(state, actions[:, j], normalize_xi(xi[:, j]))
      predictions.append(state)
    states = tuple(torch.stack([s[i] for s in predictions], 1) for i in range(6))
    return (states, torch.stack(latents, 1)) if self.cfg.depth_enabled else states

  def world_model_update(self, buffer, *, strict=False):
    """Fit one batch; skip if strict sampling has no complete same-episode window."""
    c = self.cfg
    w = buffer.sample_windows(c.wm_batch, c.wm_window, strict=strict)
    if w is None:
      return None
    if c.depth_enabled:
      target_latents = self._depth_targets(w["depth_history"], w["depth_next"])
      predicted, latents = self._world_rollout(w["s"], w["a"], w["xi"], w["depth_history"])
      images = self._decode_depth(latents)
      latent_error = (latents - target_latents).square().mean(-1)
      latent_loss = (latent_error * w["mask"]).sum() / c.wm_batch
      depth_error = (images - depth_float(w["depth_next"])).square().mean((-2, -1))
      depth_loss = (depth_error * w["mask"]).sum() / c.wm_batch
    else:
      predicted = self._world_rollout(w["s"], w["a"], w["xi"])
    scale = torch.maximum(self.wm.out_norm.std(), self.error_floor)
    error = (self.wm.state_error(predicted, unflatten_state(w["s1"])) / scale).square().mean(-1)
    state_loss = (error * w["mask"]).sum() / c.wm_batch
    loss = c.state_loss_weight * state_loss
    if c.depth_enabled:
      loss = loss + c.latent_loss_weight * latent_loss + c.depth_loss_weight * depth_loss
    self.opt_wm.zero_grad(set_to_none=True)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
    if not torch.isfinite(loss) or not torch.isfinite(norm):
      raise FloatingPointError("Non-finite world-model loss/gradient")
    self.opt_wm.step()
    last = (error[:, -1] * w["mask"][:, -1]).sum() / w["mask"][:, -1].sum().clamp_min(1)
    metrics = {"wm_loss": float(state_loss.detach() / c.wm_window), "wm_err_last": float(last.detach())}
    if c.depth_enabled:
      metrics.update(wm_latent_loss=float(latent_loss.detach() / c.wm_window),
                     wm_depth_loss=float(depth_loss.detach() / c.wm_window),
                     wm_total_loss=float(loss.detach() / c.wm_window))
    return metrics

  def critic_input(self, state, f, xi_n, goal, a_prev):
    return torch.cat((self.mdp.critic_features(state, f, xi_n, a_prev), goal), -1)

  def critic_forward(self, x, depth_latent=None):
    if self.cfg.depth_enabled:
      return self.critic(x, depth_latent=depth_latent)
    return self.critic(x)

  def _critic_forward(self, x, depth_latent=None):
    if not self.cfg.depth_enabled:
      return self._critic_forward_chunk(x)
    predictions = []
    for start in range(0, x.shape[0], self.cfg.depth_batch_size):
      stop = start + self.cfg.depth_batch_size
      predictions.append(self._critic_forward_chunk(
        x[start:stop], depth_latent[start:stop]))
    return torch.cat(predictions, 0)

  @staticmethod
  def critic_error(prediction, target):
    return (prediction - target).square().mean()

  def _critic_loss(self, x, target, depth_latent=None):
    # One mean/backward/optimizer step for the original complete minibatch.
    return self._critic_error(self._critic_forward(x, depth_latent), target)

  def reward_term(self, state, f, action, a_prev):
    return self.mdp.reward(state, f, action, a_prev), self.mdp.soft_termination(state, f)

  def next_target(self, f):
    m = self.mdp.motion
    return torch.minimum(f + 1, m.clip_end[m.clip_id[f]])

  def _imagine_step(self, state, history, f, xi, goal, depth_latent=None):
    if self.cfg.depth_enabled:
      action = self.policy(history, self.mdp.add_goal_noise(goal), depth_latent=depth_latent)
      next_state, next_latent = self.wm.step_visual(state, action, normalize_xi(xi), depth_latent)
    else:
      action = self.policy(history, self.mdp.add_goal_noise(goal))
      next_state = self.wm.step(state, action, normalize_xi(xi))
    # Reward evaluates next_state against f, BEFORE changing the target.
    next_f = self.next_target(f)
    obs = self.mdp.add_noise(self.mdp.obs(next_state, action, xi, next_f))
    history = torch.cat((history[:, 1:], obs[:, None]), 1)
    # Reference orientations are relative to the predicted robot at this step.
    # Keep a clean goal and its gradient for the critic; only the policy adds noise.
    next_goal = self.mdp.goal(next_state, next_f, self.cfg.goal_frames)
    result = next_state, action, history, next_f, next_goal
    return (*result, next_latent) if self.cfg.depth_enabled else result

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
    actions, previous_actions = [], []
    batch_size = f.shape[0]
    motion = self.mdp.motion
    lengths = (motion.clip_end[motion.clip_id[f]] - f + 1).clamp(max=c.horizon)
    valid = torch.arange(c.horizon, device=self.device)[:, None] < lengths[None]

    def flat(rows):
      return torch.stack(rows).flatten(0, 1)

    with frozen(self.wm, self.critic):
      if c.depth_enabled:
        with torch.no_grad():
          latent = self._encode_depth(start["depth_history"])
        visual_states = [latent]
      for t in range(c.horizon):
        if c.depth_enabled:
          next_state, action, next_history, next_f, next_goal, next_latent = self._imag_step(
            state, history, f, xi, goal, latent)
        else:
          next_state, action, next_history, next_f, next_goal = self._imag_step(state, history, f, xi, goal)
        # Keep fixed batch shapes, but hold every row at its own endpoint.
        # Padded steps must not change the bootstrap state or previous action.
        def advance(old, new):
          return torch.where(valid[t].reshape((-1,) + (1,) * (new.ndim - 1)), new, old)
        next_state = tuple(advance(old, new) for old, new in zip(state, next_state))
        action = advance(a_prev, action)
        history = advance(history, next_history)
        next_f, next_goal = advance(f, next_f), advance(goal, next_goal)
        if c.depth_enabled:
          # Hold h at each row's true motion endpoint, just like physical state.
          # Keep gradients through predicted h to all preceding Actor actions.
          latent = advance(latent, next_latent)
          visual_states.append(latent)
        actions.append(action); previous_actions.append(a_prev)
        state, f, goal, a_prev = next_state, next_f, next_goal, action
        states.append(state); frames.append(f); goals.append(goal)
      # Time-major batch: post-action s[t+1] still compares with target f[t].
      # Keep the rollout graph, including both actions in the action-rate cost.
      reward_states = tuple(flat([s[i] for s in states[1:]]) for i in range(6))
      r, d = self._reward_term(reward_states, flat(frames[:-1]), flat(actions), flat(previous_actions))
      r, d = r.reshape(c.horizon, batch_size), d.reshape(c.horizon, batch_size)
      # Motion end is a truncation, not a failure: stop rewards there while
      # retaining the last valid state's soft survival factor for bootstrapping.
      r, d = torch.where(valid, r, 0.0), torch.where(valid, d, 0.0)
      alive = (1 - d).cumprod(0)
      prior_alive = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
      discount = c.gamma ** torch.arange(c.horizon, device=self.device)[:, None]
      objective = (discount * prior_alive * r).sum(0)
      bootstrap_input = self._critic_input(state, f, xi_n, goal, a_prev)
      if c.depth_enabled:
        bootstrap = self._critic_forward(bootstrap_input, latent)
      else:
        bootstrap = self._critic_forward(bootstrap_input)
      objective += c.gamma ** lengths * alive[-1] * bootstrap
      # Always divide by the configured horizon (32 by default), never by each
      # row's shorter valid length. Scale the bootstrap together with rewards.
      loss = -objective.mean() / c.horizon
      self.opt_pi.zero_grad(set_to_none=True)
      loss.backward()
      norm = nn.utils.clip_grad_norm_(self.pi_parameters, c.pi_clip)
      nn.utils.clip_grad_value_(self.pi_parameters, c.pi_clip_value)
      if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise FloatingPointError("Non-finite policy loss/gradient")
      self.opt_pi.step()
    # All targets are detached; critic training must not retain the BPTT graph.
    with torch.no_grad():
      critic_states = tuple(flat([s[i].detach() for s in states]) for i in range(6))
      critic_xi = xi_n.unsqueeze(0).expand(c.horizon + 1, -1, -1).flatten(0, 1)
      # s[0] uses replay a_prev; s[t+1] uses imagined a[t], including bootstrap.
      critic_previous = flat([a.detach() for a in (*previous_actions, a_prev)])
      x = self._critic_input(critic_states, flat(frames), critic_xi,
                             flat([g.detach() for g in goals]), critic_previous)
      x = x.reshape(c.horizon + 1, batch_size, -1)
      state_valid = torch.arange(c.horizon + 1, device=self.device)[:, None] <= lengths[None]
      self.critic.norm.update(x[state_valid])
      if c.depth_enabled:
        # Shared perception and dynamics stay fixed during Actor/Critic updates.
        # Detached latent rows are reusable throughout every critic epoch.
        latent_sequence = torch.stack([h.detach() for h in visual_states])
        v = torch.stack([self._critic_forward(x[t], latent_sequence[t])
                         for t in range(c.horizon + 1)])
        critic_latents = latent_sequence[:-1][valid]
      else:
        v = self._critic_forward(x)
      ret, targets = v[-1], []
      for t in range(c.horizon - 1, -1, -1):
        next_return = r[t].detach() + c.gamma * (1 - d[t].detach()) * ((1 - c.lam) * v[t+1] + c.lam * ret)
        # After motion end, carry V(s_L) back to the final valid transition;
        # do not insert artificial rewards or discounts for padded steps.
        ret = torch.where(valid[t], next_return, v[t])
        targets.append(ret)
      y = torch.stack(targets[::-1])[valid]
      x = x[:-1][valid]
    losses = []
    for _ in range(c.critic_epochs):
      permutation = torch.randperm(len(y), device=self.device)
      for ids in permutation.tensor_split(c.critic_minibatches):
        if ids.numel() == 0:
          continue
        value_loss = (self._critic_loss(x[ids], y[ids], critic_latents[ids])
                      if c.depth_enabled else self._critic_loss(x[ids], y[ids]))
        self.opt_v.zero_grad(set_to_none=True)
        value_loss.backward()
        value_norm = nn.utils.clip_grad_norm_(self.v_parameters, 1.0)
        if not torch.isfinite(value_loss) or not torch.isfinite(value_norm):
          raise FloatingPointError("Non-finite critic loss/gradient")
        self.opt_v.step()
        losses.append(value_loss.detach())
    return {"policy_skipped": 0.0, "pi_loss": float(loss.detach()),
            "v_loss": float(torch.stack(losses).mean()), "pi_gradnorm": float(norm),
            "imag_survival": float(alive[-1].detach().mean()),
            "imag_reward": float((prior_alive * r).detach().sum() / valid.sum() / self.mdp.max_step_reward)}

  def state_dict(self):
    return {key: getattr(self, key).state_dict() for key in
            ("wm", "policy", "critic", "opt_wm", "opt_pi", "opt_v")}

  def load_state_dict(self, state):
    for key, value in state.items():
      getattr(self, key).load_state_dict(value)
