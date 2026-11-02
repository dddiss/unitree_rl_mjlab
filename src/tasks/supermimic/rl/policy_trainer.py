"""Variable-length imagined Actor update and detached lambda-return targets."""
import torch
from torch import nn
from ..core.state import normalize_xi, unflatten_state, state_in_reference_frame
from .optim import frozen

def policy_update(self, buffer):
  c = self.cfg
  sampled = buffer.sample_starts(c.pi_batch, c.hist)
  if sampled is None:
    return {"policy_skipped": 1.0}
  start, hist = sampled
  # Freeze the sampled episode's reference placement, not the environment's
  # latest reset. The XY-invariant WM can roll forward in this same frame.
  state = state_in_reference_frame(unflatten_state(start["s"]), start.get("reference_xy_offset"))
  xi, f, a_prev = start["xi"], start["f"], start["a_prev"]
  xi_n = normalize_xi(xi)
  with torch.no_grad():
    history_state = state_in_reference_frame(unflatten_state(hist["s"]), hist.get("reference_xy_offset"))
    history = self.mdp.add_noise(self.mdp.obs(history_state, hist["a_prev"], hist["xi"], hist["f"]))
  goal = self.mdp.goal(state, f, c.goal_frames)
  states, frames, goals = [state], [f], [goal]
  actions, previous_actions = [], []
  predicted_costs = []
  batch_size = f.shape[0]
  motion = self.mdp.motion
  lengths = (motion.clip_end[motion.clip_id[f]] - f + 1).clamp(max=c.horizon)
  valid = torch.arange(c.horizon, device=self.device)[:, None] < lengths[None]
  height_offset = None
  if self.mdp.requires_height_offset:
    # Query all targets using the sampled episode's placement, independent
    # of predicted XY and of subsequently collected/reset replay rows.
    target_frames = torch.minimum(f[None] + torch.arange(c.horizon, device=self.device)[:, None],
                                  motion.clip_end[motion.clip_id[f]][None])
    height_offset = self.mdp.termination_height_offset(
      target_frames, start.get("env_origin"), start.get("reference_xy_offset")).reshape(-1)

  def flat(rows):
    return torch.stack(rows).flatten(0, 1)

  with frozen(self.wm, self.critic):
    if c.depth_enabled:
      with torch.no_grad():
        latent = self._encode_depth(start["depth_history"])
      visual_states = [latent]
    for t in range(c.horizon):
      if c.depth_enabled:
        result = self._imag_step(state, history, f, xi, goal, latent)
      else:
        result = self._imag_step(state, history, f, xi, goal)
      if self.foot_reward_names:
        *result, costs = result
        predicted_costs.append(costs)
      if c.depth_enabled:
        next_state, action, next_history, next_f, next_goal, next_latent = result
      else:
        next_state, action, next_history, next_f, next_goal = result
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
    reward_inputs = (reward_states, flat(frames[:-1]), flat(actions), flat(previous_actions))
    if self.foot_reward_names:
      reward_inputs += (flat(predicted_costs),)
    # The rollout stays in its starting episode/terrain assignment. Expand
    # labels in the same time-major order as the flattened imagined states.
    terrain_tolerance = start.get("terrain_tolerance")
    reward_kwargs = {} if terrain_tolerance is None else {
      "terrain_tolerance": terrain_tolerance.unsqueeze(0).expand(c.horizon, -1).reshape(-1),
    }
    if height_offset is not None:
      reward_kwargs["height_offset"] = height_offset
    r, d = self._reward_term(*reward_inputs, **reward_kwargs)
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
      v = self._critic_forward(x, latent_sequence)
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
  value_loss = self._fit_critic(x, y, critic_latents if c.depth_enabled else None)
  return {"policy_skipped": 0.0, "pi_loss": float(loss.detach()),
          "v_loss": float(value_loss), "pi_gradnorm": float(norm),
          "imag_survival": float(alive[-1].detach().mean()),
          "imag_reward": float((prior_alive * r).detach().sum() / valid.sum() / self.mdp.max_step_reward)}
