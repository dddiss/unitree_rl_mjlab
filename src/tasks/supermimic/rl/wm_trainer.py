"""Autoregressive WM supervision, with optional visual and scalar reward targets.

These stage functions receive the learner context; optimizer order and window
masking intentionally match the original implementation.
"""
import torch
from torch import nn
from ..core.state import normalize_xi, unflatten_state
from ..perception.history import select_history
from ..perception.preprocessing import depth_float

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
  costs = []
  latent = None
  if self.cfg.depth_enabled:
    # Only the initial h reads real images, with gradients to the shared CNN/GRU.
    latent = self.encode_depth(depth_history)
  for j in range(self.cfg.wm_window):
    if self.foot_reward_names:
      state, latent, foot_cost = self.wm.step_with_foot_costs(
        state, actions[:, j], normalize_xi(xi[:, j]), latent)
      costs.append(foot_cost)
      if self.cfg.depth_enabled:
        latents.append(latent)
    elif self.cfg.depth_enabled:
      state, latent = self.wm.step_visual(state, actions[:, j], normalize_xi(xi[:, j]), latent)
      latents.append(latent)
    else:
      state = self.wm.step(state, actions[:, j], normalize_xi(xi[:, j]))
    predictions.append(state)
  states = tuple(torch.stack([s[i] for s in predictions], 1) for i in range(6))
  result = (states, torch.stack(latents, 1)) if self.cfg.depth_enabled else (states,)
  if self.foot_reward_names:
    return (*result, torch.stack(costs, 1))
  return result if self.cfg.depth_enabled else states


def world_model_update(self, buffer, *, strict=False):
  """Fit one batch; skip if strict sampling has no complete same-episode window."""
  c = self.cfg
  w = buffer.sample_windows(c.wm_batch, c.wm_window, strict=strict)
  if w is None:
    return None
  if self.foot_reward_names and tuple(buffer.foot_reward_names) != self.foot_reward_names:
    raise ValueError("Replay foot-cost labels must match the WM prediction heads")
  if c.depth_enabled:
    target_latents = self._depth_targets(w["depth_history"], w["depth_next"])
  rollout = self._world_rollout(w["s"], w["a"], w["xi"], w.get("depth_history"))
  if self.foot_reward_names:
    *rollout, costs = rollout
  if c.depth_enabled:
    predicted, latents = rollout
    images = self._decode_depth(latents)
    latent_error = (latents - target_latents).square().mean(-1)
    latent_loss = (latent_error * w["mask"]).sum() / c.wm_batch
    depth_error = (images - depth_float(w["depth_next"])).square().mean((-2, -1))
    depth_loss = (depth_error * w["mask"]).sum() / c.wm_batch
  else:
    predicted = rollout[0] if self.foot_reward_names else rollout
  scale = torch.maximum(self.wm.out_norm.std(), self.error_floor)
  error = (self.wm.state_error(predicted, unflatten_state(w["s1"])) / scale).square().mean(-1)
  state_loss = (error * w["mask"]).sum() / c.wm_batch
  loss = c.state_loss_weight * state_loss
  if c.depth_enabled:
    loss = loss + c.latent_loss_weight * latent_loss + c.depth_loss_weight * depth_loss
  cost_losses = {}
  for i, name in enumerate(self.foot_reward_names):
    # Same valid-window mask/reduction as state loss, including terminal rows.
    error_cost = (costs[..., i] - w["foot_costs"][..., i]).square()
    cost_losses[name] = torch.where(w["mask"].bool(), error_cost, 0.0).sum() / c.wm_batch
    loss = loss + self.reward_loss_weights[name] * cost_losses[name]
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
                   wm_depth_loss=float(depth_loss.detach() / c.wm_window))
  metrics.update({"wm_" + name + "_loss": float(value.detach() / c.wm_window)
                  for name, value in cost_losses.items()})
  if c.depth_enabled or self.foot_reward_names:
    metrics["wm_total_loss"] = float(loss.detach() / c.wm_window)
  return metrics
