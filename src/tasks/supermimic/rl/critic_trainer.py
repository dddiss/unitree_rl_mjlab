"""Batched Critic features, forward, regression and fixed-gradient clipping."""
import torch
from torch import nn

def critic_input(self, state, f, xi_n, goal, a_prev):
  return torch.cat((self.mdp.critic_features(state, f, xi_n, a_prev), goal), -1)


def critic_forward(self, x, depth_latent=None):
  if self.cfg.depth_enabled:
    return self.critic(x, depth_latent=depth_latent)
  return self.critic(x)


def critic_error(prediction, target):
  return (prediction - target).square().mean()


def _critic_loss(self, x, target, depth_latent=None):
  # One mean/backward/optimizer step for the original complete minibatch.
  return self._critic_error(self._critic_forward(x, depth_latent), target)



def fit(self, x, y, latents=None):
  c = self.cfg
  losses = []
  for _ in range(c.critic_epochs):
    permutation = torch.randperm(len(y), device=self.device)
    for ids in permutation.tensor_split(c.critic_minibatches):
      if ids.numel() == 0:
        continue
      value_loss = (self._critic_loss(x[ids], y[ids], latents[ids])
                    if c.depth_enabled else self._critic_loss(x[ids], y[ids]))
      self.opt_v.zero_grad(set_to_none=True)
      value_loss.backward()
      value_norm = nn.utils.clip_grad_norm_(self.v_parameters, 1.0)
      if not torch.isfinite(value_loss) or not torch.isfinite(value_norm):
        raise FloatingPointError("Non-finite critic loss/gradient")
      self.opt_v.step()
      losses.append(value_loss.detach())
  return torch.stack(losses).mean()
