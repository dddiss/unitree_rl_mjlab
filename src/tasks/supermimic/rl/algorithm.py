"""World-model regression, differentiable policy optimization and lambda returns.

Adapted from supermimic_vibed (THIRD_PARTY_NOTICES.md). No PPO updates.
"""
import math

import torch

from ..core.state import normalize_xi, unflatten_state
from ..networks import auxiliary
from ..networks.factory import NetworkFactory, FeatureSpec
from .replay_schema import RewardTargetSchema
from . import wm_trainer, policy_trainer, critic_trainer, imagination
from .optim import parameter_groups, frozen

class SuperMimicAlgorithm:
  def __init__(self, mdp, cfg, device):
    self.mdp, self.cfg, self.device = mdp, cfg, torch.device(device)
    self.reward_schema = RewardTargetSchema.from_mdp(mdp)
    self.foot_reward_names = self.reward_schema.names  # old public name/checkpoint compatibility
    self.reward_loss_weights = self.reward_schema.loss_weights(cfg)
    mdp.validate_imagination_rewards(self.foot_reward_names)
    auxiliary.AMP_DTYPE = torch.bfloat16 if cfg.amp == "bf16" else None
    self.network_factory = NetworkFactory(cfg, FeatureSpec.from_mdp(mdp, cfg))
    self.wm, self.policy, self.critic = self.network_factory.build(self.reward_schema, device)
    encoder = self.wm.depth_encoder if cfg.depth_enabled else None
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
    # Critic consumes ready-made h: evaluate the full trajectory/minibatch,
    # independent of perception's image workspace limit. Valid row counts vary.
    self._critic_forward = compile_fn(self.critic_forward, dynamic=True)
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
    return wm_trainer.encode_depth(self, history)

  def depth_targets(self, history, next_images):
    return wm_trainer.depth_targets(self, history, next_images)

  def _rollout(self, states, actions, xi, depth_history=None):
    return wm_trainer._rollout(self, states, actions, xi, depth_history)

  def world_model_update(self, buffer, *, strict=False):
    return wm_trainer.world_model_update(self, buffer, strict=strict)

  def critic_input(self, state, f, xi_n, goal, a_prev):
    return critic_trainer.critic_input(self, state, f, xi_n, goal, a_prev)

  def critic_forward(self, x, depth_latent=None):
    return critic_trainer.critic_forward(self, x, depth_latent)

  @staticmethod
  def critic_error(prediction, target):
    return critic_trainer.critic_error(prediction, target)

  def _critic_loss(self, x, target, depth_latent=None):
    return critic_trainer._critic_loss(self, x, target, depth_latent)

  def reward_term(self, state, f, action, a_prev, foot_costs=None, *, terrain_tolerance=None, height_offset=None):
    return imagination.reward_term(self, state, f, action, a_prev, foot_costs, terrain_tolerance=terrain_tolerance, height_offset=height_offset)

  def next_target(self, f):
    return imagination.next_target(self, f)

  def _imagine_step(self, state, history, f, xi, goal, depth_latent=None):
    return imagination._imagine_step(self, state, history, f, xi, goal, depth_latent)

  def _fit_critic(self, x, y, latents=None):
    return critic_trainer.fit(self, x, y, latents)

  def policy_update(self, buffer):
    return policy_trainer.policy_update(self, buffer)

  def state_dict(self):
    return {key: getattr(self, key).state_dict() for key in
            ("wm", "policy", "critic", "opt_wm", "opt_pi", "opt_v")}

  def load_state_dict(self, state):
    for key, value in state.items():
      getattr(self, key).load_state_dict(value)
