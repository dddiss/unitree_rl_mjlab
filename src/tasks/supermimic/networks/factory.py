"""One factory for training, checkpoint loading, evaluation and playback.

Replacement classes implement the same forward/normalizer/dynamics interfaces
as the default networks. They can change architecture without modifying a training loop.
Factories receive explicit dimensions; the G1 MLP sizes live in rl_cfg.py.
"""
from dataclasses import asdict, dataclass
from importlib import import_module

from ..core.state import GOAL_FRAME_DIM, INV_DIM, OBS_DIM, NUM_J, XI_DIM


def resolve_class(path):
  if not isinstance(path, str):
    raise TypeError("Network class must be an import path so checkpoints can reconstruct it")
  module, name = path.rsplit(":", 1)
  return getattr(import_module(module, package=__package__), name)


@dataclass(frozen=True)
class FeatureSpec:
  obs_dim: int
  goal_dim: int
  history: int
  critic_dim: int
  action_dim: int
  latent_dim: int

  @classmethod
  def from_mdp(cls, mdp, cfg):
    goal = GOAL_FRAME_DIM * cfg.goal_frames
    critic = INV_DIM + NUM_J + len(mdp.body_indexes) * 3 + 3 + NUM_J + XI_DIM + goal
    return cls(OBS_DIM, goal, cfg.hist, critic, NUM_J,
               cfg.depth_latent_dim if cfg.depth_enabled else 0)


class NetworkFactory:
  def __init__(self, cfg, features):
    self.cfg, self.features = cfg, features

  def world_model(self, schema):
    c = self.cfg
    model = resolve_class(c.world_model_class)(
      hidden=c.wm_hidden, depth_cfg=c, foot_reward_names=schema.names, **c.world_model_kwargs)
    # Default costs are all nonnegative, retaining identical modules/state keys.
    if schema.names and not all(schema.nonnegative):
      model.set_reward_output_domains(schema.nonnegative)
    return model

  def policy(self, encoder=None):
    c, f = self.cfg, self.features
    return resolve_class(c.actor_class)(
      f.obs_dim, f.goal_dim, f.history, hidden=c.policy_hidden,
      depth_cfg=c, depth_encoder=encoder, **c.actor_kwargs)

  def critic(self, encoder=None):
    c = self.cfg
    return resolve_class(c.critic_class)(self.features.critic_dim, c.critic_hidden,
                                        depth_cfg=c, depth_encoder=encoder, **c.critic_kwargs)

  def build(self, schema, device):
    # Keep construction order/RNG draws identical to the original algorithm.
    wm = self.world_model(schema).to(device)
    encoder = wm.depth_encoder if self.cfg.depth_enabled else None
    actor = self.policy(encoder).to(device)
    critic = self.critic(encoder).to(device)
    if encoder is not None and (actor.depth_encoder is not encoder or critic.depth_encoder is not encoder):
      raise ValueError("Visual Actor/WM/Critic must share the supplied perception object")
    return wm, actor, critic

  def metadata(self, schema):
    names = ("world_model_class", "actor_class", "critic_class")
    if self.cfg.depth_enabled:
      names += ("perception_class", "depth_decoder_class")
    return {"features": asdict(self.features), "reward_targets": schema.metadata(),
            "classes": {name: getattr(self.cfg, name) for name in names}}


def make_depth_encoder(cfg):
  cls = resolve_class(getattr(cfg, "perception_class", ".depth_encoder:DepthEncoder"))
  return cls(cfg.depth_latent_dim, cfg.depth_frame_dim, cfg.depth_checkpoint,
             cfg.depth_batch_size, cfg.depth_frames)
