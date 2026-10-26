"""Return-based reset sampling, adapted from vibed (see THIRD_PARTY_NOTICES.md)."""
import warnings

import torch


class EpisodeBinReturns:
  """Score only the reset bin, using allmimic's finite attainable return.

  The first target is f=k+1, leaving clip_end-k valid control transitions.
  An active time limit can shorten this potential horizon; early failure cannot.
  No terminal case adds a bootstrap tail to the observed discounted return.
  """

  def __init__(self, motion, num_envs, gamma=0.99, max_step_reward=0.1,
               max_episode_steps=None):
    self.motion = motion
    self.gamma = gamma
    self.max_step_reward = max_step_reward
    self.max_episode_steps = max_episode_steps
    self.device = motion.device
    self.env_ids = torch.arange(num_envs, device=self.device)
    self.start_bin = torch.full_like(self.env_ids, -1)
    self.horizon = torch.zeros_like(self.env_ids)
    self.returns = torch.zeros(num_envs, device=self.device)
    self.discounts = torch.ones(num_envs, device=self.device)
    self.steps = torch.zeros_like(self.env_ids)

  @torch.no_grad()
  def add(self, target_frames, reward):
    """Consume pre-advance transition targets; capture k only on the first step."""
    first = self.steps == 0
    source_frames = target_frames - 1
    horizon = self.motion.clip_end[self.motion.clip_id[target_frames]] - source_frames
    if self.max_episode_steps is not None:
      horizon = horizon.clamp_max(self.max_episode_steps)
    self.start_bin.copy_(torch.where(first, self.motion.frame_bin[source_frames], self.start_bin))
    self.horizon.copy_(torch.where(first, horizon, self.horizon))
    self.returns += self.discounts * reward
    self.discounts *= self.gamma
    self.steps += 1

  @torch.no_grad()
  def finish(self, env_ids):
    """Normalize by the original potential horizon, including on early failure."""
    valid = (self.start_bin[env_ids] >= 0) & (self.horizon[env_ids] > 0)
    ids = env_ids[valid]
    max_return = self.max_step_reward * (1 - self.gamma ** self.horizon[ids].float()) / (1 - self.gamma)
    bins = self.start_bin[ids]
    scores = self.returns[ids] / max_return
    self.reset(env_ids)
    return bins, scores

  @torch.no_grad()
  def reset(self, env_ids=None):
    """Discard episode accumulators without altering other environments."""
    if env_ids is None:
      env_ids = self.env_ids
    self.start_bin[env_ids] = -1
    self.horizon[env_ids] = 0
    self.returns[env_ids] = 0
    self.discounts[env_ids] = 1
    self.steps[env_ids] = 0


class ReturnBasedSampler:
  def __init__(self, motion, gamma=0.99, max_step_reward=0.1, *, bin_ema=0.1):
    self.motion = motion
    self.gamma = gamma
    self.bin_ema = bin_ema
    # Scores are already divided by each episode's finite attainable return.
    self.bin_return = torch.zeros(motion.num_bins, device=motion.device)
    self.past_softmax = torch.full_like(self.bin_return, 1 / motion.num_bins)

  @torch.no_grad()
  def probabilities(self, update_history=False):
    soft = torch.softmax(-25 * self.bin_return, 0)
    if update_history:
      self.past_softmax.lerp_(soft, 0.001)
    return (0.5 / self.motion.num_bins + 0.4 * soft
            + 0.1 * self.past_softmax / self.past_softmax.sum())

  @torch.no_grad()
  def sample(self, count):
    p = self.probabilities(update_history=count > 0)
    bins = torch.multinomial(p, count, replacement=True) if count else torch.empty(
      0, dtype=torch.long, device=self.motion.device)
    return self.motion.sample_frames_from_bins(bins), bins

  @torch.no_grad()
  def update_returns(self, bins, values):
    """Average normalized scores for shared start bins, then apply the configured EMA."""
    valid = bins >= 0
    bins, values = bins[valid], values[valid]
    sums = torch.zeros_like(self.bin_return).scatter_add_(0, bins, values)
    counts = torch.zeros_like(self.bin_return).scatter_add_(0, bins, torch.ones_like(values))
    seen = counts > 0
    self.bin_return[seen] = ((1 - self.bin_ema) * self.bin_return[seen]
                             + self.bin_ema * sums[seen] / counts[seen])

  def state_dict(self):
    return {"normalized_bin_return": self.bin_return, "past_softmax": self.past_softmax}

  def load_state_dict(self, state):
    if "normalized_bin_return" not in state:
      # Old bootstrapped, along-trajectory returns cannot be converted to reset
      # scores without the original episodes. Do not silently reinterpret them.
      self.bin_return.zero_()
      self.past_softmax.fill_(1 / self.motion.num_bins)
      warnings.warn("Legacy bin returns use bootstrap/visited-bin scoring; resetting "
                    "sampling statistics for finite-horizon start-bin scoring. "
                    "Network, optimizer and training progress are restored normally.",
                    UserWarning, stacklevel=2)
      return
    self.bin_return.copy_(state["normalized_bin_return"])
    self.past_softmax.copy_(state["past_softmax"])
