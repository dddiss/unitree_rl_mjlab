"""Return-based reset sampling, adapted from vibed (see THIRD_PARTY_NOTICES.md)."""
import warnings

import torch


def _cap_probs(probabilities, cap):
  """Redistribute excess softmax mass as in allmimic's capped reset sampler."""
  for _ in range(8):
    over = probabilities > cap
    excess = (probabilities - cap).clamp_min(0).sum()
    under = torch.where(over, torch.zeros_like(probabilities), probabilities)
    # Our scores can be negative. If softmax underflows all remaining mass,
    # distribute the excess uniformly among uncapped bins instead of losing it.
    under = torch.where(under.sum() > 1e-30, under, (~over).to(probabilities.dtype))
    probabilities = torch.where(
      over, torch.full_like(probabilities, cap),
      probabilities + excess * under / under.sum().clamp_min(1e-30),
    )
  return probabilities


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
  def __init__(self, motion, gamma=0.99, max_step_reward=0.1, *, bin_ema=0.1,
               softmax_cap=50.0):
    self.motion = motion
    self.gamma = gamma
    self.bin_ema = bin_ema
    self.softmax_cap = softmax_cap
    # Scores are already divided by each episode's finite attainable return.
    self.bin_return = torch.zeros(motion.num_bins, device=motion.device)
    self.bin_initialized = torch.zeros_like(self.bin_return, dtype=torch.bool)
    self.past_softmax = torch.full_like(self.bin_return, 1 / motion.num_bins)

  @torch.no_grad()
  def probabilities(self, update_history=False):
    soft = torch.softmax(-25 * self.bin_return, 0)
    if 0 < self.softmax_cap < self.motion.num_bins:
      soft = _cap_probs(soft, self.softmax_cap / self.motion.num_bins)
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
    """Average shared start-bin scores; initialize directly, then use the EMA."""
    valid = bins >= 0
    bins, values = bins[valid], values[valid]
    sums = torch.zeros_like(self.bin_return).scatter_add_(0, bins, values)
    counts = torch.zeros_like(self.bin_return).scatter_add_(0, bins, torch.ones_like(values))
    seen = counts > 0
    mean = sums[seen] / counts[seen]
    self.bin_return[seen] = torch.where(
      self.bin_initialized[seen],
      (1 - self.bin_ema) * self.bin_return[seen] + self.bin_ema * mean,
      mean,
    )
    self.bin_initialized[seen] = True

  def state_dict(self):
    return {"normalized_bin_return": self.bin_return, "past_softmax": self.past_softmax,
            "bin_initialized": self.bin_initialized}

  def load_state_dict(self, state):
    if "normalized_bin_return" not in state:
      # Old bootstrapped, along-trajectory returns cannot be converted to reset
      # scores without the original episodes. Do not silently reinterpret them.
      self.bin_return.zero_()
      self.bin_initialized.zero_()
      self.past_softmax.fill_(1 / self.motion.num_bins)
      warnings.warn("Legacy bin returns use bootstrap/visited-bin scoring; resetting "
                    "sampling statistics for finite-horizon start-bin scoring. "
                    "Network, optimizer and training progress are restored normally.",
                    UserWarning, stacklevel=2)
      return
    self.bin_return.copy_(state["normalized_bin_return"])
    self.past_softmax.copy_(state["past_softmax"])
    if "bin_initialized" in state:
      self.bin_initialized.copy_(state["bin_initialized"])
    else:
      # Older normalized checkpoints did not track visits. Preserve their EMA
      # scores (including zero) rather than overwrite them as first observations.
      self.bin_initialized.fill_(True)
