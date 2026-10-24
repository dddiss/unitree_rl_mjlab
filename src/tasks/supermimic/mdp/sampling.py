"""Return-based reset sampling, adapted from vibed (see THIRD_PARTY_NOTICES.md)."""
import torch


class EpisodeBinReturns:
  """Accumulate one return-to-go per bin visited during each episode.

  The first entry represents the sampled initialization frame k. Later entries
  represent the first source frame in each newly reached bin (source = target
  f - 1). Within a bin, store only its local discounted reward and discount;
  this avoids a per-frame history and unstable subtraction of long discounted
  prefixes. Completed episodes are folded backwards over their bin segments.
  """

  def __init__(self, motion, num_envs, gamma=0.99):
    self.gamma = gamma
    self.device = motion.device
    self.env_ids = torch.arange(num_envs, device=self.device)
    # An episode advances within one clip; storage scales with bins, not frames.
    max_frames = int(motion.clip_len.max().item())
    capacity = (max_frames + motion.bin_frames - 1) // motion.bin_frames + 1
    shape = (capacity, num_envs)
    self.bins = torch.full(shape, -1, dtype=torch.long, device=self.device)
    self.returns = torch.zeros(shape, device=self.device)
    self.discounts = torch.ones(shape, device=self.device)
    self.num_entries = torch.zeros(num_envs, dtype=torch.long, device=self.device)
    self.last_bin = torch.full_like(self.num_entries, -1)
    self.total_return = torch.zeros(num_envs, device=self.device)
    self.steps = torch.zeros_like(self.num_entries)

  @torch.no_grad()
  def add(self, bins, reward):
    """Record one simultaneous control step, before any environment resets."""
    valid = bins >= 0
    new = valid & (bins != self.last_bin)
    self.num_entries += new.long()
    self.last_bin.copy_(bins)
    rows = (self.num_entries - 1).clamp_min(0)
    # Fixed-size indexing keeps the per-step update on the device: no nonzero()
    # or Python branch is needed when different environments enter new bins.
    old_returns = self.returns[rows, self.env_ids]
    old_discounts = self.discounts[rows, self.env_ids]
    discount = torch.where(new, 1.0, old_discounts)
    self.bins[rows, self.env_ids] = torch.where(
      valid, bins, self.bins[rows, self.env_ids])
    self.returns[rows, self.env_ids] = (
      torch.where(new, 0.0, old_returns) + discount * reward * valid)
    self.discounts[rows, self.env_ids] = discount * torch.where(
      valid, self.gamma, 1.0)
    self.total_return += reward
    self.steps += 1

  @torch.no_grad()
  def finish(self, env_ids, terminated):
    """Return flat (bin IDs, returns) and clear only the completed environments.

    Failure has a zero tail. A timeout or clip end retains the existing average
    episode reward / (1 - gamma) tail, discounted from every bin's entry point.
    The mean is over the entire episode, including for later bins.
    """
    counts = self.num_entries[env_ids]
    n = int(counts.max().item()) if counts.numel() else 0
    mean_reward = self.total_return[env_ids] / self.steps[env_ids].clamp_min(1)
    tail = torch.where(terminated, 0.0, mean_reward / (1 - self.gamma))
    values = torch.empty((n, env_ids.numel()), device=self.device)
    for index in range(n - 1, -1, -1):
      tail = torch.where(
        index < counts,
        self.returns[index, env_ids] + self.discounts[index, env_ids] * tail,
        tail,
      )
      values[index] = tail
    valid = torch.arange(n, device=self.device)[:, None] < counts[None, :]
    bins = self.bins[:n, env_ids][valid]
    returns = values[valid]
    self.reset(env_ids)
    return bins, returns

  @torch.no_grad()
  def reset(self, env_ids=None):
    """Discard episode accumulators without altering other environments."""
    if env_ids is None:
      env_ids = self.env_ids
    self.num_entries[env_ids] = 0
    self.last_bin[env_ids] = -1
    self.total_return[env_ids] = 0
    self.steps[env_ids] = 0


class ReturnBasedSampler:
  def __init__(self, motion, gamma=0.99, max_step_reward=0.1):
    self.motion = motion
    self.gamma = gamma
    self.g_max = max_step_reward / (1 - gamma)
    self.bin_return = torch.zeros(motion.num_bins, device=motion.device)
    self.past_softmax = torch.full_like(self.bin_return, 1 / motion.num_bins)

  @torch.no_grad()
  def probabilities(self, update_history=False):
    soft = torch.softmax(-25 * self.bin_return / self.g_max, 0)
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
  def update(self, bins, discounted_return, total_return, steps, terminated):
    valid = bins >= 0  # An explicit GUI request may start on the final frame.
    bins = bins[valid]
    steps = steps[valid].float().clamp_min(1)
    boot = self.gamma ** steps * total_return[valid] / steps / (1 - self.gamma)
    values = discounted_return[valid] + torch.where(terminated[valid], 0.0, boot)
    self.update_returns(bins, values)

  @torch.no_grad()
  def update_returns(self, bins, values):
    """Apply one EMA update per bin after averaging all completed visits."""
    valid = bins >= 0
    bins, values = bins[valid], values[valid]
    sums = torch.zeros_like(self.bin_return).scatter_add_(0, bins, values)
    counts = torch.zeros_like(self.bin_return).scatter_add_(0, bins, torch.ones_like(values))
    seen = counts > 0
    self.bin_return[seen] = 0.8 * self.bin_return[seen] + 0.2 * sums[seen] / counts[seen]

  def state_dict(self):
    return {"bin_return": self.bin_return, "past_softmax": self.past_softmax}

  def load_state_dict(self, state):
    for key, value in state.items():
      getattr(self, key).copy_(value)
