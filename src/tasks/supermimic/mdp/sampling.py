"""Return-based reset sampling, adapted from vibed (see THIRD_PARTY_NOTICES.md)."""
import torch


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
    sums = torch.zeros_like(self.bin_return).scatter_add_(0, bins, values)
    counts = torch.zeros_like(self.bin_return).scatter_add_(0, bins, torch.ones_like(values))
    seen = counts > 0
    self.bin_return[seen] = 0.8 * self.bin_return[seen] + 0.2 * sums[seen] / counts[seen]

  def state_dict(self):
    return {"bin_return": self.bin_return, "past_softmax": self.past_softmax}

  def load_state_dict(self, state):
    for key, value in state.items():
      getattr(self, key).copy_(value)
