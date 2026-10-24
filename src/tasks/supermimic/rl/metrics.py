"""Interval reductions for logging; no replay, optimizer or sampler mutation."""
import math

import torch

from ..mdp.logging_metrics import TRACKING_METRICS


LEGACY_FIELDS = (
  "iteration", "wall_h", "sim_steps", "env_steps", "lr",
  "wm_loss", "wm_err_last", "pi_loss", "v_loss", "pi_gradnorm",
  "imag_survival", "imag_reward", "policy_skipped", "sim_reward",
  "episode_length", "episode_return", "failure_fraction",
)
SAMPLING_FIELDS = ("Sampling/entropy", "Sampling/top1_prob", "Sampling/top1_bin")
PERFORMANCE_FIELDS = (
  "Performance/collection_s", "Performance/wm_update_s", "Performance/policy_critic_update_s",
  "Performance/iterations_per_s", "Performance/transitions_per_s",
)


class IntervalMetrics:
  """Accumulate on device, transferring scalar reductions only when logging.

  Episode sums belong to the runner and survive interval boundaries. This
  object only receives completed episodes, so unequal completion batches do
  not bias their means. A reset can match multiple termination reasons.
  """
  def __init__(self, reward_names, termination_names, step_dt, device):
    self.reward_names = tuple(reward_names)
    self.termination_names = tuple(termination_names)
    self.step_dt = step_dt
    self.step_fields = tuple(f"Tracking/{key}" for key in TRACKING_METRICS) + tuple(
      f"Reward/{key}" for key in self.reward_names)
    self.fields = self.step_fields + tuple(
      f"Termination/{key}_{kind}" for key in self.termination_names for kind in ("count", "fraction")
    ) + ("Episode/completed", "Episode/return_raw", "Episode/length_seconds")
    self.step_sums = torch.zeros(len(self.step_fields), dtype=torch.float64, device=device)
    self.reason_counts = torch.zeros(len(self.termination_names), dtype=torch.long, device=device)
    self.episode_sums = torch.zeros(2, dtype=torch.float64, device=device)
    self.sample_count = self.completed = 0

  @torch.no_grad()
  def add_step(self, snapshot):
    values = [snapshot["tracking"][key] for key in TRACKING_METRICS]
    values += [snapshot["reward"][key] for key in self.reward_names]
    self.step_sums.add_(torch.stack([v.sum(dtype=torch.float64) for v in values]))
    self.sample_count += values[0].numel()
    if self.termination_names:
      self.reason_counts.add_(torch.stack([
        snapshot["termination"][key].sum(dtype=torch.long) for key in self.termination_names]))

  @torch.no_grad()
  def add_episodes(self, returns, steps):
    self.completed += returns.numel()
    self.episode_sums.add_(torch.stack((returns.sum(dtype=torch.float64),
                                      steps.sum(dtype=torch.float64) * self.step_dt)))

  def values(self):
    step_sums, reason_counts, episode_sums = (
      self.step_sums.cpu().tolist(), self.reason_counts.cpu().tolist(), self.episode_sums.cpu().tolist())
    row = {key: value / self.sample_count if self.sample_count else ""
           for key, value in zip(self.step_fields, step_sums)}
    for key, count in zip(self.termination_names, reason_counts):
      row[f"Termination/{key}_count"] = count
      row[f"Termination/{key}_fraction"] = count / self.completed if self.completed else ""
    row["Episode/completed"] = self.completed
    row["Episode/return_raw"] = episode_sums[0] / self.completed if self.completed else ""
    row["Episode/length_seconds"] = episode_sums[1] / self.completed if self.completed else ""
    return row

  def clear(self):
    self.step_sums.zero_()
    self.reason_counts.zero_()
    self.episode_sums.zero_()
    self.sample_count = self.completed = 0


@torch.no_grad()
def sampling_metrics(command):
  """Describe the actual reset distribution without sampling/updating its EMA."""
  motion = command.motion
  mode = command.cfg.sampling_mode
  if mode == "return_based":
    probabilities = command.sampler.probabilities(update_history=False)
  elif mode == "uniform":
    probabilities = torch.full((motion.num_bins,), 1 / motion.num_bins, device=motion.device)
  elif mode == "start":
    # Start mode samples clips uniformly, assigning mass only to their first bin.
    probabilities = torch.zeros(motion.num_bins, device=motion.device)
    bins = motion.frame_bin[motion.clip_start]
    probabilities.scatter_add_(0, bins, torch.full_like(bins, 1 / motion.num_clips, dtype=torch.float32))
  else:
    raise ValueError(f"Unknown sampling_mode: {mode}")
  entropy = (-(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum()
             / math.log(motion.num_bins)) if motion.num_bins > 1 else probabilities.new_tensor(1.0)
  top = probabilities.argmax()
  entropy_value, top_probability = torch.stack((entropy, probabilities[top])).cpu().tolist()
  return dict(zip(SAMPLING_FIELDS, (entropy_value, top_probability, int(top))))
