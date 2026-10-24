"""Metric reductions must preserve weighting and leave reset sampling untouched."""
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from supermimic.mdp.logging_metrics import TRACKING_METRICS
from supermimic.mdp.sampling import ReturnBasedSampler
from supermimic.rl.metrics import IntervalMetrics, sampling_metrics


REWARDS = (
  "motion_global_root_pos", "motion_global_root_ori", "motion_body_pos",
  "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel",
  "action_rate_l2", "joint_limit",
)
REASONS = ("time_out", "motion_end", "anchor_pos", "anchor_ori")


def snapshot(values, *, reward=None, termination=None):
  values = torch.as_tensor(values, dtype=torch.float32)
  return {
    "tracking": {name: values.clone() for name in TRACKING_METRICS},
    "reward": reward or {name: torch.zeros_like(values) for name in REWARDS},
    "termination": termination or {
      name: torch.zeros_like(values, dtype=torch.bool) for name in REASONS},
  }


def metrics():
  return IntervalMetrics(REWARDS, REASONS, step_dt=0.02, device="cpu")


def test_episode_means_weight_each_completed_episode_not_each_batch():
  accumulated = metrics()
  accumulated.add_episodes(torch.tensor([4.]), torch.tensor([10]))
  accumulated.add_episodes(torch.tensor([2., 6., 12.]), torch.tensor([20, 30, 40]))
  row = accumulated.values()
  assert row["Episode/completed"] == 4
  assert row["Episode/return_raw"] == pytest.approx(6.)
  assert row["Episode/length_seconds"] == pytest.approx(0.5)
  assert row["Episode/return_raw"] != pytest.approx((4. + 20. / 3.) / 2.)


def test_step_metrics_weight_transitions_and_reward_contributions_sum():
  accumulated = metrics()
  # Already weighted and dt-scaled contributions, including negative penalties.
  weights = (0.5, 0.5, 1., 1., 1., 1., -0.1, -2.)
  raw_batches = (torch.tensor([1., 2.]), torch.tensor([3., 4., 5.]))
  actual_total_rewards = []
  for raw in raw_batches:
    contributions = {name: raw * weight * 0.02 for name, weight in zip(REWARDS, weights)}
    actual_total_rewards.append(torch.stack(list(contributions.values())).sum(0))
    accumulated.add_step(snapshot(raw, reward=contributions))
  row = accumulated.values()
  for name in TRACKING_METRICS:
    assert row[f"Tracking/{name}"] == pytest.approx(3.)
  assert sum(row[f"Reward/{name}"] for name in REWARDS) == pytest.approx(
    torch.cat(actual_total_rewards).mean().item())
  assert row["Reward/action_rate_l2"] < 0
  assert row["Reward/joint_limit"] < 0
  assert accumulated.sample_count == 5


def test_no_completed_episodes_keeps_counts_zero_and_means_empty():
  accumulated = metrics()
  accumulated.add_step(snapshot([1., 3.]))
  accumulated.add_episodes(torch.empty(0), torch.empty(0, dtype=torch.long))
  row = accumulated.values()
  assert row["Episode/completed"] == 0
  assert row["Episode/return_raw"] == row["Episode/length_seconds"] == ""
  for name in REASONS:
    assert row[f"Termination/{name}_count"] == 0
    assert row[f"Termination/{name}_fraction"] == ""
  assert row["Tracking/error_anchor_pos"] == 2.


def test_termination_causes_can_overlap_and_sum_above_one():
  accumulated = metrics()
  terms = {
    "time_out": torch.tensor([False, True, False]),
    "motion_end": torch.tensor([False, False, False]),
    "anchor_pos": torch.tensor([True, True, False]),
    "anchor_ori": torch.tensor([True, False, False]),
  }
  accumulated.add_step(snapshot([0., 0., 0.], termination=terms))
  accumulated.add_episodes(torch.tensor([1., 2.]), torch.tensor([10, 500]))
  row = accumulated.values()
  assert row["Episode/completed"] == 2
  assert row["Termination/anchor_pos_count"] == 2
  assert row["Termination/anchor_pos_fraction"] == 1.
  assert row["Termination/anchor_ori_fraction"] == 0.5
  assert row["Termination/time_out_fraction"] == 0.5
  assert row["Termination/motion_end_fraction"] == 0.
  assert sum(row[f"Termination/{name}_fraction"] for name in REASONS) == 2.


def test_interval_clear_resets_reductions_without_mutating_episode_inputs():
  accumulated = metrics()
  returns, steps = torch.tensor([2., 6.]), torch.tensor([10, 30])
  accumulated.add_step(snapshot([2., 6.]))
  accumulated.add_episodes(returns, steps)
  previous_row = accumulated.values()
  accumulated.clear()
  empty = accumulated.values()
  assert empty["Episode/completed"] == 0
  assert empty["Episode/return_raw"] == ""
  assert all(empty[key] == "" for key in accumulated.step_fields)
  assert accumulated.sample_count == 0
  assert previous_row["Episode/completed"] == 2
  torch.testing.assert_close(returns, torch.tensor([2., 6.]))
  torch.testing.assert_close(steps, torch.tensor([10, 30]))
  accumulated.add_step(snapshot([12.]))
  accumulated.add_episodes(torch.tensor([15.]), torch.tensor([50]))
  following = accumulated.values()
  assert following["Tracking/error_anchor_pos"] == 12.
  assert following["Episode/completed"] == 1
  assert following["Episode/return_raw"] == 15.
  assert following["Episode/length_seconds"] == 1.


def command(mode, num_bins=4):
  if num_bins == 1:
    starts, frame_bin = [0], [0, 0, -1]
  else:
    # Two clips, two bins each; the terminal frames are not reset starts.
    starts, frame_bin = [0, 5], [0, 0, 1, 1, -1, 2, 2, 3, -1]
  motion = SimpleNamespace(
    num_bins=num_bins, device=torch.device("cpu"), num_clips=len(starts),
    clip_start=torch.tensor(starts), frame_bin=torch.tensor(frame_bin),
  )
  return SimpleNamespace(
    motion=motion, cfg=SimpleNamespace(sampling_mode=mode),
    sampler=ReturnBasedSampler(motion),
  )


@pytest.mark.parametrize("mode", ["return_based", "uniform", "start"])
def test_single_bin_sampling_is_normalized_and_finite(mode):
  row = sampling_metrics(command(mode, num_bins=1))
  assert row == {"Sampling/entropy": 1., "Sampling/top1_prob": 1., "Sampling/top1_bin": 0}


def test_uniform_diagnostics_describe_uniform_bins_not_uniform_clips():
  cmd = command("uniform")
  # A skewed return sampler must not leak into this mode's displayed distribution.
  cmd.sampler.bin_return.copy_(torch.tensor([0., 10., 20., 30.]))
  row = sampling_metrics(cmd)
  assert row["Sampling/entropy"] == pytest.approx(1.)
  assert row["Sampling/top1_prob"] == pytest.approx(0.25)
  assert row["Sampling/top1_bin"] == 0


def test_start_diagnostics_only_assign_probability_to_each_clips_first_bin():
  cmd = command("start")
  row = sampling_metrics(cmd)
  # p = [1/2, 0, 1/2, 0], normalized by log(total bins), not log(active bins).
  assert row["Sampling/entropy"] == pytest.approx(math.log(2) / math.log(4))
  assert row["Sampling/top1_prob"] == pytest.approx(0.5)
  assert row["Sampling/top1_bin"] == 0


def test_return_based_diagnostics_preserve_sampler_and_all_rng_states():
  cmd = command("return_based")
  cmd.sampler.bin_return.copy_(torch.tensor([4., 3., 0., 2.]))
  cmd.sampler.past_softmax.copy_(torch.tensor([0.1, 0.2, 0.6, 0.1]))
  before = {key: value.clone() for key, value in cmd.sampler.state_dict().items()}
  python_rng = random.getstate()
  numpy_rng = np.random.get_state()
  torch_rng = torch.get_rng_state().clone()
  p = cmd.sampler.probabilities(update_history=False)
  row = sampling_metrics(cmd)
  assert sampling_metrics(cmd) == row
  expected_entropy = -(p * p.log()).sum() / math.log(cmd.motion.num_bins)
  assert row["Sampling/entropy"] == pytest.approx(expected_entropy.item())
  assert row["Sampling/top1_prob"] == pytest.approx(p[2].item())
  assert row["Sampling/top1_bin"] == 2
  assert isinstance(row["Sampling/top1_bin"], int)
  for key, value in before.items():
    torch.testing.assert_close(cmd.sampler.state_dict()[key], value, rtol=0, atol=0)
  assert random.getstate() == python_rng
  numpy_after = np.random.get_state()
  assert numpy_after[0] == numpy_rng[0]
  np.testing.assert_array_equal(numpy_after[1], numpy_rng[1])
  assert numpy_after[2:] == numpy_rng[2:]
  torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
