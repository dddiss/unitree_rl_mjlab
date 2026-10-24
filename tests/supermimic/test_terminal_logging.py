"""Terminal formatting uses the logged interval without altering training data."""
from copy import deepcopy
import random
import re

import numpy as np
import pytest
import torch

from supermimic.rl.logging import format_terminal_summary


def row():
  return {
    "iteration": 600, "wall_h": 25. + 61. / 3600, "sim_steps": 6144000,
    "env_steps": 1500, "lr": 2e-4, "wm_loss": .125, "wm_err_last": .25,
    "pi_loss": -1.5, "v_loss": 2.5, "episode_length": 123.5,
    "Performance/iterations_per_s": 2., "Performance/transitions_per_s": 20480.,
    "Performance/collection_s": .06, "Performance/wm_update_s": .12,
    "Performance/policy_critic_update_s": .3,
    "Reward/motion_global_root_pos": .5, "Reward/motion_body_pos": 1.,
    "Reward/action_rate_l2": -.1,
    "Termination/motion_end_fraction": .125, "Termination/anchor_pos_fraction": .25,
    # These remain in CSV/W&B, but the compact terminal block omits them.
    "pi_gradnorm": 4., "policy_skipped": 0., "sim_reward": .8,
    "Episode/completed": 6, "Episode/return_raw": 3., "Episode/length_seconds": 2.,
    "imag_reward": .9, "imag_survival": .95,
  }


def render(values=None, **kwargs):
  options = dict(total_iterations=1000, interval_iterations=50, interval_seconds=25.)
  options.update(kwargs)
  return format_terminal_summary(row() if values is None else values, **options)


def field(text, label):
  matches = [line for line in text.splitlines() if line.lstrip().startswith(label + ":")]
  assert len(matches) == 1, (label, text)
  before, after = matches[0].split(":", 1)
  assert before + ":" == (label + ":").rjust(40)
  return after.strip()


def number(value):
  match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value.replace(",", ""))
  assert match is not None, value
  return float(match.group())


def test_summary_reads_csv_names_and_puts_episode_length_immediately_after_lr():
  text = render()
  for source, label in (
    ("wm_loss", "Loss/wm_loss"), ("wm_err_last", "Loss/wm_err_last"),
    ("pi_loss", "Loss/pi_loss"), ("v_loss", "Loss/v_loss"),
    ("lr", "Training/lr"), ("episode_length", "Episode/length_steps"),
  ):
    assert number(field(text, label)) == pytest.approx(row()[source])
  lines = text.splitlines()
  lr_index = next(i for i, line in enumerate(lines) if "Training/lr:" in line)
  assert "Episode/length_steps:" in lines[lr_index + 1]
  assert any(len(line) == 80 and len(set(line)) == 1 for line in lines)
  assert field(text, "Loss/wm_loss") == "0.125000"
  assert field(text, "Training/lr") == "2.000e-04"
  assert field(text, "Episode/length_steps") == "123.50 steps"


def test_summary_omits_excluded_metrics_without_changing_source_names():
  values = row()
  values["Reward/total_normalized"] = .8
  text = render(values)
  for name in (
    "pi_gradnorm", "policy_skipped", "sim_reward", "Reward/total_normalized",
    "Episode/completed", "Episode/return_raw", "Episode/length_seconds",
    "imag_reward", "imag_survival", "Imagination/reward", "Imagination/survival",
  ):
    assert name not in text


def test_rewards_are_unscaled_and_termination_percentages_are_dynamic():
  values = row()
  values["Reward/new_penalty"] = -.375
  values["Termination/new_reason_fraction"] = 1.25
  values["Termination/new_reason_count"] = 10
  text = render(values)
  assert field(text, "Reward/motion_global_root_pos") == "0.5000"
  assert field(text, "Termination/motion_end_fraction") == "12.50%"
  for key, value in values.items():
    if key.startswith("Reward/"):
      assert number(field(text, key)) == pytest.approx(value)
    elif key.startswith("Termination/") and key.endswith("_fraction"):
      shown = field(text, key)
      assert "%" in shown
      assert number(shown) == pytest.approx(100. * value)
  assert "Termination/new_reason_count" not in text
  assert "Termination/time_out_fraction" not in text
  values.pop("Termination/motion_end_fraction")
  assert "Termination/motion_end_fraction" not in render(values)


def test_missing_metrics_are_dashes_but_zero_remains_numeric():
  values = row()
  values.pop("wm_loss")
  values.update(wm_err_last="", pi_loss=None, v_loss=0., lr=0., episode_length="")
  values["Reward/motion_body_pos"] = 0.
  values["Termination/anchor_pos_fraction"] = 0.
  values["Termination/motion_end_fraction"] = ""
  text = render(values)
  for label in ("Loss/wm_loss", "Loss/wm_err_last", "Loss/pi_loss", "Episode/length_steps",
                "Termination/motion_end_fraction"):
    assert field(text, label) == "--"
  for label in ("Loss/v_loss", "Training/lr", "Reward/motion_body_pos",
                "Termination/anchor_pos_fraction"):
    assert number(field(text, label)) == 0.


def test_interval_window_and_stage_times_use_the_actual_recording_interval():
  text = render(interval_iterations=17, interval_seconds=8.5)
  assert "Learning iteration 600/1000" in text
  assert "SuperMimic | iterations 584–600" in text
  assert number(field(text, "Interval time")) == pytest.approx(8.5)
  assert number(field(text, "Mean iteration time")) == pytest.approx(.5)
  for label, expected in (
    ("Collection time", .06), ("WM update time", .12),
    ("Policy+Critic update time", .3), ("Total transitions", 6144000),
  ):
    assert number(field(text, label)) == pytest.approx(expected)
  # Throughput is read from the existing interval metrics, not recomputed from
  # the lifetime transition counter or the configured default log_every.
  computation = field(text, "Computation").replace(",", "")
  assert "20480" in computation
  assert re.search(r"\b2(?:\.0+)?\s+(?:it|iteration)", computation)


def test_total_elapsed_keeps_days_and_eta_uses_remaining_iterations():
  text = render()
  elapsed = field(text, "Total elapsed")
  assert re.search(r"1\s*(?:d|day)", elapsed)
  assert "01:01:01" in elapsed
  # (1000 - 600) / 2 iterations/s = 200 seconds.
  assert field(text, "ETA (estimated)") == "00:03:20"
  values = row()
  values["Performance/iterations_per_s"] = .001
  eta = field(render(values), "ETA (estimated)")
  assert re.search(r"4\s*(?:d|day)", eta)
  assert "15:06:40" in eta


@pytest.mark.parametrize("rate", [0., "", None])
def test_eta_is_missing_without_a_positive_measured_rate(rate):
  values = row()
  values["Performance/iterations_per_s"] = rate
  assert field(render(values), "ETA (estimated)") == "--"


def test_warmup_eta_is_hidden_but_completed_training_is_zero():
  assert field(render(wm_warmup=True), "ETA (estimated)") == "--"
  values = row()
  values["iteration"] = 1000
  values["Performance/iterations_per_s"] = 0.
  assert field(render(values, wm_warmup=True), "ETA (estimated)") == "00:00:00"


def test_terminal_formatting_preserves_input_and_random_streams():
  values = row()
  before = deepcopy(values)
  python_rng, numpy_rng = random.getstate(), np.random.get_state()
  torch_rng = torch.get_rng_state().clone()
  text = render(values)
  assert render(values) == text
  assert values == before
  assert random.getstate() == python_rng
  numpy_after = np.random.get_state()
  assert numpy_after[0] == numpy_rng[0]
  np.testing.assert_array_equal(numpy_after[1], numpy_rng[1])
  assert numpy_after[2:] == numpy_rng[2:]
  torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
