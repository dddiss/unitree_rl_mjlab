"""Logging exercises real CPU optimization without a W&B account or GPU."""
import csv
import random
import re

import numpy as np
import pytest
import torch

from supermimic.rl.metrics import LEGACY_FIELDS
from supermimic.rl import runner as runner_module
from supermimic.rl.runner import SuperMimicRunner
from test_algorithm import small_config
from test_logging_backend import EXPECTED_WANDB_NAMES, sdk, read_csv
from test_train_resume_eval import TensorEnv


def seed():
  random.seed(713)
  np.random.seed(713)
  torch.manual_seed(713)


def test_csv_rows_preserve_training_reward_and_display_reward_without_dt(mdp, tmp_path):
  runner = SuperMimicRunner(TensorEnv(mdp), small_config(
    iterations=2, warmup_steps=4, wm_only_iters=1, log_every=1), tmp_path)
  runner.learn()
  fields, rows = read_csv(tmp_path / "train_log.csv")
  assert fields[:17] == list(LEGACY_FIELDS)
  assert all(not key.endswith("_count") for key in fields if key.startswith("Termination/"))
  assert len(rows) == 2
  for row in rows:
    assert float(row["Tracking/error_anchor_pos"]) == 0
    reward = sum(float(value) for key, value in row.items() if key.startswith("Reward/"))
    assert reward * runner.env.step_dt == pytest.approx(
      float(row["sim_reward"]) * mdp.max_step_reward, abs=1e-7)
    assert row["Episode/completed"] == "0"
    assert row["Episode/return_raw"] == row["Termination/anchor_pos_fraction"] == ""
    assert float(row["Performance/transitions_per_s"]) > 0
  assert float(rows[0]["Performance/policy_critic_update_s"]) == 0
  assert float(rows[1]["Performance/policy_critic_update_s"]) > 0
  assert runner.logger._closed


def test_recording_does_not_change_actions_rng_or_parameter_updates(mdp, tmp_path, sdk):
  torch.set_num_threads(2)
  results = []
  for mode in ("no_recording", "csv", "wandb"):
    seed()
    runner = SuperMimicRunner(TensorEnv(mdp), small_config(
      iterations=3, warmup_steps=4, wm_only_iters=1, log_every=1,
      logger="wandb" if mode == "wandb" else "csv", wandb_mode="offline"), tmp_path / mode)
    transitions = []
    original_add = runner.buffer.add
    def add(tr):
      transitions.append({key: value.clone() for key, value in tr.items()})
      original_add(tr)
    runner.buffer.add = add
    if mode == "no_recording":
      runner.interval_metrics.add_step = lambda _: None
      runner.interval_metrics.add_episodes = lambda *args: None
      runner._log = lambda _: None
    runner.learn()
    results.append((transitions, runner.algorithm.state_dict(),
                    (random.random(), np.random.rand(), torch.rand(4))))
  reference = results[0]
  for transitions, state, rng in results[1:]:
    assert len(transitions) == len(reference[0])
    for expected, actual in zip(reference[0], transitions):
      for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
    for name in ("wm", "policy", "critic"):
      for key, value in state[name].items():
        torch.testing.assert_close(value, reference[1][name][key], rtol=0, atol=0)
    assert rng[:2] == reference[2][:2]
    torch.testing.assert_close(rng[2], reference[2][2], rtol=0, atol=0)
  fields, rows = read_csv(tmp_path / "wandb/train_log.csv")
  assert fields[:17] == list(LEGACY_FIELDS)
  for row, payload in zip(rows, sdk.runs[0].rows):
    assert all(not key.endswith("_count") for key in row if key.startswith("Termination/"))
    expected = {EXPECTED_WANDB_NAMES.get(key, key): float(value)
                for key, value in row.items() if value != ""}
    assert payload == expected
    assert not set(EXPECTED_WANDB_NAMES).intersection(payload)
    displayed_reward = sum(value for key, value in payload.items()
                           if key.startswith("Reward/") and key != "Reward/total_normalized")
    assert displayed_reward * runner.env.step_dt == pytest.approx(
      payload["Reward/total_normalized"] * mdp.max_step_reward, abs=1e-7)


def test_legacy_checkpoint_and_csv_resume_start_new_wandb_run(mdp, tmp_path, sdk):
  cfg = small_config(iterations=1, warmup_steps=4, wm_only_iters=0, log_every=1,
                     logger="wandb", wandb_mode="offline")
  first = SuperMimicRunner(TensorEnv(mdp), cfg, tmp_path)
  first.learn()
  checkpoint = tmp_path / "last.pt"
  saved = torch.load(checkpoint, weights_only=False)
  for name in ("logger", "wandb_project", "wandb_entity", "wandb_mode", "log_dir", "run_name"):
    saved["config"].pop(name)
  torch.save(saved, checkpoint)
  # Reproduce the old 17-column file in the same resumed run directory.
  _, original_rows = read_csv(tmp_path / "train_log.csv")
  old = {key: original_rows[0][key] for key in LEGACY_FIELDS}
  with (tmp_path / "train_log.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=LEGACY_FIELDS)
    writer.writeheader(); writer.writerow(old)
  cfg.iterations = 2
  resumed = SuperMimicRunner(TensorEnv(mdp), cfg, tmp_path)
  resumed.load(checkpoint)
  assert resumed.logger is None and resumed.buffer.size == 0
  resumed.learn()
  assert len(sdk.runs) == 2
  assert sdk.runs[0].kwargs["id"] != sdk.runs[1].kwargs["id"]
  assert sdk.runs[1].kwargs["config"]["start_iteration"] == 1
  assert sdk.runs[1].kwargs["config"]["resume_checkpoint"] == str(checkpoint.resolve())
  assert sdk.runs[1].rows[0]["iteration"] == 2
  _, rows = read_csv(tmp_path / "train_log.csv")
  assert len(rows) == 2
  assert {key: rows[0][key] for key in LEGACY_FIELDS} == old
  assert rows[0]["Tracking/error_anchor_pos"] == ""
  assert rows[1]["Tracking/error_anchor_pos"] == "0.0"


def test_learner_exception_closes_logger(mdp, tmp_path, sdk, monkeypatch):
  runner = SuperMimicRunner(TensorEnv(mdp), small_config(
    iterations=1, logger="wandb", wandb_mode="offline"), tmp_path)
  def failed_collect(_):
    raise ValueError("collection failed")
  monkeypatch.setattr(runner, "collect", failed_collect)
  with pytest.raises(ValueError, match="collection failed"):
    runner.learn()
  assert sdk.runs[0].finishes == [1]
  assert runner.logger._file.closed
  runner.close()
  assert sdk.runs[0].finishes == [1]


def test_cleanup_failure_preserves_original_training_exception(mdp, tmp_path, sdk, monkeypatch):
  runner = SuperMimicRunner(TensorEnv(mdp), small_config(
    iterations=1, logger="wandb", wandb_mode="offline"), tmp_path)
  runner._start_logging()
  def failed_collect(_):
    raise ValueError("original training failure")
  def failed_finish(**kwargs):
    raise OSError("SDK cleanup failure")
  monkeypatch.setattr(runner, "collect", failed_collect)
  monkeypatch.setattr(sdk.runs[0], "finish", failed_finish)
  with pytest.raises(ValueError, match="original training failure") as caught:
    runner.learn()
  assert any("SDK cleanup failure" in note for note in caught.value.__notes__)
  assert runner.logger._file.closed


def test_episode_totals_survive_log_boundaries_and_partial_reset(mdp, tmp_path):
  runner = SuperMimicRunner(TensorEnv(mdp), small_config(), tmp_path)
  runner.collect(2)
  returns = runner.episode_return.clone()
  runner.iteration = 1
  runner._log(runner.cfg.lr)
  torch.testing.assert_close(runner.episode_return, returns)
  # End only row 0 on the next real transition, with overlapping reasons.
  original = runner.env.step
  def step(action):
    result = original(action)
    runner.env.last_transition["terminated"][0] = True
    for key in ("anchor_pos", "anchor_ori"):
      runner.env.last_step_metrics["termination"][key][0] = True
    return result
  runner.env.step = step
  runner.collect(1)
  metrics = runner.interval_metrics.values()
  assert metrics["Episode/completed"] == 1
  expected = returns[0] + runner.env.last_transition["r"][0]
  assert metrics["Episode/return_raw"] == pytest.approx(float(expected))
  assert metrics["Episode/length_seconds"] == pytest.approx(.06)
  assert metrics["Termination/anchor_pos_fraction"] == metrics["Termination/anchor_ori_fraction"] == 1
  assert runner.env.episode_length_buf.tolist() == [0, 3]
  assert runner.command.time_steps.tolist() == [1, 4]
  assert runner.episode_return[0] == 0
  torch.testing.assert_close(runner.episode_return[1], returns[1] + runner.env.last_transition["r"][1])
  runner.close()


def test_terminal_summary_follows_log_intervals_and_preserves_csv_wandb(
  mdp, tmp_path, sdk, monkeypatch, capsys,
):
  """Keep collection/log scheduling real; stub optimizer work for 105 iterations."""
  torch.set_num_threads(2)
  runner = SuperMimicRunner(TensorEnv(mdp), small_config(
    iterations=105, enable_warmup=False, log_every=50,
    logger="wandb", wandb_mode="offline"), tmp_path)
  monkeypatch.setattr(runner, "controller", lambda: torch.zeros(2, 29))
  monkeypatch.setattr(runner.algorithm, "observe", lambda *args: None)
  monkeypatch.setattr(runner.algorithm, "world_model_update", lambda _: {
    "wm_loss": .125, "wm_err_last": .25})
  monkeypatch.setattr(runner.algorithm, "policy_update", lambda _: {
    "pi_loss": -1.5, "v_loss": 2.5, "pi_gradnorm": .75,
    "policy_skipped": 0., "imag_reward": .6, "imag_survival": .8})
  monkeypatch.setattr(runner, "save", lambda _: None)

  # Include actual failures as well as motion-end truncations, so percentages
  # exercise nontrivial fractions and CSV/W&B cannot silently become 0–100.
  original_step = runner.env.step
  def step(action):
    result = original_step(action)
    if (runner.env_steps + 1) % 10 == 0:
      runner.env.last_transition["terminated"][0] = True
      runner.env.last_step_metrics["termination"]["anchor_pos"][0] = True
    return result
  monkeypatch.setattr(runner.env, "step", step)

  summaries, calls = [], []
  original_format = runner_module.format_terminal_summary
  def format_summary(row, **kwargs):
    before = row.copy()
    text = original_format(row, **kwargs)
    assert row == before
    summaries.append(text)
    calls.append((before, kwargs))
    return text
  monkeypatch.setattr(runner_module, "format_terminal_summary", format_summary)
  capsys.readouterr()
  runner.learn()
  output = capsys.readouterr().out

  assert output == "".join(f"{summary}\n" for summary in summaries)
  assert [row["iteration"] for row, _ in calls] == [50, 100, 105]
  assert [kwargs["interval_iterations"] for _, kwargs in calls] == [50, 50, 5]
  for (_, kwargs), summary, (start, stop) in zip(calls, summaries, ((1, 50), (51, 100), (101, 105))):
    assert kwargs["total_iterations"] == 105
    assert kwargs["interval_seconds"] > 0
    assert kwargs["wm_warmup"] is False
    assert f"Learning iteration {stop}/105" in summary
    assert f"iterations {start}–{stop}" in summary

  fields, rows = read_csv(tmp_path / "train_log.csv")
  assert [int(row["iteration"]) for row in rows] == [50, 100, 105]
  assert len(sdk.runs[0].rows) == len(summaries) == 3
  hidden_fields = (
    "pi_gradnorm", "policy_skipped", "sim_reward", "Episode/completed",
    "Episode/return_raw", "Episode/length_seconds", "imag_reward", "imag_survival",
  )
  assert any(0 < row["Termination/anchor_pos_fraction"] < 1 for row, _ in calls)
  for (original, _), csv_row, payload, summary in zip(calls, rows, sdk.runs[0].rows, summaries):
    expected = {EXPECTED_WANDB_NAMES.get(key, key): float(value)
                for key, value in csv_row.items() if value != ""}
    assert payload == expected
    for name in hidden_fields:
      display_name = EXPECTED_WANDB_NAMES.get(name, name)
      assert name in fields and csv_row[name] != ""
      assert payload[display_name] == original[name]
      assert display_name not in summary

    lines = [line.strip() for line in summary.splitlines() if line.strip()]
    learning_rate_index = next(i for i, line in enumerate(lines) if "Training/lr:" in line)
    assert lines[learning_rate_index + 1].startswith("Episode/length_steps:")
    assert f"{original['lr']:.3e}" in lines[learning_rate_index]
    assert f"{original['episode_length']:.2f} steps" in lines[learning_rate_index + 1]
    for name in ("Termination/anchor_pos_fraction", "Termination/motion_end_fraction"):
      fraction = original[name]
      assert 0 <= fraction <= 1
      line = next(line for line in lines if line.startswith(name + ":"))
      match = re.search(r"([0-9.]+)%", line)
      assert match is not None
      assert float(match.group(1)) == pytest.approx(100 * fraction, abs=.051)
      assert float(csv_row[name]) == payload[name] == fraction
