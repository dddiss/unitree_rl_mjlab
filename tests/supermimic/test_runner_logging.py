"""Logging exercises real CPU optimization without a W&B account or GPU."""
import csv
import random

import numpy as np
import pytest
import torch

from supermimic.rl.metrics import LEGACY_FIELDS
from supermimic.rl.runner import SuperMimicRunner
from test_algorithm import small_config
from test_logging_backend import sdk, read_csv
from test_train_resume_eval import TensorEnv


def seed():
  random.seed(713)
  np.random.seed(713)
  torch.manual_seed(713)


def test_csv_rows_preserve_legacy_and_reward_totals(mdp, tmp_path):
  runner = SuperMimicRunner(TensorEnv(mdp), small_config(
    iterations=2, warmup_steps=4, wm_only_iters=1, log_every=1), tmp_path)
  runner.learn()
  fields, rows = read_csv(tmp_path / "train_log.csv")
  assert fields[:17] == list(LEGACY_FIELDS)
  assert len(rows) == 2
  for row in rows:
    assert float(row["Tracking/error_anchor_pos"]) == 0
    reward = sum(float(value) for key, value in row.items() if key.startswith("Reward/"))
    assert reward == pytest.approx(float(row["sim_reward"]) * mdp.max_step_reward, abs=1e-7)
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
  _, rows = read_csv(tmp_path / "wandb/train_log.csv")
  for row, payload in zip(rows, sdk.runs[0].rows):
    assert set(payload) == {key for key, value in row.items() if value != ""}
    for key, value in payload.items():
      assert float(row[key]) == value


def test_legacy_checkpoint_and_csv_resume_start_new_wandb_run(mdp, tmp_path, sdk):
  cfg = small_config(iterations=1, warmup_steps=4, wm_only_iters=0, log_every=1,
                     logger="wandb", wandb_mode="offline")
  first = SuperMimicRunner(TensorEnv(mdp), cfg, tmp_path)
  first.learn()
  checkpoint = tmp_path / "last.pt"
  saved = torch.load(checkpoint, weights_only=False)
  for name in ("logger", "wandb_project", "wandb_entity", "wandb_name", "wandb_mode"):
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
