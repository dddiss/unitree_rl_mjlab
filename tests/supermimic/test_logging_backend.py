"""CSV/W&B logging contracts, without credentials or network access."""
import builtins
import csv
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from supermimic.rl import logging as logging_module
from supermimic.rl.logging import TrainingLogger


EXPECTED_WANDB_NAMES = {
  "wm_loss": "Loss/wm_loss", "wm_err_last": "Loss/wm_err_last",
  "pi_loss": "Loss/pi_loss", "v_loss": "Loss/v_loss",
  "lr": "Training/lr", "pi_gradnorm": "Training/pi_gradnorm",
  "policy_skipped": "Training/policy_skipped", "sim_steps": "Training/sim_steps",
  "env_steps": "Training/env_steps", "wall_h": "Performance/wall_h",
  "imag_reward": "Imagination/reward", "imag_survival": "Imagination/survival",
  "sim_reward": "Reward/total_normalized", "episode_length": "Episode/length_steps",
  "episode_return": "Episode/return_normalized", "failure_fraction": "Episode/failure_fraction",
}


@dataclass
class Config:
  logger: str = "wandb"
  wandb_project: str = "unitree-supermimic"
  wandb_entity: str | None = None
  run_name: str = "g1"
  wandb_mode: str = "offline"
  seed: int = 123


def read_csv(path):
  with Path(path).open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream)
    return reader.fieldnames, list(reader)


def consume_rng():
  random.random()
  np.random.rand()
  torch.rand(1)


class FakeRun:
  def __init__(self, kwargs):
    self.kwargs = kwargs
    self.axes = []
    self.rows = []
    self.finishes = []

  def define_metric(self, *args, **kwargs):
    consume_rng()
    self.axes.append((args, kwargs))

  def log(self, row):
    consume_rng()
    # Every W&B submission must already be visible to a CSV reader.
    _, rows = read_csv(Path(self.kwargs["dir"]) / "train_log.csv")
    assert rows[-1]["iteration"] == str(row["iteration"])
    self.rows.append(row.copy())

  def finish(self, exit_code=0):
    consume_rng()
    self.finishes.append(exit_code)


@pytest.fixture
def sdk(monkeypatch):
  result = SimpleNamespace(runs=[], settings=[])

  def settings(**kwargs):
    consume_rng()
    result.settings.append(kwargs)
    return kwargs

  def init(**kwargs):
    consume_rng()
    run = FakeRun(kwargs)
    result.runs.append(run)
    return run

  result.init = init
  result.Settings = settings
  monkeypatch.setitem(sys.modules, "wandb", result)
  return result


def test_csv_mode_does_not_import_wandb(tmp_path, monkeypatch):
  original = builtins.__import__

  def guarded_import(name, *args, **kwargs):
    if name == "wandb":
      raise AssertionError("CSV mode must not require W&B")
    return original(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", guarded_import)
  logger = TrainingLogger(tmp_path, Config(logger="csv"), ["iteration", "wm_loss"])
  logger.log({"iteration": 1, "wm_loss": .125})
  logger.close()
  assert read_csv(tmp_path / "train_log.csv") == (
    ["iteration", "wm_loss"], [{"iteration": "1", "wm_loss": "0.125"}],
  )


def test_extending_csv_retains_old_values_and_columns(tmp_path):
  path = tmp_path / "train_log.csv"
  path.write_text('iteration,wm_loss,legacy\n1,0.500000,"keep,value"\n', encoding="utf-8")
  logger = TrainingLogger(tmp_path, Config(logger="csv"), ["iteration", "wm_loss", "Tracking/error_anchor_pos"])
  logger.log({"iteration": 2, "wm_loss": .25, "Tracking/error_anchor_pos": .05})
  logger.close()
  fields, rows = read_csv(path)
  assert fields == ["iteration", "wm_loss", "legacy", "Tracking/error_anchor_pos"]
  assert rows[0] == {"iteration": "1", "wm_loss": "0.500000", "legacy": "keep,value", "Tracking/error_anchor_pos": ""}
  assert rows[1]["legacy"] == ""
  assert rows[1]["Tracking/error_anchor_pos"] == "0.05"


def test_failed_atomic_migration_keeps_original_csv(tmp_path, monkeypatch):
  path = tmp_path / "train_log.csv"
  original = b"iteration,wm_loss\n1,0.500000\n"
  path.write_bytes(original)

  def fail(*args):
    raise OSError("simulated replacement failure")

  monkeypatch.setattr(logging_module.os, "replace", fail)
  with pytest.raises(OSError, match="replacement failure"):
    TrainingLogger(tmp_path, Config(logger="csv"), ["iteration", "wm_loss", "new"])
  assert path.read_bytes() == original
  assert list(tmp_path.glob(".*.tmp")) == []


def test_wandb_matches_csv_and_omits_missing_metrics(tmp_path, sdk):
  logger = TrainingLogger(tmp_path, Config(), ["iteration", "loss", "empty", "missing"])
  row = {"iteration": 50, "loss": np.float64(.125), "empty": "", "missing": None}
  logger.log(row)
  logger.close()
  _, csv_rows = read_csv(tmp_path / "train_log.csv")
  assert sdk.runs[0].rows == [{"iteration": 50, "loss": .125}]
  assert csv_rows == [{"iteration": "50", "loss": "0.125", "empty": "", "missing": ""}]
  assert sdk.runs[0].axes == [(("iteration",), {}), (("*",), {"step_metric": "iteration"})]
  assert sdk.settings[0] == dict(console="off", disable_code=True, disable_git=True,
                                x_disable_stats=True, x_disable_meta=True)


def test_wandb_groups_metrics_without_changing_csv_or_input(tmp_path, sdk):
  row = {
    "iteration": 50, "wm_loss": np.float64(.125), "wm_err_last": .25,
    "pi_loss": -1.5, "v_loss": 2.5, "pi_gradnorm": .75,
    "imag_reward": .8, "imag_survival": .9, "lr": .0002, "policy_skipped": 0.,
    "sim_steps": 512000, "env_steps": 125, "wall_h": .01,
    "sim_reward": .7, "episode_length": 40., "episode_return": .6, "failure_fraction": .5,
    "Reward/motion_body_pos": .9, "Tracking/error_anchor_pos": .05,
    "Termination/anchor_pos_fraction": .5, "Episode/return_raw": 3.,
    "Sampling/entropy": .8, "Performance/collection_s": .065,
  }
  original = row.copy()
  logger = TrainingLogger(tmp_path, Config(), list(row))
  logger.log(row)
  # Cover missing keys, None and blank values across all renamed categories.
  missing = {"iteration": 100, **{
    key: None if index % 3 == 0 else ""
    for index, key in enumerate(EXPECTED_WANDB_NAMES) if index % 3 != 2}}
  missing_original = missing.copy()
  logger.log(missing)
  logger.close()
  assert row == original
  assert isinstance(row["wm_loss"], np.float64)
  assert missing == missing_original
  expected = {EXPECTED_WANDB_NAMES.get(key, key): value for key, value in original.items()}
  assert sdk.runs[0].rows == [expected, {"iteration": 100}]
  assert not set(EXPECTED_WANDB_NAMES).intersection(sdk.runs[0].rows[0])
  fields, rows = read_csv(tmp_path / "train_log.csv")
  assert fields == list(original)
  for key, value in original.items():
    assert float(rows[0][key]) == value
  assert rows[1] == {key: "100" if key == "iteration" else "" for key in fields}


def test_each_invocation_gets_new_wandb_id_and_resume_metadata(tmp_path, sdk, monkeypatch):
  monkeypatch.setenv("WANDB_RUN_ID", "old-run-from-environment")
  first = TrainingLogger(tmp_path / "first", Config(), ["iteration"])
  first.close()
  resumed = TrainingLogger(tmp_path / "resumed", Config(), ["iteration"],
                           resume_checkpoint=tmp_path / "first/last.pt", start_iteration=120)
  resumed.close(exit_code=1)
  initial, second = [run.kwargs for run in sdk.runs]
  assert initial["id"] != second["id"]
  for kwargs in (initial, second):
    assert kwargs["id"] != "old-run-from-environment"
    assert kwargs["resume"] == "never" and kwargs["reinit"] == "create_new"
    assert kwargs["mode"] == "offline"
    assert kwargs["save_code"] is False and kwargs["sync_tensorboard"] is False
  assert initial["config"]["start_iteration"] == 0
  assert initial["config"]["resume_checkpoint"] is None
  assert second["config"]["start_iteration"] == 120
  assert second["config"]["resume_checkpoint"] == str(tmp_path / "first/last.pt")
  assert initial["name"] == "first"
  assert second["name"] == "resumed"
  assert sdk.runs[0].finishes == [0] and sdk.runs[1].finishes == [1]


def test_wandb_uses_exact_resolved_directory_name_without_adding_timestamp(tmp_path, sdk):
  cfg = Config(run_name="g1_exp01")
  final_name = "g1_exp01_20261024_183045_123456"
  logger = TrainingLogger(tmp_path / final_name, cfg, ["iteration"])
  logger.close()
  assert sdk.runs[0].kwargs["name"] == final_name
  assert Path(sdk.runs[0].kwargs["dir"]).name == final_name
  assert sdk.runs[0].kwargs["config"]["run_name"] == cfg.run_name
  assert "wandb_name" not in sdk.runs[0].kwargs["config"]


def test_sdk_calls_preserve_training_random_streams(tmp_path, sdk):
  random.seed(42)
  np.random.seed(42)
  torch.manual_seed(42)
  expected = (random.random(), np.random.rand(), torch.rand(10))
  random.seed(42)
  np.random.seed(42)
  torch.manual_seed(42)
  logger = TrainingLogger(tmp_path, Config(), ["iteration", "loss"])
  logger.log({"iteration": 1, "loss": .125})
  logger.close()
  actual = (random.random(), np.random.rand(), torch.rand(10))
  assert actual[:2] == expected[:2]
  torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_logger_does_not_initialize_cuda_for_cpu_logging(tmp_path, sdk, monkeypatch):
  monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

  def forbidden():
    raise AssertionError("must not initialize CUDA for logging")

  monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden)
  logger = TrainingLogger(tmp_path, Config(), ["iteration"])
  logger.log({"iteration": 1})
  logger.close()


def test_existing_cuda_random_stream_is_restored(tmp_path, sdk, monkeypatch):
  state = {"value": torch.tensor([7], dtype=torch.uint8)}
  monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
  monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [state["value"].clone()])
  monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda saved: state.update(value=saved[0]))
  original = sdk.init

  def init(**kwargs):
    state["value"] += 1
    return original(**kwargs)

  sdk.init = init
  logger = TrainingLogger(tmp_path, Config(), ["iteration"])
  assert state["value"].item() == 7
  logger.close()


def test_initialization_failure_closes_file_and_explains_options(tmp_path, sdk):
  def fail(**kwargs):
    consume_rng()
    raise ValueError("missing credentials")

  sdk.init = fail
  logger = TrainingLogger.__new__(TrainingLogger)
  with pytest.raises(RuntimeError, match="--wandb-mode offline / --logger csv"):
    logger.__init__(tmp_path, Config(), ["iteration"])
  assert logger._file.closed


def test_partial_initialization_failure_finishes_created_run(tmp_path, sdk):
  original = sdk.init

  def init(**kwargs):
    run = original(**kwargs)
    def fail(*args, **kwargs):
      raise RuntimeError("axis definition failed")
    run.define_metric = fail
    return run

  sdk.init = init
  with pytest.raises(RuntimeError, match="axis definition failed"):
    TrainingLogger(tmp_path, Config(), ["iteration"])
  assert sdk.runs[0].finishes == [1]


def test_interrupted_initialization_closes_csv(tmp_path, sdk):
  def interrupted(**kwargs):
    raise KeyboardInterrupt()

  sdk.init = interrupted
  logger = TrainingLogger.__new__(TrainingLogger)
  with pytest.raises(KeyboardInterrupt):
    logger.__init__(tmp_path, Config(), ["iteration"])
  assert logger._file.closed


def test_wandb_failure_keeps_csv_and_close_is_idempotent(tmp_path, sdk):
  logger = TrainingLogger(tmp_path, Config(), ["iteration", "loss"])

  def fail(row):
    raise RuntimeError("upload failed")

  sdk.runs[0].log = fail
  with pytest.raises(RuntimeError, match="upload failed"):
    logger.log({"iteration": 1, "loss": .5})
  logger.close(exit_code=1)
  logger.close()
  assert sdk.runs[0].finishes == [1]
  assert read_csv(tmp_path / "train_log.csv")[1] == [{"iteration": "1", "loss": "0.5"}]
  with pytest.raises(RuntimeError, match="closed"):
    logger.log({"iteration": 2})
