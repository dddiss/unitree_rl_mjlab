"""Exercise training argument/config merging without requiring a GPU environment."""
from dataclasses import asdict
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from supermimic.config.g1 import rl_cfg


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/train_supermimic.py"


@pytest.mark.parametrize("saved_setting,flags,expected", [
  (None, [], True),
  (None, ["--enable-warmup"], True),
  (None, ["--no-enable-warmup"], False),
  (False, [], False),
  (False, ["--enable-warmup"], True),
  (True, ["--no-enable-warmup"], False),
  ("legacy", [], True),
  ("legacy", ["--no-enable-warmup"], False),
])
def test_warmup_cli_defaults_and_resume_overrides(tmp_path, monkeypatch, saved_setting, flags, expected):
  captured = {}
  env = SimpleNamespace(close=lambda: captured.update(closed=True))

  class Runner:
    def __init__(self, environment, cfg, log_dir):
      assert environment is env
      captured["cfg"] = cfg

    def load(self, path):
      captured["checkpoint"] = path

    def learn(self):
      captured["learned"] = True

    def close(self, exit_code=0):
      captured["exit_code"] = exit_code

  # Keep the real CLI and configuration; replace only environment/runner startup.
  config_module = ModuleType("src.tasks.supermimic.config")
  config_module.make_env = lambda *args, **kwargs: env
  runner_module = ModuleType("src.tasks.supermimic.rl.runner")
  runner_module.SuperMimicRunner = Runner
  monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
  monkeypatch.setitem(sys.modules, "src.tasks.supermimic.config.g1.rl_cfg", rl_cfg)
  monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)
  monkeypatch.setattr(sys, "path", list(sys.path))
  args = [str(SCRIPT), "--motion-file", str(tmp_path / "motions"),
          "--log-dir", str(tmp_path / "run"), "--device", "cpu", *flags]
  if saved_setting is not None:
    saved = asdict(rl_cfg.SuperMimicConfig())
    saved["warmup_steps"] = 17
    if saved_setting == "legacy":
      saved.pop("enable_warmup")
    else:
      saved["enable_warmup"] = saved_setting
    checkpoint = tmp_path / "last.pt"
    torch.save({"config": saved}, checkpoint)
    args.extend(["--resume", str(checkpoint)])
  monkeypatch.setattr(sys, "argv", args)
  runpy.run_path(str(SCRIPT))["main"]()
  assert captured["cfg"].enable_warmup is expected
  assert captured["cfg"].warmup_steps == (64 if saved_setting is None else 17)
  assert captured["cfg"].wm_only_iters == 500
  assert captured["learned"] and captured["closed"]
  assert captured["exit_code"] == 0
  if saved_setting is not None:
    assert captured["checkpoint"] == str(checkpoint)


def test_warmup_cli_help():
  result = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                          capture_output=True, text=True, check=True)
  assert "--enable-warmup" in result.stdout
  assert "--no-enable-warmup" in result.stdout
  for option in ("--logger", "--wandb-project", "--wandb-entity", "--wandb-name", "--wandb-mode"):
    assert option in result.stdout


@pytest.fixture
def logging_cli(tmp_path, monkeypatch):
  captured = {"events": []}
  env = SimpleNamespace(close=lambda: captured["events"].append("env.close"))

  def run(flags=(), saved=None, fail_at=None):
    class Runner:
      def __init__(self, environment, cfg, log_dir):
        assert environment is env
        captured["cfg"] = cfg
        if fail_at == "init":
          raise RuntimeError("init failed")

      def load(self, path):
        captured["events"].append("load")
        if fail_at == "load":
          raise RuntimeError("load failed")

      def learn(self):
        captured["events"].append("learn")
        if fail_at == "learn":
          raise RuntimeError("learn failed")

      def close(self, exit_code=0):
        captured["events"].append("runner.close")
        captured["exit_code"] = exit_code
        if fail_at == "close":
          raise RuntimeError("close failed")

    config_module = ModuleType("src.tasks.supermimic.config")
    config_module.make_env = lambda *args, **kwargs: env
    runner_module = ModuleType("src.tasks.supermimic.rl.runner")
    runner_module.SuperMimicRunner = Runner
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    monkeypatch.setitem(sys.modules, "src.tasks.supermimic.config.g1.rl_cfg", rl_cfg)
    monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)
    monkeypatch.setattr(sys, "path", list(sys.path))
    args = [str(SCRIPT), "--motion-file", str(tmp_path / "motions"),
            "--log-dir", str(tmp_path / "run"), "--device", "cpu", *flags]
    if saved is not None:
      checkpoint = tmp_path / "last.pt"
      torch.save({"config": saved}, checkpoint)
      args.extend(["--resume", str(checkpoint)])
    monkeypatch.setattr(sys, "argv", args)
    runpy.run_path(str(SCRIPT))["main"]()
    return captured

  return run, captured


@pytest.mark.parametrize("flags,saved_settings,expected", [
  ([], None, ("wandb", "unitree-supermimic", None, None, "online")),
  (["--logger", "csv"], None, ("csv", "unitree-supermimic", None, None, "online")),
  (["--wandb-mode", "offline"], None, ("wandb", "unitree-supermimic", None, None, "offline")),
  (["--wandb-project", "experiment", "--wandb-entity", "robotics", "--wandb-name", "g1"],
   None, ("wandb", "experiment", "robotics", "g1", "online")),
  ([], {}, ("wandb", "unitree-supermimic", None, None, "online")),
  ([], {"logger": "csv", "wandb_project": "saved", "wandb_mode": "offline"},
   ("csv", "saved", None, None, "offline")),
  (["--logger", "wandb", "--wandb-mode", "online", "--wandb-name", "new-run"],
   {"logger": "csv", "wandb_project": "saved", "wandb_mode": "offline"},
   ("wandb", "saved", None, "new-run", "online")),
])
def test_logging_cli_defaults_and_resume(logging_cli, flags, saved_settings, expected):
  run, captured = logging_cli
  fields = ("logger", "wandb_project", "wandb_entity", "wandb_name", "wandb_mode")
  saved = None
  if saved_settings is not None:
    saved = asdict(rl_cfg.SuperMimicConfig())
    for field in fields:
      saved.pop(field)
    saved.update(saved_settings)
  run(flags, saved)
  assert tuple(getattr(captured["cfg"], field) for field in fields) == expected
  assert captured["events"] == (["load"] if saved is not None else []) + [
    "learn", "runner.close", "env.close"]
  assert captured["exit_code"] == 0


@pytest.mark.parametrize("failure", ["init", "load", "learn", "close"])
def test_training_cleanup_on_failure(logging_cli, failure):
  run, captured = logging_cli
  with pytest.raises(RuntimeError, match=f"{failure} failed"):
    run(saved=asdict(rl_cfg.SuperMimicConfig()), fail_at=failure)
  assert captured["events"][-1] == "env.close"
  if failure != "init":
    assert captured["events"][-2] == "runner.close"
    assert captured["exit_code"] == (0 if failure == "close" else 1)


@pytest.mark.parametrize("field,value", [("logger", "tensorboard"), ("wandb_mode", "disabled")])
def test_invalid_logging_config(field, value):
  with pytest.raises(ValueError, match=field):
    rl_cfg.SuperMimicConfig(**{field: value})
