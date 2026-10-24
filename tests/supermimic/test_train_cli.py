"""Exercise training argument/config merging without requiring a GPU environment."""
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from supermimic.config.g1 import rl_cfg
from supermimic.rl.logging import TrainingLogger


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/train_supermimic.py"
START_TIME = datetime(2026, 10, 24, 18, 30, 45, 123456)
STAMP = START_TIME.strftime("%Y%m%d_%H%M%S_%f")


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
  for option in ("--log-dir", "--run-name", "--logger", "--wandb-project", "--wandb-entity",
                 "--wandb-name", "--wandb-mode"):
    assert option in result.stdout


@pytest.fixture
def logging_cli(tmp_path, monkeypatch):
  captured = {"events": []}
  env = SimpleNamespace(close=lambda: captured["events"].append("env.close"))

  def run(flags=(), saved=None, fail_at=None, factory_settings=None, use_logging_backend=False,
          start_time=START_TIME):
    captured["clock_calls"] = 0

    class Clock:
      @staticmethod
      def now():
        captured["clock_calls"] += 1
        return start_time

    class Runner:
      def __init__(self, environment, cfg, log_dir):
        assert environment is env
        captured["cfg"] = cfg
        captured["log_dir"] = log_dir
        self.logger = None
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
        if use_logging_backend:
          self.logger = TrainingLogger(captured["log_dir"], captured["cfg"], ["iteration"])
          self.logger.log({"iteration": 50})
          torch.save({"config": asdict(captured["cfg"])}, Path(captured["log_dir"]) / "last.pt")

      def close(self, exit_code=0):
        captured["events"].append("runner.close")
        captured["exit_code"] = exit_code
        if self.logger is not None:
          self.logger.close(exit_code=exit_code)
        if fail_at == "close":
          raise RuntimeError("close failed")

    config_module = ModuleType("src.tasks.supermimic.config")
    config_module.make_env = lambda *args, **kwargs: env
    runner_module = ModuleType("src.tasks.supermimic.rl.runner")
    runner_module.SuperMimicRunner = Runner
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    monkeypatch.setitem(sys.modules, "src.tasks.supermimic.config.g1.rl_cfg", rl_cfg)
    monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)
    if factory_settings is not None:
      monkeypatch.setattr(rl_cfg, "unitree_g1_supermimic_runner_cfg",
                          lambda: rl_cfg.SuperMimicConfig(**factory_settings))
    monkeypatch.setattr(sys, "path", list(sys.path))
    args = [str(SCRIPT), "--motion-file", str(tmp_path / "motions"),
            "--device", "cpu", *flags]
    if saved is not None:
      checkpoint = tmp_path / "last.pt"
      torch.save({"config": saved}, checkpoint)
      args.extend(["--resume", str(checkpoint)])
    monkeypatch.setattr(sys, "argv", args)
    main = runpy.run_path(str(SCRIPT))["main"]
    main.__globals__["datetime"] = Clock
    main()
    return captured

  return run, captured


@pytest.mark.parametrize("flags,saved_settings,expected", [
  ([], None, ("wandb", "unitree-supermimic", None, "g1", "online")),
  (["--logger", "csv"], None, ("csv", "unitree-supermimic", None, "g1", "online")),
  (["--wandb-mode", "offline"], None, ("wandb", "unitree-supermimic", None, "g1", "offline")),
  (["--wandb-project", "experiment", "--wandb-entity", "robotics", "--wandb-name", "g1"],
   None, ("wandb", "experiment", "robotics", "g1", "online")),
  ([], {}, ("wandb", "unitree-supermimic", None, "g1", "online")),
  ([], {"logger": "csv", "wandb_project": "saved", "wandb_mode": "offline"},
   ("csv", "saved", None, "g1", "offline")),
  (["--logger", "wandb", "--wandb-mode", "online", "--wandb-name", "new-run"],
   {"logger": "csv", "wandb_project": "saved", "wandb_mode": "offline"},
   ("wandb", "saved", None, "new-run", "online")),
])
def test_logging_cli_defaults_and_resume(logging_cli, flags, saved_settings, expected):
  run, captured = logging_cli
  fields = ("logger", "wandb_project", "wandb_entity", "run_name", "wandb_mode")
  saved = None
  if saved_settings is not None:
    saved = asdict(rl_cfg.SuperMimicConfig())
    for field in fields:
      saved.pop(field)
    saved.update(saved_settings)
  run(flags, saved)
  assert tuple(getattr(captured["cfg"], field) for field in fields) == expected
  assert captured["cfg"].log_dir == "logs/supermimic"
  assert Path(captured["log_dir"]) == Path("logs/supermimic") / f"{expected[3]}_{STAMP}"
  assert captured["clock_calls"] == 1
  assert captured["events"] == (["load"] if saved is not None else []) + [
    "learn", "runner.close", "env.close"]
  assert captured["exit_code"] == 0


@pytest.mark.parametrize("name_flag", [None, "--run-name", "--wandb-name"])
def test_run_directory_and_name_use_factory_config_with_cli_overrides(
    logging_cli, tmp_path, name_flag):
  run, captured = logging_cli
  configured_dir = str(tmp_path / "configured-run")
  selected_dir = str(tmp_path / "cli-run") if name_flag else configured_dir
  flags = ["--log-dir", selected_dir, name_flag, "cli-name"] if name_flag else []
  run(flags, factory_settings={"log_dir": configured_dir, "run_name": "configured-name"})
  expected_name = "cli-name" if name_flag else "configured-name"
  assert captured["cfg"].log_dir == selected_dir
  assert captured["cfg"].run_name == expected_name
  assert Path(captured["log_dir"]) == Path(selected_dir) / f"{expected_name}_{STAMP}"
  assert captured["clock_calls"] == 1


@pytest.mark.parametrize("override", [False, True])
def test_resume_run_directory_and_name_with_cli_overrides(logging_cli, tmp_path, override):
  run, captured = logging_cli
  saved = asdict(rl_cfg.SuperMimicConfig())
  saved.update(log_dir=str(tmp_path / "saved-run"), run_name="saved-name")
  flags = ["--log-dir", str(tmp_path / "new-run"), "--run-name", "new-name"] if override else []
  run(flags, saved, factory_settings={"log_dir": "configured-dir", "run_name": "configured-name"})
  expected_dir = str(tmp_path / "new-run") if override else saved["log_dir"]
  expected_name = "new-name" if override else "saved-name"
  assert captured["cfg"].log_dir == expected_dir
  assert captured["cfg"].run_name == expected_name
  assert Path(captured["log_dir"]) == Path(expected_dir) / f"{expected_name}_{STAMP}"
  assert captured["clock_calls"] == 1


@pytest.mark.parametrize("saved_names,flags,expected_name", [
  ({}, [], "configured-name"),
  ({"wandb_name": None}, [], "configured-name"),
  ({"wandb_name": ""}, [], "configured-name"),
  ({"wandb_name": "legacy-name"}, [], "legacy-name"),
  ({"wandb_name": "legacy-name", "run_name": "saved-name"}, [], "saved-name"),
  ({"wandb_name": "legacy-name"}, ["--run-name", "cli-name"], "cli-name"),
  ({"wandb_name": "legacy-name"}, ["--wandb-name", "cli-name"], "cli-name"),
])
def test_resume_legacy_run_configuration(logging_cli, tmp_path, saved_names, flags, expected_name):
  run, captured = logging_cli
  saved = asdict(rl_cfg.SuperMimicConfig())
  saved.pop("log_dir")
  saved.pop("run_name")
  saved.update(saved_names)
  configured_dir = str(tmp_path / "configured-run")
  run(flags, saved, factory_settings={"log_dir": configured_dir, "run_name": "configured-name"})
  assert captured["cfg"].log_dir == configured_dir
  assert Path(captured["log_dir"]) == Path(configured_dir) / f"{expected_name}_{STAMP}"
  assert captured["cfg"].run_name == expected_name
  assert not hasattr(captured["cfg"], "wandb_name")
  assert captured["clock_calls"] == 1


def test_cli_directory_and_wandb_share_one_timestamp_and_resume_cleanly(logging_cli, tmp_path, monkeypatch):
  """The real CLI and logger share one name, while persisted config stays reusable."""
  calls = []
  sdk = SimpleNamespace(Settings=lambda **kwargs: kwargs)

  def init(**kwargs):
    calls.append(kwargs)
    return SimpleNamespace(define_metric=lambda *args, **kwargs: None,
                           log=lambda row: None, finish=lambda **kwargs: None)

  sdk.init = init
  monkeypatch.setitem(sys.modules, "wandb", sdk)
  run, captured = logging_cli
  root = str(tmp_path / "logs")
  run(factory_settings={"log_dir": root, "run_name": "experiment"},
      use_logging_backend=True)
  expected_name = f"experiment_{STAMP}"
  expected_dir = Path(root) / expected_name
  assert captured["clock_calls"] == 1
  assert Path(captured["log_dir"]) == expected_dir
  assert calls[0]["name"] == expected_dir.name
  assert Path(calls[0]["dir"]) == expected_dir
  assert (expected_dir / "train_log.csv").exists()
  persisted = torch.load(expected_dir / "last.pt", weights_only=False)["config"]
  assert persisted["log_dir"] == calls[0]["config"]["log_dir"] == root
  assert persisted["run_name"] == calls[0]["config"]["run_name"] == "experiment"
  assert expected_dir.name.count(STAMP) == 1

  resumed_time = datetime(2026, 10, 24, 19, 30, 45, 987654)
  run(saved=persisted, start_time=resumed_time, use_logging_backend=True)
  resumed_name = f"experiment_{resumed_time:%Y%m%d_%H%M%S_%f}"
  assert captured["clock_calls"] == 1
  assert Path(captured["log_dir"]) == Path(root) / resumed_name
  assert calls[1]["name"] == resumed_name
  assert calls[1]["config"]["run_name"] == "experiment"
  assert (expected_dir / "last.pt").exists()
  assert (Path(root) / resumed_name / "last.pt").exists()


@pytest.mark.parametrize("name", ["", " ", ".", "..", "nested/run", "nested\\run", "/absolute"])
def test_run_name_must_be_one_directory_component(name):
  with pytest.raises(ValueError, match="run_name"):
    rl_cfg.SuperMimicConfig(run_name=name)


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
