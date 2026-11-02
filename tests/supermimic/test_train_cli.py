"""Exercise training argument/config merging without requiring a GPU environment."""
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from supermimic.config.g1 import rl_cfg
from supermimic.tools import registry
from supermimic.rl.logging import TrainingLogger


@pytest.fixture(autouse=True)
def registry_for_cli(monkeypatch):
  # The CPU suite imports the standalone package without invoking the PPO registry.
  monkeypatch.setitem(sys.modules, "src.tasks.supermimic.tools.registry", registry)


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/train_supermimic.py"
START_TIME = datetime(2026, 10, 24, 18, 30, 45, 123456)
STAMP = START_TIME.strftime("%Y%m%d_%H%M%S_%f")
BASE_TASK = "Unitree-G1-SuperMimic"
TERRAIN_TASK = "Unitree-G1-SuperMimic-Terrain"


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
  env = SimpleNamespace(depth_enabled=True, close=lambda: captured.update(closed=True))

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
  def make_env(*args, **kwargs):
    assert kwargs['collect_foot_rewards'] is True
    return env
  config_module.make_env = make_env
  runner_module = ModuleType("src.tasks.supermimic.rl.runner")
  runner_module.SuperMimicRunner = Runner
  monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
  monkeypatch.setitem(sys.modules, "src.tasks.supermimic.config.g1.rl_cfg", rl_cfg)
  monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)
  monkeypatch.setattr(sys, "path", list(sys.path))
  args = [str(SCRIPT), "--motion-file", str(tmp_path / "motions"),
          "--log-dir", str(tmp_path / "run"), "--device", "cpu", *flags]
  if saved_setting is not None:
    saved = asdict(rl_cfg.unitree_g1_terrain_runner_cfg())
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
  assert "initial replay filling always runs" in " ".join(result.stdout.split())
  assert "--collect-det-frac" not in result.stdout
  assert "--bin-ema" not in result.stdout
  assert "--task" in result.stdout and "--profile" not in result.stdout
  assert "--depth-enabled" not in result.stdout and "--no-depth-enabled" not in result.stdout
  for option in ("--log-dir", "--run-name", "--logger", "--wandb-project", "--wandb-entity",
                 "--wandb-name", "--wandb-mode"):
    assert option in result.stdout


@pytest.fixture
def logging_cli(tmp_path, monkeypatch):
  captured = {"events": []}
  env = SimpleNamespace(close=lambda: captured["events"].append("env.close"))
  terrain_defaults = rl_cfg.unitree_g1_terrain_runner_cfg

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
    def make_env(*args, **kwargs):
      captured["env_kwargs"] = kwargs
      env.depth_enabled = kwargs['task_id'] != BASE_TASK
      return env
    config_module.make_env = make_env
    runner_module = ModuleType("src.tasks.supermimic.rl.runner")
    runner_module.SuperMimicRunner = Runner
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    monkeypatch.setitem(sys.modules, "src.tasks.supermimic.config.g1.rl_cfg", rl_cfg)
    monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)
    if factory_settings is not None:
      monkeypatch.setattr(rl_cfg, "unitree_g1_terrain_runner_cfg",
                          lambda: replace(terrain_defaults(), **factory_settings))
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


@pytest.mark.parametrize("flags,saved,settings,expected", [
  ([], None, None, (TERRAIN_TASK, True)),
  (["--task", BASE_TASK], None, None, (BASE_TASK, False)),
  (["--task", TERRAIN_TASK], None, None, (TERRAIN_TASK, True)),
  ([], None, {"task_id": BASE_TASK, "depth_enabled": False}, (BASE_TASK, False)),
  ([], {"task_id": BASE_TASK, "depth_enabled": False}, None, (BASE_TASK, False)),
  ([], {"task_id": TERRAIN_TASK, "depth_enabled": True}, None, (TERRAIN_TASK, True)),
  ([], {"depth_enabled": False}, None, (TERRAIN_TASK, True)),
  ([], None, {"depth_enabled": False}, (TERRAIN_TASK, True)),
  ([], {"depth_enabled": True}, None, (TERRAIN_TASK, True)),
  (["--task", BASE_TASK], {"depth_enabled": True}, None, (BASE_TASK, False)),
])
def test_task_cli_and_saved_configuration_select_matching_environment(
    logging_cli, flags, saved, settings, expected):
  run, captured = logging_cli
  run(flags, saved=saved, factory_settings=settings)
  assert (captured["cfg"].task_id, captured["cfg"].depth_enabled) == expected
  config_type = rl_cfg.TerrainSuperMimicConfig if expected[0] == TERRAIN_TASK else rl_cfg.SuperMimicConfig
  assert type(captured["cfg"]) is config_type
  assert captured["env_kwargs"]["task_id"] == expected[0]
  assert 'depth_enabled' not in captured["env_kwargs"]


@pytest.mark.parametrize("task", [BASE_TASK, TERRAIN_TASK])
def test_resume_keeps_registered_config_type_and_saved_fields(logging_cli, task):
  run, captured = logging_cli
  cfg = registry.load_runner_cfg(task)
  cfg.lr = 1e-4
  if task == TERRAIN_TASK:
    cfg.depth_frames = 4
    cfg.wm_perception_lr = 3e-4
    cfg.latent_loss_weight = .3
  run(["--lr", "0.00015"], saved=asdict(cfg))
  actual = captured["cfg"]
  assert type(actual) is type(cfg) and actual.lr == 1.5e-4
  if task == TERRAIN_TASK:
    assert (actual.depth_frames, actual.wm_perception_lr, actual.latent_loss_weight) == (4, 3e-4, .3)
  else:
    assert not hasattr(actual, "perception_class") and not hasattr(actual, "depth_frames")


@pytest.mark.parametrize('flag', ['--depth-enabled', '--no-depth-enabled'])
def test_removed_depth_cli_flags_do_not_override_task_sensors(logging_cli, flag):
  run, captured = logging_cli
  with pytest.raises(SystemExit) as error:
    run([flag])
  assert error.value.code == 2
  assert 'env_kwargs' not in captured and captured['events'] == []


@pytest.mark.parametrize("saved_fraction", [None, "legacy", 0., .75])
def test_collection_fraction_always_comes_from_local_config(logging_cli, saved_fraction):
  run, captured = logging_cli
  saved = None
  if saved_fraction is not None:
    saved = asdict(rl_cfg.unitree_g1_terrain_runner_cfg())
    saved["action_noise"] = .2
    if saved_fraction == "legacy":
      saved.pop("collect_det_frac")
    else:
      saved["collect_det_frac"] = saved_fraction
  run(saved=saved, factory_settings={"collect_det_frac": .4, "action_noise": .15})
  assert captured["cfg"].collect_det_frac == .4
  # Only the new collection fraction changes resume precedence.
  assert captured["cfg"].action_noise == (.15 if saved is None else .2)


@pytest.mark.parametrize("saved_ema", [None, "legacy", .2, .7])
def test_return_ema_comes_from_local_config_on_start_and_resume(logging_cli, saved_ema):
  run, captured = logging_cli
  saved = None
  if saved_ema is not None:
    saved = asdict(rl_cfg.unitree_g1_terrain_runner_cfg())
    if saved_ema == "legacy":
      saved.pop("bin_ema")
    else:
      saved["bin_ema"] = saved_ema
  run(saved=saved, factory_settings={"bin_ema": .35})
  assert captured["cfg"].bin_ema == .35


@pytest.mark.parametrize("saved_cap", [None, "legacy", 0., 50.])
@pytest.mark.parametrize("local_cap", [0., 20., 50.])
def test_softmax_cap_comes_from_rl_config_on_start_and_resume(logging_cli, saved_cap, local_cap):
  run, captured = logging_cli
  saved = None
  if saved_cap is not None:
    saved = asdict(rl_cfg.unitree_g1_terrain_runner_cfg())
    if saved_cap == "legacy":
      saved.pop("softmax_cap")
    else:
      saved["softmax_cap"] = saved_cap
  run(saved=saved, factory_settings={"softmax_cap": local_cap})
  assert captured["cfg"].softmax_cap == local_cap


@pytest.mark.parametrize("saved_bounds", [None, "legacy", (1., .1)])
def test_actor_clipping_comes_from_rl_config_on_start_and_resume(logging_cli, saved_bounds):
  run, captured = logging_cli
  saved = None
  if saved_bounds is not None:
    saved = asdict(rl_cfg.unitree_g1_terrain_runner_cfg())
    for field, value in zip(("pi_clip", "pi_clip_value"), (1., .1)):
      if saved_bounds == "legacy":
        saved.pop(field)
      else:
        saved[field] = value
  run(saved=saved, factory_settings={"pi_clip": .025, "pi_clip_value": .002})
  assert captured["cfg"].pi_clip == .025
  assert captured["cfg"].pi_clip_value == .002


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
    saved = asdict(rl_cfg.unitree_g1_terrain_runner_cfg())
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
  saved = asdict(rl_cfg.unitree_g1_terrain_runner_cfg())
  saved.update(log_dir=str(tmp_path / "saved-run"), run_name="saved-name")
  flags = ["--log-dir", str(tmp_path / "new-run"), "--run-name", "new-name"] if override else []
  run(flags, saved, factory_settings={"log_dir": "configured-dir", "run_name": "configured-name"})
  expected_dir = str(tmp_path / "new-run") if override else saved["log_dir"]
  expected_name = "new-name" if override else "saved-name"
  assert captured["cfg"].log_dir == expected_dir
  assert captured["cfg"].run_name == expected_name
  assert Path(captured["log_dir"]) == Path(expected_dir) / f"{expected_name}_{STAMP}"
  assert captured["clock_calls"] == 1


@pytest.mark.parametrize("field", ["wandb_name", "actor_perception_lr", "critic_perception_lr"])
def test_resume_rejects_removed_config_fields_before_creating_environment(logging_cli, field):
  run, captured = logging_cli
  saved = asdict(rl_cfg.unitree_g1_terrain_runner_cfg())
  saved[field] = "obsolete"
  with pytest.raises(TypeError, match=field):
    run(saved=saved)
  assert "env_kwargs" not in captured and captured["events"] == []


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
    run(saved=asdict(rl_cfg.unitree_g1_terrain_runner_cfg()), fail_at=failure)
  assert captured["events"][-1] == "env.close"
  if failure != "init":
    assert captured["events"][-2] == "runner.close"
    assert captured["exit_code"] == (0 if failure == "close" else 1)


@pytest.mark.parametrize("field,value", [("logger", "tensorboard"), ("wandb_mode", "disabled")])
def test_invalid_logging_config(field, value):
  with pytest.raises(ValueError, match=field):
    rl_cfg.SuperMimicConfig(**{field: value})
