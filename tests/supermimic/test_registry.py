"""Independent task selection, configuration isolation and checkpoint identity."""
import copy
from dataclasses import asdict, dataclass, fields
from types import SimpleNamespace as NS

import pytest

from supermimic.tools import registry
from supermimic.config.g1.rl_cfg import SuperMimicConfig, TerrainSuperMimicConfig
from supermimic.playback.evaluation import read_checkpoint_config
from supermimic.rl.runner import SuperMimicRunner
from test_algorithm import small_config
from test_train_resume_eval import TensorEnv
from test_visual_algorithm import visual_config
from test_visual_train_resume import VisualTensorEnv


BASE_TASK = "Unitree-G1-SuperMimic"
TERRAIN_TASK = "Unitree-G1-SuperMimic-Terrain"


def custom_env_cfg(**kwargs):
  return NS(**kwargs)


@dataclass
class CustomConfig(SuperMimicConfig):
  custom_scale: float = 0.5


def custom_runner_cfg():
  return CustomConfig(actor_kwargs={"z_dim": 16})


def test_builtin_task_defaults_are_independent():
  assert registry.list_tasks() == ('Unitree-G1-SuperMimic', 'Unitree-G1-SuperMimic-Terrain')
  task = registry.get_task(BASE_TASK)
  assert task.env_cfg_factory == '.config.g1.env_cfgs:unitree_g1_env_cfg'
  assert task.runner_cfg_factory == '.config.g1.rl_cfg:unitree_g1_runner_cfg'
  assert registry.get_task(TERRAIN_TASK).runner_cfg_factory == '.config.g1.rl_cfg:unitree_g1_terrain_runner_cfg'
  flat = registry.load_runner_cfg(BASE_TASK)
  terrain = registry.load_runner_cfg(TERRAIN_TASK)
  assert type(flat) is SuperMimicConfig
  assert type(terrain) is TerrainSuperMimicConfig
  assert isinstance(terrain, SuperMimicConfig)
  assert flat.task_id == BASE_TASK and not flat.depth_enabled
  assert asdict(flat) == asdict(SuperMimicConfig())
  assert terrain.task_id == TERRAIN_TASK and terrain.depth_enabled
  # Both tasks inherit identical common defaults; only Terrain declares vision.
  a, b = asdict(flat), asdict(terrain)
  for name in ("task_id", "depth_enabled"):
    a.pop(name)
  assert a == {name: b[name] for name in a}
  extensions = {field.name for field in fields(terrain)} - {field.name for field in fields(flat)}
  assert extensions == {
    "perception_class", "depth_decoder_class", "depth_frames", "depth_skip",
    "depth_latent_dim", "depth_frame_dim", "depth_checkpoint", "depth_batch_size",
    "latent_loss_weight", "depth_loss_weight", "wm_perception_lr", "depth_decoder_lr",
    "feet_stumble_loss_weight", "feet_at_plane_loss_weight",
  }
  assert all(not hasattr(flat, name) for name in extensions)
  assert not hasattr(flat, "depth_history_length") and terrain.depth_history_length == 36
  flat.actor_kwargs["z_dim"] = 8
  assert registry.load_runner_cfg(BASE_TASK).actor_kwargs == {}
  assert registry.load_runner_cfg(TERRAIN_TASK).actor_kwargs == {}


def test_new_task_uses_its_own_factories_without_changing_runner(monkeypatch, tmp_path):
  monkeypatch.setattr(registry, "_TASKS", dict(registry._TASKS))
  name = "Custom-SuperMimic"
  registry.register_task(name, env_cfg_factory=f"{__name__}:custom_env_cfg",
                         runner_cfg_factory=f"{__name__}:custom_runner_cfg")
  cfg = registry.load_runner_cfg(name)
  assert cfg.task_id == name and cfg.actor_kwargs == {"z_dim": 16}
  # Loading uses the registered subclass rather than G1-specific dispatch.
  import torch
  cfg.custom_scale = 1.5
  checkpoint = tmp_path / "custom.pt"
  torch.save({"config": asdict(cfg)}, checkpoint)
  loaded = read_checkpoint_config(checkpoint)
  assert type(loaded) is CustomConfig and asdict(loaded) == asdict(cfg)
  env = registry.load_env_cfg(name, play=True, collect_foot_rewards=False)
  assert env.play and not env.collect_foot_rewards
  assert registry.load_env_cfg(name, play=False) is not env
  with pytest.raises(ValueError, match="already registered"):
    registry.register_task(name, env_cfg_factory="unused", runner_cfg_factory="unused")


def test_unknown_task_is_rejected_with_available_names():
  with pytest.raises(ValueError, match=TERRAIN_TASK):
    registry.load_runner_cfg("not-a-task")


@pytest.mark.parametrize("task", [BASE_TASK, TERRAIN_TASK])
def test_checkpoint_task_roundtrip_and_resume_rejects_task_change(mdp, tmp_path, task):
  visual = task == TERRAIN_TASK
  cfg = (visual_config if visual else small_config)(task_id=task)
  env_type = VisualTensorEnv if visual else TensorEnv
  runner = SuperMimicRunner(env_type(mdp), cfg, tmp_path / "source")
  checkpoint = tmp_path / "last.pt"
  runner.save(checkpoint)
  loaded = read_checkpoint_config(checkpoint)
  assert loaded.task_id == task and type(loaded) is type(cfg)
  assert asdict(loaded) == asdict(cfg)
  resumed = SuperMimicRunner(env_type(mdp), loaded, tmp_path / "resumed")
  resumed.load(checkpoint)
  assert resumed.cfg.task_id == task
  other = copy.deepcopy(loaded)
  other.task_id = TERRAIN_TASK if task == BASE_TASK else BASE_TASK
  mismatched = SuperMimicRunner(env_type(mdp), other, tmp_path / "other")
  with pytest.raises(ValueError, match="changed task_id"):
    mismatched.load(checkpoint)


@pytest.mark.parametrize("config_type", [SuperMimicConfig, TerrainSuperMimicConfig])
@pytest.mark.parametrize("kwargs,field", [
  ({"pi_clip": 0.}, "pi_clip"), ({"state_loss_weight": -1.}, "state_loss_weight"),
  ({"reward_loss_weights": {"cost": -1.}}, "cost"),
])
def test_subclass_keeps_common_validation(config_type, kwargs, field):
  with pytest.raises(ValueError, match=field):
    config_type(**kwargs)


@pytest.mark.parametrize("kwargs,field", [
  ({"depth_frames": 0}, "depth_frames"), ({"wm_perception_lr": 0.}, "wm_perception_lr"),
  ({"latent_loss_weight": -1.}, "latent_loss_weight"),
])
def test_subclass_validates_its_own_fields(kwargs, field):
  with pytest.raises(ValueError, match=field):
    TerrainSuperMimicConfig(**kwargs)


def test_base_config_does_not_silently_accept_visual_fields():
  with pytest.raises(TypeError, match="perception_class"):
    registry.runner_cfg_from_dict({"task_id": BASE_TASK, "perception_class": "unused"})
