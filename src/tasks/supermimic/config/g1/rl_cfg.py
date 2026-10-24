"""G1 defaults for the independent SuperMimic learner."""
from dataclasses import dataclass
from typing import Literal


@dataclass
class SuperMimicConfig:
  num_envs: int = 4096
  iterations: int = 320_000
  buffer_steps: int = 1000
  utd: float = 0.4
  horizon: int = 32
  goal_frames: int = 10
  hist: int = 10
  wm_window: int = 8
  wm_batches: int = 10
  wm_batch: int = 2048
  pi_batch: int = 4096
  lr: float = 2e-4
  gamma: float = 0.99
  lam: float = 0.95
  action_noise: float = 0.1
  critic_epochs: int = 5
  critic_minibatches: int = 4
  enable_warmup: bool = True
  warmup_steps: int = 64
  wm_only_iters: int = 500
  lr_decay_iters: int = 250_000
  ckpt_every: int = 2000
  log_every: int = 50
  log_dir: str = "logs/supermimic"  # Root for timestamped experiment directories.
  run_name: str = "g1"  # Base name; startup appends one timestamp for directory/W&B.
  logger: Literal["wandb", "csv"] = "wandb"
  wandb_project: str = "unitree-supermimic"
  wandb_entity: str | None = None
  wandb_mode: Literal["online", "offline"] = "online"
  seed: int = 0
  amp: str = "bf16"
  compile: bool = True
  wm_hidden: tuple[int, ...] = (2048, 2048, 2048)
  policy_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
  critic_hidden: tuple[int, ...] = (256, 128)

  def __post_init__(self):
    for name in ("num_envs", "buffer_steps", "horizon", "goal_frames", "hist", "wm_window",
                 "wm_batches", "wm_batch", "pi_batch", "critic_epochs", "critic_minibatches",
                 "warmup_steps", "lr_decay_iters", "ckpt_every", "log_every"):
      if getattr(self, name) < 1:
        raise ValueError(f"{name} must be positive")
    if self.buffer_steps < self.wm_window:
      raise ValueError("buffer_steps must cover wm_window")
    if self.utd <= 0 or not 0 < self.gamma < 1:
      raise ValueError("utd must be positive and gamma must lie in (0, 1)")
    if self.amp not in ("none", "bf16"):
      raise ValueError("amp must be none or bf16")
    if self.logger not in ("wandb", "csv"):
      raise ValueError("logger must be wandb or csv")
    if self.wandb_mode not in ("online", "offline"):
      raise ValueError("wandb_mode must be online or offline")
    if not self.log_dir:
      raise ValueError("log_dir must not be empty")
    if not self.run_name or not self.run_name.strip():
      raise ValueError("run_name must not be empty")
    if self.run_name in (".", "..") or "/" in self.run_name or "\\" in self.run_name:
      raise ValueError("run_name must be a single directory name")


def migrate_legacy_run_name(config: dict) -> dict:
  """Read old checkpoint names without keeping a second naming option."""
  config = dict(config)
  legacy_name = config.pop("wandb_name", None)
  if "run_name" not in config and legacy_name:
    config["run_name"] = legacy_name
  return config


def unitree_g1_supermimic_runner_cfg() -> SuperMimicConfig:
  return SuperMimicConfig()
