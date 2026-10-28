"""G1 defaults for the independent SuperMimic learner."""
from dataclasses import dataclass
import math
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
  pi_clip: float = 1.0 / 32  # Actor global gradient norm, after loss / horizon.
  pi_clip_value: float = 0.1 / 32  # Actor per-element gradient bound.
  lr: float = 2e-4
  depth_enabled: bool = True
  depth_frames: int = 8
  depth_skip: int = 5
  depth_latent_dim: int = 256
  depth_frame_dim: int = 256
  depth_checkpoint: bool = True  # Recompute visual activations during BPTT to save memory.
  depth_batch_size: int = 256  # Bound CNN workspace for large critic minibatches.
  depth_loss_weight: float = 1.0  # Normalized pixel MSE, summed over valid WM steps.
  actor_perception_lr: float = 2e-4
  wm_perception_lr: float = 2e-4
  critic_perception_lr: float = 2e-4
  depth_decoder_lr: float = 2e-4
  gamma: float = 0.99
  lam: float = 0.95
  action_noise: float = 0.1  # Gaussian std for environments with exploration noise.
  collect_det_frac: float = 0.25  # First round(frac * num_envs) collect without action noise.
  bin_ema: float = 0.1  # New return-score weight: (1 - bin_ema) * old + bin_ema * new.
  softmax_cap: float = 50.0  # Cap current softmax at this multiple of 1 / num_bins; 0 disables.
  critic_epochs: int = 5
  critic_minibatches: int = 4
  enable_warmup: bool = True  # Controls only the WM-only phase, not replay filling.
  warmup_steps: int = 64  # Initial replay fill for both fresh and resumed runs.
  wm_only_iters: int = 500  # Masked windows here; full same-episode windows afterward.
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
    for name in ("depth_frames", "depth_skip", "depth_latent_dim", "depth_frame_dim", "depth_batch_size"):
      if getattr(self, name) < 1:
        raise ValueError(f"{name} must be positive")
    for name in ("actor_perception_lr", "wm_perception_lr", "critic_perception_lr",
                 "depth_decoder_lr", "depth_loss_weight"):
      if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
        raise ValueError(f"{name} must be finite and positive")
    for name in ("pi_clip", "pi_clip_value"):
      if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
        raise ValueError(f"{name} must be finite and positive")
    if self.utd <= 0 or not 0 < self.gamma < 1:
      raise ValueError("utd must be positive and gamma must lie in (0, 1)")
    if not 0 <= self.collect_det_frac <= 1:
      raise ValueError("collect_det_frac must lie in [0, 1]")
    if not 0 <= self.bin_ema <= 1:
      raise ValueError("bin_ema must lie in [0, 1]")
    if not math.isfinite(self.softmax_cap) or (self.softmax_cap != 0 and self.softmax_cap < 1):
      raise ValueError("softmax_cap must be 0 (disabled) or finite and at least 1")
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

  @property
  def depth_history_length(self):
    return (self.depth_frames - 1) * self.depth_skip + 1


def migrate_legacy_run_name(config: dict) -> dict:
  """Read old checkpoint names without keeping a second naming option."""
  config = dict(config)
  legacy_name = config.pop("wandb_name", None)
  if "run_name" not in config and legacy_name:
    config["run_name"] = legacy_name
  return config


def unitree_g1_supermimic_runner_cfg() -> SuperMimicConfig:
  return SuperMimicConfig()


def migrate_checkpoint_config(config: dict) -> dict:
  """Old checkpoints predate vision; never reinterpret their weights as visual."""
  config = migrate_legacy_run_name(config)
  config.setdefault("depth_enabled", False)
  return config
