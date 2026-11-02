"""CSV, optional W&B and terminal logging for the SuperMimic learner."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import csv
from dataclasses import asdict, is_dataclass
import math
from numbers import Real
import os
from pathlib import Path
import random
import tempfile
import uuid

import numpy as np
import torch


WANDB_METRIC_NAMES = {
  "wm_loss": "Loss/wm_loss",
  "wm_err_last": "Loss/wm_err_last",
  "wm_depth_loss": "Loss/wm_depth_loss",
  "wm_latent_loss": "Loss/wm_latent_loss",
  "wm_total_loss": "Loss/wm_total_loss",
  "wm_feet_stumble_loss": "Loss/wm_feet_stumble_loss",
  "wm_feet_at_plane_loss": "Loss/wm_feet_at_plane_loss",
  "pi_loss": "Loss/pi_loss",
  "v_loss": "Loss/v_loss",
  "lr": "Training/lr",
  "pi_gradnorm": "Training/pi_gradnorm",
  "policy_skipped": "Training/policy_skipped",
  "sim_steps": "Training/sim_steps",
  "env_steps": "Training/env_steps",
  "wall_h": "Performance/wall_h",
  "imag_reward": "Imagination/reward",
  "imag_survival": "Imagination/survival",
  "sim_reward": "Reward/total_normalized",
  "episode_length": "Episode/length_steps",
  "episode_return": "Episode/return_normalized",
  "failure_fraction": "Episode/failure_fraction",
}


def format_terminal_summary(
  row: dict, *, total_iterations: int, interval_iterations: int,
  interval_seconds: float, wm_warmup: bool = False,
) -> str:
  """Format existing scalar metrics without changing the CSV/W&B payload."""
  def value(number, spec=".4f", suffix=""):
    if number is None or number == "":
      return "--"
    return f"{number:{spec}}{suffix}"

  def duration(seconds):
    if seconds is None or not math.isfinite(seconds):
      return "--"
    days, remainder = divmod(max(0, int(seconds)), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}d {clock}" if days else clock

  width = 80
  iteration = row["iteration"]
  first = iteration - interval_iterations + 1
  window = f"iterations {first}–{iteration}" if interval_iterations else "no completed iterations"
  lines = ["#" * width, f"Learning iteration {iteration}/{total_iterations}".center(width),
           f"SuperMimic | {window}".center(width), "-" * width]

  def add(label, formatted):
    lines.append(f"{label + ':':>40} {formatted}")

  speed = row.get("Performance/iterations_per_s")
  add("Computation", value(row.get("Performance/transitions_per_s"), ",.0f", " transitions/s")
      + " | " + value(speed, ".2f", " iterations/s"))
  add("Interval time", value(interval_seconds, ".3f", " s"))
  mean_seconds = interval_seconds / interval_iterations if interval_iterations else None
  add("Mean iteration time", value(mean_seconds, ".3f", " s"))
  for label, key in (
    ("Collection time", "Performance/collection_s"),
    ("WM update time", "Performance/wm_update_s"),
    ("Policy+Critic update time", "Performance/policy_critic_update_s"),
  ):
    add(label, value(row.get(key), ".3f", " s/iteration"))

  lines.append("-" * width)
  for key in ("wm_loss", "wm_err_last", "pi_loss", "v_loss"):
    add(WANDB_METRIC_NAMES[key], value(row.get(key), ".6f"))
  visual_fields = ("wm_latent_loss", "wm_depth_loss")
  reward_fields = tuple(key for key in row if key.startswith("wm_") and key.endswith("_loss")
                       and key not in (*visual_fields, "wm_total_loss", "wm_loss"))
  for key in (*visual_fields, *reward_fields, "wm_total_loss"):
    if key in row:
      add("Loss/" + key, value(row.get(key), ".6f"))
  add("Training/lr", value(row.get("lr"), ".3e"))
  add("Episode/length_steps", value(row.get("episode_length"), ".2f", " steps"))

  lines.append("-" * width)
  for key, number in row.items():
    if key.startswith("Reward/") and key != "Reward/total_normalized":
      add(key, value(number))

  lines.append("-" * width)
  for key, number in row.items():
    if key.startswith("Termination/") and key.endswith("_fraction"):
      add(key, value(number, ".2%"))

  lines.append("-" * width)
  add("Total transitions", value(row.get("sim_steps"), ",.0f"))
  wall_h = row.get("wall_h")
  add("Total elapsed", duration(None if wall_h is None or wall_h == "" else wall_h * 3600))
  remaining = max(0, total_iterations - iteration)
  eta = None
  if not remaining:
    eta = 0
  elif not wm_warmup and speed is not None and speed != "" and speed > 0:
    eta = remaining / speed
  add("ETA (estimated)", duration(eta))
  lines.append("#" * width)
  return "\n".join(lines)


@contextmanager
def _preserve_training_rng():
  """Keep SDK imports and calls out of the learner's random number streams."""
  python_state = random.getstate()
  numpy_state = np.random.get_state()
  torch_state = torch.random.get_rng_state()
  # Do not initialize CUDA just for logging (including CPU/offline tests).
  cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
  try:
    yield
  finally:
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(torch_state)
    if cuda_state is not None:
      torch.cuda.set_rng_state_all(cuda_state)


def _extend_csv_header(path: Path, fieldnames: list[str]) -> list[str]:
  """Atomically add columns while preserving historical cell values and order."""
  if not path.exists() or path.stat().st_size == 0:
    return list(fieldnames)
  with path.open(newline="", encoding="utf-8") as source:
    reader = csv.DictReader(source)
    previous = reader.fieldnames or []
    merged = list(dict.fromkeys([*previous, *fieldnames]))
    if merged == previous:
      return merged
    temporary = None
    try:
      with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
      ) as output:
        temporary = Path(output.name)
        writer = csv.DictWriter(output, fieldnames=merged)
        writer.writeheader()
        writer.writerows(reader)
        output.flush()
        os.fsync(output.fileno())
      os.replace(temporary, path)
    finally:
      if temporary is not None:
        temporary.unlink(missing_ok=True)
  return merged


class TrainingLogger:
  """Write one scalar payload locally first, then to a fresh W&B run.

  Construct after checkpoint loading so ``start_iteration`` and
  ``resume_checkpoint`` describe this invocation. The caller owns closing the
  logger on both successful and exceptional training exits.
  """

  def __init__(
    self, log_dir: str | Path, cfg, fieldnames, *,
    resume_checkpoint: str | Path | None = None, start_iteration: int = 0,
  ):
    backend = cfg.logger
    if backend not in ("csv", "wandb"):
      raise ValueError("logger must be csv or wandb")
    self.log_dir = Path(log_dir)
    self.log_dir.mkdir(parents=True, exist_ok=True)
    self.csv_path = self.log_dir / "train_log.csv"
    self.fieldnames = _extend_csv_header(self.csv_path, list(fieldnames))
    self._file = self.csv_path.open("a", newline="", encoding="utf-8")
    self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
    if self._file.tell() == 0:
      self._writer.writeheader()
      self._file.flush()
    self._run = None
    self._closed = False
    if backend == "csv":
      return
    try:
      with _preserve_training_rng():
        import wandb

        config = asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))
        config["resume_checkpoint"] = (
          str(Path(resume_checkpoint).resolve()) if resume_checkpoint is not None else None
        )
        config["start_iteration"] = start_iteration
        self._run = wandb.init(
          project=cfg.wandb_project, entity=cfg.wandb_entity, name=self.log_dir.name,
          mode=cfg.wandb_mode, dir=str(self.log_dir), config=config,
          id=uuid.uuid4().hex, resume="never", reinit="create_new",
          save_code=False, sync_tensorboard=False,
          settings=wandb.Settings(
            console="auto", disable_code=True, disable_git=True,
            x_disable_stats=True, x_disable_meta=True,
          ),
        )
        self._run.define_metric("iteration")
        self._run.define_metric("*", step_metric="iteration")
    except BaseException as exc:
      try:
        self.close(exit_code=1)
      except Exception:
        pass  # Preserve the original initialization error.
      if not isinstance(exc, Exception):
        raise
      raise RuntimeError(
        "W&B initialization failed. Check your W&B credentials and connection, "
        "or use --wandb-mode offline / --logger csv. "
        f"Original error: {exc}"
      ) from exc

  def log(self, row: dict) -> None:
    """Preserve CSV names; group W&B metrics and omit blank/missing values."""
    if self._closed:
      raise RuntimeError("TrainingLogger is closed")
    values = {}
    payload = {}
    for name, value in row.items():
      if value is None or (isinstance(value, str) and value == ""):
        values[name] = ""
        continue
      if isinstance(value, np.generic):
        value = value.item()
      if not isinstance(value, Real):
        raise TypeError(f"{name}: training log values must be numeric scalars or blank")
      values[name] = value
      metric_name = WANDB_METRIC_NAMES.get(name, "Loss/" + name if name.startswith("wm_") and name.endswith("_loss") else name)
      payload[metric_name] = value
    self._writer.writerow(values)
    self._file.flush()
    if self._run is not None:
      with _preserve_training_rng():
        self._run.log(payload)

  def close(self, exit_code: int = 0) -> None:
    if self._closed:
      return
    self._closed = True
    try:
      if self._run is not None:
        with _preserve_training_rng():
          self._run.finish(exit_code=exit_code)
    finally:
      self._file.close()


# Step snapshots are captured before command advance/reset.
TRACKING_METRICS = (
  "error_anchor_pos", "error_anchor_rot", "error_anchor_lin_vel", "error_anchor_ang_vel",
  "error_body_pos", "error_body_rot", "error_body_lin_vel", "error_body_ang_vel",
  "error_joint_pos", "error_joint_vel",
)


@torch.no_grad()
def capture_step_metrics(
  command,
  raw_reward_terms: Mapping[str, torch.Tensor],
  reward_weights: Mapping[str, float],
  termination_terms: Mapping[str, torch.Tensor],
  step_dt: float,
  *,
  reward_contributions: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
  """Capture per-environment values before advancing targets or resetting.

  The command already has post-physics relative body poses. Its metric update
  only refreshes diagnostic tensors; it neither advances frames nor samples.
  All returned tensors own their storage, so later command/reset writes cannot
  change a transition's log values. Reward scaling matches mjlab 1.6.0's
  default RewardManager, including zeroing non-finite weighted contributions.
  """
  command._update_metrics()
  return {
    "tracking": {name: command.metrics[name].detach().clone() for name in TRACKING_METRICS},
    # The environment supplies the mjlab 1.6 manager's already weighted/sanitized
    # cache. This also covers measured terms absent from the analytic MDP and
    # zero-weight terms whose sensor/function was deliberately not evaluated.
    "reward": ({name: value.detach().clone() for name, value in reward_contributions.items()}
               if reward_contributions is not None else {
      name: torch.nan_to_num(raw_reward_terms[name].detach() * weight * step_dt,
                             nan=0.0, posinf=0.0, neginf=0.0)
      for name, weight in reward_weights.items()
    }),
    "termination": {name: value.detach().clone() for name, value in termination_terms.items()},
  }


LEGACY_FIELDS = (
  "iteration", "wall_h", "sim_steps", "env_steps", "lr",
  "wm_loss", "wm_err_last", "pi_loss", "v_loss", "pi_gradnorm",
  "imag_survival", "imag_reward", "policy_skipped", "sim_reward",
  "episode_length", "episode_return", "failure_fraction",
)
SAMPLING_FIELDS = ("Sampling/entropy", "Sampling/top1_prob", "Sampling/top1_bin")
PERFORMANCE_FIELDS = (
  "Performance/collection_s", "Performance/wm_update_s", "Performance/policy_critic_update_s",
  "Performance/iterations_per_s", "Performance/transitions_per_s",
)


class IntervalMetrics:
  """Accumulate on device, transferring scalar reductions only when logging.

  Episode sums belong to the runner and survive interval boundaries. This
  object only receives completed episodes, so unequal completion batches do
  not bias their means. A reset can match multiple termination reasons.
  """
  def __init__(self, reward_names, termination_names, step_dt, device):
    self.reward_names = tuple(reward_names)
    self.termination_names = tuple(termination_names)
    self.step_dt = step_dt
    self.step_fields = tuple(f"Tracking/{key}" for key in TRACKING_METRICS) + tuple(
      f"Reward/{key}" for key in self.reward_names)
    self.fields = self.step_fields + tuple(
      f"Termination/{key}_fraction" for key in self.termination_names
    ) + ("Episode/completed", "Episode/return_raw", "Episode/length_seconds")
    self.step_sums = torch.zeros(len(self.step_fields), dtype=torch.float64, device=device)
    self.reason_counts = torch.zeros(len(self.termination_names), dtype=torch.long, device=device)
    self.episode_sums = torch.zeros(2, dtype=torch.float64, device=device)
    self.sample_count = self.completed = 0

  @torch.no_grad()
  def add_step(self, snapshot):
    values = [snapshot["tracking"][key] for key in TRACKING_METRICS]
    values += [snapshot["reward"][key] for key in self.reward_names]
    self.step_sums.add_(torch.stack([v.sum(dtype=torch.float64) for v in values]))
    self.sample_count += values[0].numel()
    if self.termination_names:
      self.reason_counts.add_(torch.stack([
        snapshot["termination"][key].sum(dtype=torch.long) for key in self.termination_names]))

  @torch.no_grad()
  def add_episodes(self, returns, steps):
    self.completed += returns.numel()
    self.episode_sums.add_(torch.stack((returns.sum(dtype=torch.float64),
                                      steps.sum(dtype=torch.float64) * self.step_dt)))

  def values(self):
    step_sums, reason_counts, episode_sums = (
      self.step_sums.cpu().tolist(), self.reason_counts.cpu().tolist(), self.episode_sums.cpu().tolist())
    row = {key: value / self.sample_count if self.sample_count else ""
           for key, value in zip(self.step_fields, step_sums)}
    # Undo dt scaling only for display; snapshots and training rewards retain it.
    if self.sample_count:
      for name in self.reward_names:
        row[f"Reward/{name}"] /= self.step_dt
    for key, count in zip(self.termination_names, reason_counts):
      row[f"Termination/{key}_fraction"] = count / self.completed if self.completed else ""
    row["Episode/completed"] = self.completed
    row["Episode/return_raw"] = episode_sums[0] / self.completed if self.completed else ""
    row["Episode/length_seconds"] = episode_sums[1] / self.completed if self.completed else ""
    return row

  def clear(self):
    self.step_sums.zero_()
    self.reason_counts.zero_()
    self.episode_sums.zero_()
    self.sample_count = self.completed = 0


@torch.no_grad()
def sampling_metrics(command):
  """Describe the actual reset distribution without sampling/updating its EMA."""
  motion = command.motion
  mode = command.cfg.sampling_mode
  if mode == "return_based":
    probabilities = command.sampler.probabilities(update_history=False)
  elif mode == "uniform":
    probabilities = torch.full((motion.num_bins,), 1 / motion.num_bins, device=motion.device)
  elif mode == "start":
    # Start mode samples clips uniformly, assigning mass only to their first bin.
    probabilities = torch.zeros(motion.num_bins, device=motion.device)
    bins = motion.frame_bin[motion.clip_start]
    probabilities.scatter_add_(0, bins, torch.full_like(bins, 1 / motion.num_clips, dtype=torch.float32))
  else:
    raise ValueError(f"Unknown sampling_mode: {mode}")
  entropy = (-(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum()
             / math.log(motion.num_bins)) if motion.num_bins > 1 else probabilities.new_tensor(1.0)
  top = probabilities.argmax()
  entropy_value, top_probability = torch.stack((entropy, probabilities[top])).cpu().tolist()
  return dict(zip(SAMPLING_FIELDS, (entropy_value, top_probability, int(top))))
