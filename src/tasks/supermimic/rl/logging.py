"""CSV, optional W&B and terminal logging for the SuperMimic learner."""
from __future__ import annotations

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
  for key in ("wm_latent_loss", "wm_depth_loss", "wm_total_loss"):
    if key in row:
      add(WANDB_METRIC_NAMES[key], value(row.get(key), ".6f"))
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
            console="off", disable_code=True, disable_git=True,
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
      payload[WANDB_METRIC_NAMES.get(name, name)] = value
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
