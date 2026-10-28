"""Independent SuperMimic collection, checkpoint and training logging runner."""
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..mdp.state import read_state, OBS_DIM
from ..mdp.sampling import EpisodeBinReturns
from .algorithm import SuperMimicAlgorithm
from .buffer import ReplayBuffer
from .logging import TrainingLogger, format_terminal_summary
from .metrics import LEGACY_FIELDS, PERFORMANCE_FIELDS, SAMPLING_FIELDS, IntervalMetrics, sampling_metrics
from ..perception.history import DepthHistory, select_history
from ..config.g1.rl_cfg import SuperMimicConfig, migrate_checkpoint_config

VISUAL_FIELDS = ("wm_depth_loss", "wm_total_loss")


def motion_manifest(motion):
  return [{"path": path, "frames": int(length)}
          for path, length in zip(motion.motion_files, motion.clip_len.cpu().tolist())]


def check_manifest(expected, actual):
  # Relocation is allowed, but clip IDs/lengths must keep their meaning.
  key = lambda items: [(Path(item["path"]).name, item["frames"]) for item in items]
  if key(expected) != key(actual):
    raise ValueError("Checkpoint motion order/names/lengths differ from the loaded NPZ library")


def atomic_save(state, path):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  torch.save(state, temporary)
  temporary.replace(path)


class PolicyController:
  """History is advanced once per physical step and reset only for affected rows."""
  def __init__(self, env, policy, cfg, noise=False, update_norm=False):
    self.env, self.policy, self.cfg = env, policy, cfg
    self.noise, self.update_norm = noise, update_norm
    self.history = torch.zeros(env.num_envs, cfg.hist, OBS_DIM, device=env.device)
    self.version = torch.full_like(env.reset_version, -1)
    self.steps = torch.full_like(env.episode_length_buf, -1)
    self.depth_history = (DepthHistory(env.num_envs, cfg.depth_history_length, env.device)
                          if cfg.depth_enabled else None)

  @torch.no_grad()
  def __call__(self, observations=None):
    del observations
    e = self.env
    cmd = e.command_manager.get_term("motion")
    state = read_state(e)
    obs = e.functional_mdp.obs(state, e.action_manager.action, e.context, cmd.time_steps)
    if self.noise:
      obs = e.functional_mdp.add_noise(obs)
    fresh = self.version != e.reset_version
    advanced = (self.steps != e.episode_length_buf) | fresh
    rolled = torch.cat((self.history[:, 1:], obs[:, None]), 1)
    self.history = torch.where(fresh[:, None, None], obs[:, None].expand_as(self.history),
                               torch.where(advanced[:, None, None], rolled, self.history))
    goal = e.functional_mdp.goal(state, cmd.time_steps, self.cfg.goal_frames)
    if self.noise:
      goal = e.functional_mdp.add_goal_noise(goal)
    if self.update_norm and advanced.any():
      self.policy.obs_norm.update(obs[advanced])
      self.policy.goal_norm.update(goal[advanced])
    self.version.copy_(e.reset_version)
    self.steps.copy_(e.episode_length_buf)
    if self.depth_history is not None:
      depth = self.depth_history.update(e.depth_image, fresh, advanced)
      return self.policy(self.history, goal, select_history(depth, self.cfg.depth_frames, self.cfg.depth_skip))
    return self.policy(self.history, goal)


class SuperMimicRunner:
  def __init__(self, env, cfg, log_dir):
    self.env, self.cfg = env, cfg
    self.log_dir = Path(log_dir)
    self.log_dir.mkdir(parents=True, exist_ok=True)
    self.algorithm = SuperMimicAlgorithm(env.functional_mdp, cfg, env.device)
    self.command = env.command_manager.get_term("motion")
    self.command.sampler.bin_ema = cfg.bin_ema
    self.command.sampler.softmax_cap = cfg.softmax_cap
    self.motion = self.command.motion
    self.buffer = ReplayBuffer(cfg.buffer_steps, env.num_envs, env.context,
                               self.motion.clip_end, self.motion.clip_id, cfg.horizon,
                               cfg.depth_history_length if cfg.depth_enabled else 0)
    self.controller = PolicyController(env, self.algorithm.policy, cfg, noise=True, update_norm=True)
    self.iteration = self.sim_steps = self.env_steps = 0
    self.step_acc = self.wall_offset = 0.0
    self._started = time.monotonic()
    self.episode_return = torch.zeros(env.num_envs, device=env.device)
    max_episode_steps = (env.max_episode_length
                         if "time_out" in env.termination_manager.active_terms else None)
    self.episode_bins = EpisodeBinReturns(
      self.motion, env.num_envs, cfg.gamma, env.functional_mdp.max_step_reward,
      max_episode_steps=max_episode_steps,
    )
    self.stats = {}
    self.interval_metrics = IntervalMetrics(env.reward_manager.active_terms, env.termination_manager.active_terms,
                                           env.step_dt, env.device)
    self.logger = None
    self._resume_checkpoint = None
    self._reset_performance()
    self.env.reset()
    (self.log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (self.log_dir / "motions.json").write_text(json.dumps(motion_manifest(self.motion), indent=2))

  @torch.no_grad()
  def collect(self, count):
    for _ in range(count):
      action = self.controller()
      noise = self.cfg.action_noise * torch.randn_like(action)
      # Fixed environment IDs stay noise-free across resets. Keep the full draw
      # so changing the fraction does not itself change RNG consumption.
      noise[:round(self.cfg.collect_det_frac * self.env.num_envs)] = 0
      action = (action + noise).clamp(-20, 20)
      self.env.step(action)
      tr = self.env.last_transition
      self.interval_metrics.add_step(self.env.last_step_metrics)
      self.buffer.add(tr)
      self.algorithm.observe(tr, self.env.context)
      self.episode_return += tr["r"]
      # The first target f identifies reset frame k=f-1. Subsequent targets
      # contribute reward to that same starting bin, even across bin boundaries.
      self.episode_bins.add(tr["f"], tr["r"])
      self.sim_steps += self.env.num_envs
      self.env_steps += 1
      self._record("sim_reward", float(tr["r"].mean() / self.env.functional_mdp.max_step_reward))
      done = tr["terminated"] | tr["truncated"]
      ids = done.nonzero(as_tuple=False).flatten()
      if len(ids):
        steps = tr["ep_step"][ids] + 1
        self.interval_metrics.add_episodes(self.episode_return[ids], steps)
        bins, returns = self.episode_bins.finish(ids)
        self.command.sampler.update_returns(bins, returns)
        self._record("episode_length", float(steps.float().mean()))
        self._record("episode_return", float((self.episode_return[ids] / steps
                                               / self.env.functional_mdp.max_step_reward).mean()))
        self._record("failure_fraction", float(tr["terminated"][ids].float().mean()))
        self.episode_return[ids] = 0
        self.env.reset(env_ids=ids)

  def _record(self, key, value):
    self.stats.setdefault(key, []).append(value)

  def save(self, path):
    checkpoint = {
      "format_version": 2 if self.cfg.depth_enabled else 1, "algorithm": self.algorithm.state_dict(),
      "config": asdict(self.cfg), "motions": motion_manifest(self.motion),
      "sampler": self.command.sampler.state_dict(), "iteration": self.iteration,
      "sim_steps": self.sim_steps, "env_steps": self.env_steps, "step_acc": self.step_acc,
      "wall_time": self.wall_offset + time.monotonic() - self._started,
      "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
              "torch": torch.get_rng_state(),
              "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
    }
    atomic_save(checkpoint, path)

  def load(self, path):
    ck = torch.load(path, map_location=self.env.device, weights_only=False)
    check_manifest(ck["motions"], motion_manifest(self.motion))
    saved_cfg = asdict(SuperMimicConfig(**migrate_checkpoint_config(ck["config"])))
    current_cfg = asdict(self.cfg)
    # Scheduling, logging, precision, collection mix, sampling and actor clipping may change without
    # reinterpreting weights. Replay is refilled after loading.
    mutable = {"iterations", "num_envs", "ckpt_every", "log_every", "amp", "compile",
               "buffer_steps", "warmup_steps", "enable_warmup", "logger", "wandb_project",
               "wandb_entity", "wandb_mode", "log_dir", "run_name", "collect_det_frac", "bin_ema",
               "softmax_cap", "pi_clip", "pi_clip_value", "depth_checkpoint", "depth_batch_size"}
    for key in current_cfg:
      if key not in mutable and current_cfg[key] != saved_cfg[key]:
        raise ValueError(f"Resume configuration changed {key}; use the checkpoint configuration")
    self.algorithm.load_state_dict(ck["algorithm"])
    self.command.sampler.load_state_dict(ck["sampler"])
    self.iteration, self.sim_steps, self.env_steps = ck["iteration"], ck["sim_steps"], ck["env_steps"]
    self.step_acc, self.wall_offset = ck["step_acc"], ck["wall_time"]
    self._started = time.monotonic()
    rng = ck["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu())
    if torch.cuda.is_available() and len(rng["cuda"]) == torch.cuda.device_count():
      torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
    self.episode_return.zero_()
    self.episode_bins.reset()
    self.stats.clear()
    self.interval_metrics.clear()
    self._resume_checkpoint = str(Path(path).resolve())
    self.env.reset()

  def learn(self):
    try:
      self._start_logging()
      self._learn()
    except BaseException as error:
      try:
        self.close(exit_code=1)
      except BaseException as cleanup_error:
        error.add_note(f"Training logger cleanup also failed: {cleanup_error}")
      raise
    else:
      self.close()

  def _learn(self):
    c = self.cfg
    # Replay is not checkpointed. Initial filling is required with either
    # warmup setting, including when resuming into an empty buffer.
    prefill = max(c.warmup_steps, c.wm_window)
    if self.buffer.size < min(prefill, self.buffer.capacity):
      self.collect(prefill - self.buffer.size)
    # Throughput measures the main loop, excluding initial replay filling.
    self._reset_performance()
    for iteration in range(self.iteration, c.iterations):
      lr = self.algorithm.set_lr(iteration)
      self.step_acc += 1 / c.utd
      steps = int(self.step_acc)
      self.step_acc -= steps
      started = time.monotonic()
      self.collect(steps)
      self._phase_seconds[0] += time.monotonic() - started
      wm_warmup = c.enable_warmup and iteration < c.wm_only_iters
      if self.buffer.size >= c.wm_window:
        started = time.monotonic()
        wm_updated = False
        for _ in range(c.wm_batches):
          metrics = self.algorithm.world_model_update(self.buffer, strict=not wm_warmup)
          if metrics is None:
            # Keep collecting if no complete same-episode window exists yet.
            break
          wm_updated = True
          for key, value in metrics.items():
            self._record(key, value)
        self._phase_seconds[1] += time.monotonic() - started
        if not wm_updated:
          self._record("policy_skipped", 1.0)
        elif not wm_warmup:
          started = time.monotonic()
          for key, value in self.algorithm.policy_update(self.buffer).items():
            self._record(key, value)
          self._phase_seconds[2] += time.monotonic() - started
      else:
        # Policy/Critic must wait until this iteration can fit the world model.
        self._record("policy_skipped", 1.0)
      self.iteration = iteration + 1
      if self.iteration % c.log_every == 0 or self.iteration == c.iterations:
        self._log(lr)
      if self.iteration % c.ckpt_every == 0 or self.iteration == c.iterations:
        self.save(self.log_dir / "last.pt")
        self.save(self.log_dir / "checkpoints" / f"ckpt_{self.iteration:06d}.pt")

  def _log(self, lr):
    row = {"iteration": self.iteration, "wall_h": (self.wall_offset + time.monotonic() - self._started) / 3600,
           "sim_steps": self.sim_steps, "env_steps": self.env_steps, "lr": lr}
    row.update({key: float(np.mean(self.stats[key])) if self.stats.get(key) else "" for key in LEGACY_FIELDS[5:]})
    if self.cfg.depth_enabled:
      row.update({key: float(np.mean(self.stats[key])) if self.stats.get(key) else "" for key in VISUAL_FIELDS})
    row.update(self.interval_metrics.values())
    row.update(sampling_metrics(self.command))
    now = time.monotonic()
    iterations = self.iteration - self._interval_iteration
    elapsed = max(now - self._interval_started, 1e-9)
    row.update(zip(PERFORMANCE_FIELDS, (
      *(seconds / iterations if iterations else "" for seconds in self._phase_seconds),
      iterations / elapsed, (self.sim_steps - self._interval_sim_steps) / elapsed)))
    # Start the next wall-clock window before I/O so log/checkpoint overhead is
    # included in end-to-end throughput rather than hidden by the timers.
    self._reset_performance(now)
    self._start_logging()
    self.logger.log(row)
    print(format_terminal_summary(
      row, total_iterations=self.cfg.iterations, interval_iterations=iterations,
      interval_seconds=elapsed,
      wm_warmup=self.cfg.enable_warmup and self.iteration <= self.cfg.wm_only_iters,
    ), flush=True)
    self.stats.clear()
    self.interval_metrics.clear()

  def _reset_performance(self, now=None):
    self._interval_started = time.monotonic() if now is None else now
    self._interval_iteration = self.iteration
    self._interval_sim_steps = self.sim_steps
    self._phase_seconds = [0.0, 0.0, 0.0]

  def _start_logging(self):
    if self.logger is None:
      fields = LEGACY_FIELDS + self.interval_metrics.fields + SAMPLING_FIELDS + PERFORMANCE_FIELDS
      if self.cfg.depth_enabled:
        fields += VISUAL_FIELDS
      self.logger = TrainingLogger(self.log_dir, self.cfg, fields,
                                   resume_checkpoint=self._resume_checkpoint, start_iteration=self.iteration)

  def close(self, exit_code=0):
    if self.logger is not None:
      self.logger.close(exit_code=exit_code)
