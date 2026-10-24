"""Deterministic per-clip evaluation and MP4 recording on the mjlab backend."""
from pathlib import Path

import numpy as np
import torch

from . import networks
from ..config.g1.rl_cfg import SuperMimicConfig, migrate_legacy_run_name
from .networks import Policy
from .runner import PolicyController, check_manifest, motion_manifest
from ..mdp.state import GOAL_FRAME_DIM, OBS_DIM, read_state, flatten_state


def load_policy(checkpoint, env):
  ck = torch.load(checkpoint, map_location=env.device, weights_only=False)
  check_manifest(ck["motions"], motion_manifest(env.command_manager.get_term("motion").motion))
  cfg = SuperMimicConfig(**migrate_legacy_run_name(ck["config"]))
  networks.AMP_DTYPE = torch.bfloat16 if cfg.amp == "bf16" else None
  policy = Policy(OBS_DIM, GOAL_FRAME_DIM * cfg.goal_frames, cfg.hist, hidden=cfg.policy_hidden).to(env.device)
  policy.load_state_dict(ck["algorithm"]["policy"])
  return policy.eval(), cfg, ck


@torch.no_grad()
def evaluate(env, policy, cfg, record_dir=None):
  cmd = env.command_manager.get_term("motion")
  motion = cmd.motion
  if env.num_envs != motion.num_clips:
    raise ValueError("Evaluation creates one environment per motion clip")
  ids = torch.arange(env.num_envs, device=env.device)
  cmd.request_frames(motion.clip_start.clone())
  env.reset()
  controller = PolicyController(env, policy, cfg)
  completed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  failed = torch.zeros_like(completed)
  returns = torch.zeros(env.num_envs, device=env.device)
  steps = torch.zeros_like(ids)
  trajectories = [[flatten_state(read_state(env))[i].cpu().numpy()] for i in range(env.num_envs)] if record_dir else None
  while not bool(completed.all()):
    active = ~completed
    env.step(controller())
    tr = env.last_transition
    returns += torch.where(active, tr["r"], 0.)
    steps += active.long()
    failed |= active & tr["terminated"]
    if trajectories is not None:
      for i in active.nonzero().flatten().tolist():
        trajectories[i].append(tr["s1"][i].cpu().numpy())
    done = tr["terminated"] | tr["truncated"]
    completed |= done
    reset_ids = ids[done]
    if len(reset_ids):
      # Completed rows can keep stepping, but never contribute scores twice.
      cmd.request_frames(motion.clip_start[reset_ids], reset_ids)
      env.reset(env_ids=reset_ids)
  denominator = (motion.clip_len - 1) * env.functional_mdp.max_step_reward
  normalized = returns / denominator
  names = motion.motion_names
  result = {"norm_return": float(normalized.mean()),
            "norm_return_framew": float(returns.sum() / denominator.sum()),
            "n_failed": int(failed.sum()),
            "failed": [names[i] for i in failed.nonzero().flatten().tolist()],
            "per_clip": [{"name": name, "return": float(returns[i]),
                          "normalized_return": float(normalized[i]), "steps": int(steps[i]),
                          "failed": bool(failed[i])} for i, name in enumerate(names)]}
  if trajectories is not None:
    folder = Path(record_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
      np.savez_compressed(folder / f"{name}.npz", states=np.stack(trajectories[i]),
                          clip_start=int(motion.clip_start[i]), fps=1/env.step_dt)
  return result


@torch.no_grad()
def record_clips(env, policy, cfg, output_dir, clips=None):
  """One complete deterministic episode/video per clip; requires one render env."""
  import mediapy as media
  if env.num_envs != 1 or env.render_mode != "rgb_array":
    raise ValueError("Video recording requires one env with render_mode='rgb_array'")
  cmd = env.command_manager.get_term("motion")
  motion = cmd.motion
  chosen = range(motion.num_clips) if clips is None else clips
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  output = []
  for clip in chosen:
    cmd.request_frames(motion.clip_start[clip:clip+1])
    env.reset()
    controller = PolicyController(env, policy, cfg)
    path = output_dir / f"{motion.motion_names[clip]}.mp4"
    frame = env.render()
    with media.VideoWriter(str(path), shape=frame.shape[:2], fps=1/env.step_dt, codec="h264") as writer:
      writer.add_image(frame)
      while True:
        env.step(controller())
        writer.add_image(env.render())
        tr = env.last_transition
        if bool((tr["terminated"] | tr["truncated"]).any()):
          break
    output.append(str(path))
  return output


class PlaybackEnv:
  """Viewer adapter: explicit reset after each episode, including timeouts."""
  def __init__(self, env):
    self.env = env

  def __getattr__(self, name):
    return getattr(self.env, name)

  @property
  def unwrapped(self):
    return self.env

  def step(self, action):
    obs, reward, term, truncated, info = self.env.step(action)
    done = term | truncated
    ids = done.nonzero(as_tuple=False).flatten()
    if len(ids):
      cmd = self.env.command_manager.get_term("motion")
      cmd.request_frames(cmd.motion.clip_start[cmd.motion_ids[ids]], ids)
      obs, _ = self.env.reset(env_ids=ids)
    return obs, reward, done, info
