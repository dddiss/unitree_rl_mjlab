"""Train the independent G1 SuperMimic learner using local NPZ motions."""
import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motion-file", required=True, help="NPZ file or directory")
  parser.add_argument("--log-dir", help="Log root directory (default: rl_cfg.py log_dir)")
  parser.add_argument("--run-name", "--wandb-name", dest="run_name",
                      help="Base experiment name; a timestamp is appended for the directory and W&B "
                           "(default: rl_cfg.py run_name)")
  parser.add_argument("--task", help="Registered SuperMimic task name (default: Terrain runner config; resume uses checkpoint)")
  parser.add_argument("--resume", help="Local full SuperMimic checkpoint")
  parser.add_argument("--device", default="cuda:0")
  integer_options = ("num-envs", "iterations", "buffer-steps", "horizon", "goal-frames", "hist",
                     "wm-window", "wm-batches", "wm-batch", "pi-batch", "critic-epochs",
                     "critic-minibatches", "warmup-steps", "wm-only-iters", "lr-decay-iters",
                     "ckpt-every", "log-every", "seed")
  for name in integer_options:
    parser.add_argument("--" + name, type=int)
  for name in ("utd", "lr", "gamma", "lam", "action-noise"):
    parser.add_argument("--" + name, type=float)
  parser.add_argument("--amp", choices=("none", "bf16"))
  parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
  parser.add_argument("--enable-warmup", action=argparse.BooleanOptionalAction, default=None,
                      help="Enable the WM-only phase (default: enabled); initial replay filling "
                           "always runs. On resume, use the saved setting unless overridden.")
  parser.add_argument("--logger", choices=("wandb", "csv"),
                      help="wandb writes CSV and W&B; csv writes only CSV (default: wandb)")
  parser.add_argument("--wandb-project", help="W&B project (default: unitree-supermimic)")
  parser.add_argument("--wandb-entity", help="W&B account/team (default: current account)")
  parser.add_argument("--wandb-mode", choices=("online", "offline"),
                      help="W&B connection mode (default: online)")
  args = parser.parse_args()
  import random
  import numpy as np
  import torch
  from dataclasses import asdict
  from src.tasks.supermimic.config import make_env
  from src.tasks.supermimic.config.g1.rl_cfg import unitree_g1_terrain_runner_cfg
  from src.tasks.supermimic.rl.runner import SuperMimicRunner
  from src.tasks.supermimic.tools.registry import load_runner_cfg, runner_cfg_from_dict
  saved = torch.load(args.resume, map_location="cpu", weights_only=False)["config"] if args.resume else {}
  task_id = args.task or saved.get("task_id", unitree_g1_terrain_runner_cfg().task_id)
  defaults = asdict(load_runner_cfg(task_id))
  local_settings = {name: defaults[name] for name in
                    ("collect_det_frac", "bin_ema", "softmax_cap", "pi_clip", "pi_clip_value")}
  defaults.update(saved)
  # These settings are configured in rl_cfg.py, including on resume.
  defaults.update(local_settings)
  defaults.update({key: value for key, value in vars(args).items() if key in defaults and value is not None})
  defaults["task_id"] = task_id
  cfg = runner_cfg_from_dict(defaults)
  # Generate once; the directory and W&B share this exact name. Keep the base
  # name in cfg/checkpoints so a resumed run receives only its new timestamp.
  run_name = f"{cfg.run_name}_{datetime.now():%Y%m%d_%H%M%S_%f}"
  log_dir = Path(cfg.log_dir) / run_name
  random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  if args.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("SuperMimic GPU training requires an available NVIDIA CUDA device")
  env = make_env(args.motion_file, num_envs=cfg.num_envs, device=args.device,
                 gamma=cfg.gamma, seed=cfg.seed,
                 collect_foot_rewards=True, task_id=cfg.task_id)
  runner = None
  exit_code = 1
  try:
    cfg.depth_enabled = env.depth_enabled
    runner = SuperMimicRunner(env, cfg, log_dir)
    if args.resume:
      runner.load(args.resume)
    runner.learn()
    exit_code = 0
  finally:
    try:
      if runner is not None:
        runner.close(exit_code=exit_code)
    finally:
      env.close()


if __name__ == "__main__":
  main()
