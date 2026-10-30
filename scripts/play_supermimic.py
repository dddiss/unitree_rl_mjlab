"""Play a local SuperMimic policy in Native/Viser, or write one MP4 per motion."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motion-file", required=True, help="NPZ file or directory used by checkpoint")
  parser.add_argument("--checkpoint-file", required=True)
  parser.add_argument("--clip", help="Motion filename stem; defaults to first clip in viewer, all clips for video")
  parser.add_argument("--viewer", choices=("auto", "native", "viser"), default="auto")
  parser.add_argument("--video-dir", help="Write MP4 files here instead of opening a viewer")
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  from src.tasks.supermimic.config import make_env
  from src.tasks.supermimic.rl.evaluation import load_policy, read_checkpoint_config, record_clips, PlaybackEnv
  from src.tasks.supermimic.rl.runner import PolicyController
  from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
  saved_cfg = read_checkpoint_config(args.checkpoint_file)
  env = make_env(args.motion_file, num_envs=1, device=args.device, evaluation=True,
                 randomize_terrain_on_reset=True,
                 depth_enabled=saved_cfg.depth_enabled,
                 render_mode="rgb_array" if args.video_dir else None)
  try:
    policy, cfg, _ = load_policy(args.checkpoint_file, env)
    cmd = env.command_manager.get_term("motion")
    clip = cmd.motion.motion_names.index(args.clip) if args.clip else 0
    if args.video_dir:
      for path in record_clips(env, policy, cfg, args.video_dir, [clip] if args.clip else None):
        print(f"Saved {path}")
      return
    cmd.request_frames(cmd.motion.clip_start[clip:clip+1])
    env.reset()
    controller = PolicyController(env, policy, cfg)
    viewer = args.viewer
    if viewer == "auto":
      viewer = "native" if sys.platform == "darwin" or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") else "viser"
    viewer_cls = NativeMujocoViewer if viewer == "native" else ViserPlayViewer
    viewer_cls(PlaybackEnv(env), controller).run()
  finally:
    env.close()


if __name__ == "__main__":
  main()
