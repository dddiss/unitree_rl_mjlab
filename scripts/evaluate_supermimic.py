"""Evaluate local SuperMimic checkpoints on each NPZ clip once."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motion-file", required=True, help="NPZ file or directory")
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument("--checkpoint-file")
  source.add_argument("--run-dir")
  parser.add_argument("--watch", action="store_true")
  parser.add_argument("--record", action="store_true", help="Save per-clip state trajectories")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output-dir")
  args = parser.parse_args()
  if args.watch and not args.run_dir:
    parser.error("--watch requires --run-dir")
  from src.tasks.supermimic.config import make_env
  from src.tasks.supermimic.mdp.motion_lib import _resolve_motion_files
  from src.tasks.supermimic.rl.evaluation import load_policy, evaluate
  files = _resolve_motion_files(args.motion_file)
  base = Path(args.run_dir) if args.run_dir else Path(args.checkpoint_file).parent
  folder = Path(args.output_dir) if args.output_dir else base / "evaluation"
  folder.mkdir(parents=True, exist_ok=True)
  env = make_env(files, num_envs=len(files), device=args.device, evaluation=True)
  try:
    while True:
      checkpoints = [Path(args.checkpoint_file)] if args.checkpoint_file else sorted((base / "checkpoints").glob("ckpt_*.pt"))
      for checkpoint in checkpoints:
        report = folder / f"{checkpoint.stem}.json"
        if args.run_dir and report.exists():
          continue
        policy, cfg, ck = load_policy(checkpoint, env)
        result = evaluate(env, policy, cfg, folder / checkpoint.stem if args.record else None)
        result.update(iteration=ck["iteration"], checkpoint=str(checkpoint))
        temporary = report.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(report)
        csv_path = folder / "eval.csv"
        row = {k: result[k] for k in ("iteration", "norm_return", "norm_return_framew", "n_failed")}
        row["failed"] = "|".join(result["failed"])
        new = not csv_path.exists()
        with csv_path.open("a", newline="") as output:
          writer = csv.DictWriter(output, fieldnames=list(row))
          if new: writer.writeheader()
          writer.writerow(row)
        print(json.dumps(row), flush=True)
      if not args.watch:
        break
      time.sleep(30)
  finally:
    env.close()


if __name__ == "__main__":
  main()
