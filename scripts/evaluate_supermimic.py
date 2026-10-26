"""Evaluate local SuperMimic checkpoints on each NPZ clip once."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


NOISE_MODES = ("none", "action", "observation", "both")


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motion-file", required=True, help="NPZ file or directory")
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument("--checkpoint-file")
  source.add_argument("--run-dir")
  parser.add_argument("--watch", action="store_true")
  parser.add_argument("--record", action="store_true", help="Save per-clip state trajectories")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output-dir")
  parser.add_argument("--noise-mode", choices=(*NOISE_MODES, "all"), default="none",
                      help="Noise diagnostic: none/action/observation/both, or all four")
  parser.add_argument("--seeds", type=int, nargs="+",
                      help="Repeat each diagnostic mode with these seeds (default: 0 1 2)")
  parser.add_argument("--action-noise", type=float,
                      help="Exploration standard deviation for action/both; defaults to checkpoint config")
  parser.add_argument("--action-clip", type=float,
                      help="Optional symmetric action limit for EVERY mode; default: no clipping")
  args = parser.parse_args(argv)
  if args.watch and not args.run_dir:
    parser.error("--watch requires --run-dir")
  if args.action_noise is not None and (not math.isfinite(args.action_noise) or args.action_noise < 0):
    parser.error("--action-noise must be finite and nonnegative")
  if args.action_clip is not None and (not math.isfinite(args.action_clip) or args.action_clip <= 0):
    parser.error("--action-clip must be finite and positive")
  if args.seeds is not None:
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
      parser.error("--seeds must be integers in [0, 2**32)")
    if len(set(args.seeds)) != len(args.seeds):
      parser.error("--seeds must not contain duplicates")
  return args


def _write_json(path, result):
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(".json.tmp")
  temporary.write_text(json.dumps(result, indent=2))
  temporary.replace(path)


def _write_csv(path, rows):
  temporary = path.with_suffix(".csv.tmp")
  with temporary.open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  temporary.replace(path)


def evaluate_noise_modes(env, policy, cfg, ck, checkpoint, folder, args, evaluate):
  """Keep diagnostic reports separate from standard evaluation and playback."""
  modes = NOISE_MODES if args.noise_mode == "all" else (args.noise_mode,)
  seeds = args.seeds if args.seeds is not None else [0, 1, 2]
  std = cfg.action_noise if args.action_noise is None else args.action_noise
  root = folder / "noise_evaluation" / checkpoint.stem
  results, rows, summaries = [], [], []
  for mode in modes:
    mode_results = []
    mode_updated = False
    for seed in seeds:
      settings = dict(noise_mode=mode, seed=seed,
                      observation_noise=mode in ("observation", "both"),
                      action_noise=std if mode in ("action", "both") else 0.0,
                      action_clip=args.action_clip)
      report = root / mode / f"seed_{seed}.json"
      reused = bool(args.run_dir and report.exists())
      if reused:
        result = json.loads(report.read_text())
        if result.get("evaluation") != settings or result.get("checkpoint") != str(checkpoint):
          raise ValueError(f"{report}: different diagnostic settings; use another --output-dir")
      else:
        options = {k: v for k, v in settings.items() if k != "noise_mode"}
        result = evaluate(env, policy, cfg, report.with_suffix("") if args.record else None, **options)
        result.update(iteration=ck["iteration"], checkpoint=str(checkpoint), evaluation=settings)
        _write_json(report, result)
        mode_updated = True
      mode_results.append(result)
      results.append(result)
      row = dict(iteration=result["iteration"], **settings,
                 **{k: result[k] for k in ("norm_return", "norm_return_framew", "n_failed")},
                 failed="|".join(result["failed"]))
      rows.append(row)
      if not reused:
        print(json.dumps(row), flush=True)
    summary = dict(iteration=ck["iteration"], noise_mode=mode, num_seeds=len(seeds),
                   action_noise=mode_results[0]["evaluation"]["action_noise"],
                   action_clip=args.action_clip)
    for metric in ("norm_return", "norm_return_framew", "n_failed"):
      values = [result[metric] for result in mode_results]
      summary[f"{metric}_mean"] = statistics.mean(values)
      summary[f"{metric}_std"] = statistics.pstdev(values)
    summaries.append(summary)
    if mode_updated:
      print(json.dumps({"noise_summary": summary}), flush=True)
  # Rewrite this invocation's table so rerunning a checkpoint never double-counts seeds.
  _write_csv(root / "eval.csv", rows)
  _write_csv(root / "summary.csv", summaries)
  _write_json(root / "summary.json", dict(checkpoint=str(checkpoint), seeds=seeds, modes=summaries))
  return results


def main():
  args = parse_args()
  from src.tasks.supermimic.config import make_env
  from src.tasks.supermimic.mdp.motion_lib import _resolve_motion_files
  from src.tasks.supermimic.rl.evaluation import load_policy, evaluate
  files = _resolve_motion_files(args.motion_file)
  base = Path(args.run_dir) if args.run_dir else Path(args.checkpoint_file).parent
  folder = Path(args.output_dir) if args.output_dir else base / "evaluation"
  folder.mkdir(parents=True, exist_ok=True)
  env = make_env(files, num_envs=len(files), device=args.device, evaluation=True)
  diagnostic = (args.noise_mode != "none" or args.seeds is not None
                or args.action_noise is not None or args.action_clip is not None)
  try:
    while True:
      checkpoints = [Path(args.checkpoint_file)] if args.checkpoint_file else sorted((base / "checkpoints").glob("ckpt_*.pt"))
      for checkpoint in checkpoints:
        if diagnostic:
          policy, cfg, ck = load_policy(checkpoint, env)
          evaluate_noise_modes(env, policy, cfg, ck, checkpoint, folder, args, evaluate)
          continue
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
