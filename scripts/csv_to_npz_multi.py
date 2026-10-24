"""Batch-convert CSV motions with the existing csv_to_npz.py converter.

Example (run in the project's installed Python environment):
  python scripts/csv_to_npz_multi.py --robot g1 --input-dir ./motions \
    --output-dir ./motions_npz --recursive

Recursive conversion preserves subdirectories, including for MP4 files. Existing
complete outputs are skipped unless --overwrite is set. The robot, simulator and
optional renderer are initialized once and reused for every CSV.
"""

import argparse
from contextlib import ExitStack
from pathlib import Path
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _create_converter(robot: str, output_fps: float, device: str):
  # Keep --help and runs that skip every file free of simulation imports.
  if __package__:
    from .csv_to_npz import MotionConverter
  else:
    from csv_to_npz import MotionConverter
  return MotionConverter(robot, output_fps=output_fps, device=device)


def _boolean(value: str) -> bool:
  if value.lower() in ("true", "false"):
    return value.lower() == "true"
  raise argparse.ArgumentTypeError("Expected True or False")


def convert_directory(
  robot: str,
  input_dir: Path,
  output_dir: Path | None = None,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  save_video: bool = False,
  video_dir: Path | None = None,
  line_range: list[int] | None = None,
  recursive: bool = False,
  overwrite: bool = False,
) -> int:
  """Return 0 on success, or 1 after reporting all failed conversions."""
  input_dir = input_dir.expanduser().resolve()
  if not input_dir.is_dir():
    raise ValueError(f"Input directory does not exist: {input_dir}")
  paths = input_dir.rglob("*") if recursive else input_dir.iterdir()
  files = sorted(p for p in paths if p.is_file() and p.suffix.lower() == ".csv")
  if not files:
    raise ValueError(f"No CSV files found in {input_dir}")
  output_dir = (output_dir or PROJECT_ROOT / "src/assets/motions" / robot).expanduser().resolve()
  video_dir = (video_dir or output_dir).expanduser().resolve()
  relative_outputs = [p.relative_to(input_dir).with_suffix(".npz") for p in files]
  if len(set(relative_outputs)) != len(relative_outputs):
    raise ValueError("CSV files with the same stem would write the same output file")

  converted = skipped = 0
  failed = []
  with ExitStack() as stack:
    converter = None
    for index, (source, relative) in enumerate(zip(files, relative_outputs), 1):
      output = output_dir / relative
      video = video_dir / relative.with_suffix(".mp4")
      if not overwrite and output.is_file() and (not save_video or video.is_file()):
        print(f"[{index}/{len(files)}] Skipping existing output: {relative}", flush=True)
        skipped += 1
        continue
      if converter is None:
        converter = stack.enter_context(_create_converter(robot, output_fps, device))
      print(f"[{index}/{len(files)}] Converting {source.relative_to(input_dir)}", flush=True)
      try:
        converter.convert(
          input_file=source,
          input_fps=input_fps,
          output_dir=output.parent,
          line_range=tuple(line_range) if line_range is not None else None,
          save_video=save_video,
          video_dir=video.parent if save_video else None,
        )
        if not output.is_file() or (save_video and not video.is_file()):
          raise RuntimeError("Conversion did not produce all requested output files")
      except Exception as error:
        traceback.print_exc()
        failed.append(source)
        print(f"[FAILED] {source}: {error}; continuing.", flush=True)
      else:
        converted += 1

  print(f"\nFinished: {converted} converted, {skipped} skipped, {len(failed)} failed.")
  print(f"NPZ directory: {output_dir}")
  if save_video:
    print(f"MP4 directory: {video_dir}")
  if failed:
    print("Failed CSV files:")
    for source in failed:
      print(f"  {source}")
  return int(bool(failed))


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--robot", required=True, choices=("g1", "g1_23dof"))
  parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing CSV motions")
  parser.add_argument("--output-dir", type=Path, help="NPZ directory (default: src/assets/motions/<robot>)")
  parser.add_argument("--input-fps", type=float, default=30.0)
  parser.add_argument("--output-fps", type=float, default=50.0)
  parser.add_argument("--device", default="cuda:0", help="CUDA device or cpu; unavailable CUDA falls back to CPU")
  parser.add_argument("--save-video", nargs="?", const=True, default=False, type=_boolean,
                      help="Write one MP4 per CSV; accepts --save-video or --save-video True/False")
  parser.add_argument("--video-dir", type=Path, help="MP4 directory (default: NPZ directory)")
  parser.add_argument("--line-range", nargs=2, type=int, metavar=("START", "END"),
                      help="Inclusive 1-based line range, applied to every CSV")
  parser.add_argument("--recursive", action="store_true", help="Include subdirectories and preserve their layout")
  parser.add_argument("--overwrite", action="store_true", help="Reconvert even if the requested outputs exist")
  args = parser.parse_args(argv)
  try:
    return convert_directory(**vars(args))
  except ValueError as error:
    parser.error(str(error))


if __name__ == "__main__":
  raise SystemExit(main())
