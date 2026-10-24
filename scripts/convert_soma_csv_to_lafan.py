"""Convert SOMA/Bones-SEED G1 CSVs to the LAFAN1 G1 CSV layout.

Input: one headered CSV per motion, with root_translate{X,Y,Z} in cm,
root_rotate{X,Y,Z} in degrees and 29 named *_joint_dof columns in degrees.
Output: no header, [root xyz (m), root quaternion xyzw, 29 joint angles (rad)],
in the same joint order as LAFAN1_Retargeting_Dataset/g1 and csv_to_npz.py.

Like SONIC's convert_soma_csv_to_motion_lib.py, 120 -> 30 Hz uses every fourth
frame. Only equal rates or integer downsampling are supported; use the existing
csv_to_npz.py for interpolation to 50 Hz. CSV itself does not store frame rate.
This does not perform retargeting or FK and does not read split deployment CSVs.

Example:
  python scripts/convert_soma_csv_to_lafan.py --input /data/bones_seed/g1/csv \
    --output /data/bones_lafan --fps-source 120 --fps 30 --num-workers 16

Discovery is recursive, subdirectories are preserved, and existing outputs are
skipped unless --overwrite is given. Only NumPy and the standard library are used.
"""

import argparse
import csv
import math
from multiprocessing import Pool
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


# LAFAN G1 / MuJoCo joint order. Select by name, not by input column position.
JOINT_NAMES = (
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
  "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
  "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
ROOT_COLUMNS = (
  "root_translateX", "root_translateY", "root_translateZ",
  "root_rotateX", "root_rotateY", "root_rotateZ",
)


def frame_stride(fps_source: float, fps: float) -> int:
  """Require an exact frame stride instead of silently changing motion speed."""
  if not all(math.isfinite(rate) and rate > 0 for rate in (fps_source, fps)):
    raise ValueError("Frame rates must be finite and positive")
  ratio = fps_source / fps
  if not math.isfinite(ratio) or ratio < 1 or not math.isclose(ratio, round(ratio)):
    raise ValueError(
      "--fps-source must be an integer multiple of --fps (for example 120 -> 30). "
      "Use csv_to_npz.py to interpolate the converted CSV to 50 Hz."
    )
  return round(ratio)


def euler_xyz_to_xyzw(degrees: np.ndarray) -> np.ndarray:
  """Match SONIC's scipy Rotation.from_euler('xyz', ..., degrees=True)."""
  x, y, z = (np.deg2rad(degrees) * 0.5).T
  sx, sy, sz = np.sin(x), np.sin(y), np.sin(z)
  cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
  return np.stack((
    sx * cy * cz - cx * sy * sz,
    cx * sy * cz + sx * cy * sz,
    cx * cy * sz - sx * sy * cz,
    cx * cy * cz + sx * sy * sz,
  ), axis=-1)


def load_lafan_rows(source: Path, stride: int = 1) -> np.ndarray:
  """Read one named-column CSV and return headerless LAFAN-compatible rows."""
  columns = (*ROOT_COLUMNS, *(name + "_dof" for name in JOINT_NAMES))
  with source.open(encoding="utf-8-sig", newline="") as stream:
    header = [name.strip() for name in next(csv.reader(stream), [])]
    missing = [name for name in columns if name not in header]
    if missing:
      raise ValueError(f"Missing SOMA/Bones G1 columns: {', '.join(missing)}")
    if any(header.count(name) != 1 for name in columns):
      raise ValueError("Required column names must be unique")
    values = np.loadtxt(
      stream, delimiter=",", usecols=[header.index(name) for name in columns], ndmin=2,
    )
  values = values[::stride]
  if len(values) < 2:
    raise ValueError("Motion must retain at least two frames after downsampling")
  if not np.isfinite(values).all():
    raise ValueError("Motion contains NaN or infinity")
  return np.concatenate((
    values[:, :3] / 100.0,
    euler_xyz_to_xyzw(values[:, 3:6]),
    np.deg2rad(values[:, 6:]),
  ), axis=-1)


def _convert_one(job: tuple[Path, Path, int, bool]) -> tuple[str, str, int, str]:
  source, output, stride, overwrite = job
  temporary = None
  try:
    if output.is_file() and not overwrite:
      return str(source), "skipped", 0, ""
    rows = load_lafan_rows(source, stride)
    output.parent.mkdir(parents=True, exist_ok=True)
    # A failed conversion must not leave a partial CSV that a later run skips.
    with NamedTemporaryFile(mode="w", dir=output.parent, suffix=".tmp", delete=False) as stream:
      temporary = Path(stream.name)
      np.savetxt(stream, rows, delimiter=",", fmt="%.9f")
    temporary.replace(output)
    return str(source), "converted", len(rows), ""
  except Exception as error:
    return str(source), "failed", 0, str(error)
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


def convert_directory(
  input_dir: Path,
  output_dir: Path,
  fps_source: float = 120.0,
  fps: float = 30.0,
  num_workers: int = 8,
  overwrite: bool = False,
) -> int:
  stride = frame_stride(fps_source, fps)
  input_dir, output_dir = input_dir.expanduser().resolve(), output_dir.expanduser().resolve()
  if not input_dir.is_dir():
    raise ValueError(f"Input directory does not exist: {input_dir}")
  if input_dir == output_dir or input_dir in output_dir.parents or output_dir in input_dir.parents:
    raise ValueError("Input and output directories must be separate, non-nested directories")
  if num_workers < 1:
    raise ValueError("--num-workers must be positive")
  sources = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".csv")
  if not sources:
    raise ValueError(f"No CSV files found in {input_dir}")
  outputs = [output_dir / source.relative_to(input_dir).with_suffix(".csv") for source in sources]
  if len(set(outputs)) != len(outputs):
    raise ValueError("Multiple inputs would write the same output CSV")
  jobs = ((source, output, stride, overwrite) for source, output in zip(sources, outputs))
  counts = {"converted": 0, "skipped": 0, "failed": 0}
  frames = 0

  def consume(results):
    nonlocal frames
    for index, (source, status, length, error) in enumerate(results, 1):
      counts[status] += 1
      frames += length
      if error:
        print(f"[FAILED] {source}: {error}", flush=True)
      if index % 100 == 0 or index == len(sources):
        print(f"[{index}/{len(sources)}] {counts}", flush=True)

  if num_workers == 1 or len(sources) == 1:
    consume(map(_convert_one, jobs))
  else:
    with Pool(min(num_workers, len(sources))) as pool:
      consume(pool.imap_unordered(_convert_one, jobs, chunksize=4))
  print(f"Finished: {counts['converted']} converted, {counts['skipped']} skipped, {counts['failed']} failed.")
  print(f"CSV directory: {output_dir}")
  print(f"Output FPS: {fps:g}; converted frames: {frames}. Use --input-fps {fps:g} with csv_to_npz_multi.py.")
  return int(counts["failed"] > 0)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--input", "--input-dir", dest="input_dir", required=True, type=Path)
  parser.add_argument("--output", "--output-dir", dest="output_dir", required=True, type=Path)
  parser.add_argument("--fps-source", "--fps_source", type=float, default=120.0)
  parser.add_argument("--fps", type=float, default=30.0, help="CSV output FPS (default: 30)")
  parser.add_argument("--num-workers", "--num_workers", type=int, default=8)
  parser.add_argument("--overwrite", action="store_true")
  args = parser.parse_args(argv)
  try:
    return convert_directory(**vars(args))
  except ValueError as error:
    parser.error(str(error))


if __name__ == "__main__":
  raise SystemExit(main())
