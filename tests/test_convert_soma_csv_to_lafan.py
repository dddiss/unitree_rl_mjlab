"""SOMA units, joint ordering, rotations and compatibility with the NPZ loader."""

import ast
import csv
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/convert_soma_csv_to_lafan.py"
spec = importlib.util.spec_from_file_location("convert_soma_csv_to_lafan", SCRIPT)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def write_soma(path, frames=12, reverse_columns=False):
  path.parent.mkdir(parents=True, exist_ok=True)
  columns = ["Frame", *converter.ROOT_COLUMNS, *(n + "_dof" for n in converter.JOINT_NAMES)]
  # Distinct joint angles reveal reordering errors, not just unit errors.
  data = np.zeros((frames, 36))
  data[:, 0] = np.arange(frames)
  data[:, 1] = np.arange(frames) * 10
  data[:, 2:4] = [-200, 80]
  data[:, 4:7] = [30, -20, 70]
  data[:, 7:] = np.arange(29) * 3 - 40
  order = list(reversed(range(36))) if reverse_columns else list(range(36))
  with path.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow([columns[i] for i in order])
    writer.writerows(data[:, order])
  return data


def rotation_matrix_xyz(degrees):
  x, y, z = np.deg2rad(degrees)
  rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
  ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
  rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
  return rz @ ry @ rx


def quat_matrix(q):
  x, y, z, w = q
  return np.array([
    [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
    [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
    [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
  ])


@pytest.mark.parametrize("angles", [(0, 0, 0), (0, 0, 90), (30, -20, 70), (180, 90, -160)])
def test_rotation_matches_xyz_composition(angles):
  q = converter.euler_xyz_to_xyzw(np.array([angles], dtype=float))[0]
  np.testing.assert_allclose(np.linalg.norm(q), 1, atol=1e-14)
  np.testing.assert_allclose(quat_matrix(q), rotation_matrix_xyz(angles), atol=1e-14)


def test_units_joint_order_downsampling_and_existing_loader(tmp_path):
  import torch

  source = tmp_path / "soma/dance.csv"
  original = write_soma(source, reverse_columns=True)
  output = tmp_path / "lafan/dance.csv"
  assert converter.main([
    "--input", str(source.parent), "--output", str(output.parent), "--num-workers", "1",
  ]) == 0
  rows = np.loadtxt(output, delimiter=",")
  assert rows.shape == (3, 36)
  np.testing.assert_allclose(rows[:, :3], original[::4, 1:4] / 100, atol=5e-10)
  np.testing.assert_allclose(rows[:, 7:], np.deg2rad(original[::4, 7:]), atol=5e-10)
  np.testing.assert_allclose(quat_matrix(rows[0, 3:7]), rotation_matrix_xyz([30, -20, 70]), atol=2e-9)
  # Read the produced CSV with the existing converter's actual loader, without
  # importing its GPU environment and renderer dependencies.
  file = ROOT / "scripts/csv_to_npz.py"
  node = next(n for n in ast.parse(file.read_text()).body
              if isinstance(n, ast.ClassDef) and n.name == "MotionLoader")
  namespace = {"np": np, "torch": torch}
  exec(compile(ast.Module(body=[node], type_ignores=[]), str(file), "exec"), namespace)
  cls = namespace["MotionLoader"]
  loader = cls.__new__(cls)
  loader.motion_file, loader.line_range = str(output), None
  loader.device, loader.input_dt = "cpu", 1 / 30
  loader._load_motion()
  assert loader.input_frames == 3
  torch.testing.assert_close(loader.motion_base_poss_input, torch.tensor(rows[:, :3], dtype=torch.float32))
  torch.testing.assert_close(loader.motion_dof_poss_input, torch.tensor(rows[:, 7:], dtype=torch.float32))
  torch.testing.assert_close(loader.motion_base_rots_input, torch.tensor(rows[:, [6, 3, 4, 5]], dtype=torch.float32))


def test_real_parallel_cli_preserves_subdirectories(tmp_path):
  source, output = tmp_path / "source", tmp_path / "output"
  write_soma(source / "a/dance.csv")
  write_soma(source / "b/deep/dance.CSV")
  write_soma(source / "root.csv")
  result = subprocess.run([
    sys.executable, "-B", str(SCRIPT), "--input", str(source), "--output", str(output),
    "--fps_source", "120", "--fps", "30", "--num_workers", "2",
  ], capture_output=True, text=True)
  assert result.returncode == 0, result.stdout + result.stderr
  assert "3 converted" in result.stdout
  assert sorted(p.relative_to(output).as_posix() for p in output.rglob("*.csv")) == [
    "a/dance.csv", "b/deep/dance.csv", "root.csv",
  ]


def test_skip_overwrite_and_failure_keep_old_output(tmp_path, capsys):
  source, output = tmp_path / "source", tmp_path / "output"
  write_soma(source / "good.csv")
  (source / "broken.csv").write_text("unrecognized,format\n1,2\n")
  output.mkdir()
  (output / "good.csv").write_text("existing")
  args = ["--input", str(source), "--output", str(output), "--num-workers", "1"]
  assert converter.main(args) == 1
  assert (output / "good.csv").read_text() == "existing"
  assert not (output / "broken.csv").exists()
  assert "Missing SOMA/Bones G1 columns" in capsys.readouterr().out
  assert converter.main([*args, "--overwrite"]) == 1
  saved = (output / "good.csv").read_bytes()
  (source / "good.csv").write_text("bad\n")
  assert converter.main([*args, "--overwrite"]) == 1
  assert (output / "good.csv").read_bytes() == saved
  assert not list(output.glob("*.tmp"))


@pytest.mark.parametrize("rates", [(120, 50), (30, 120), (0, 30), (float("nan"), 30)])
def test_nonintegral_or_invalid_rates_are_rejected(rates):
  with pytest.raises(ValueError):
    converter.frame_stride(*rates)


def test_keep_source_rate_and_reject_short_or_nonfinite_motion(tmp_path):
  source = tmp_path / "dance.csv"
  write_soma(source)
  assert converter.frame_stride(120, 120) == 1
  assert converter.load_lafan_rows(source, 1).shape == (12, 36)
  with pytest.raises(ValueError, match="two frames"):
    converter.load_lafan_rows(source, 12)
  source.write_text(source.read_text().replace("-200.0", "nan"))
  with pytest.raises(ValueError, match="NaN"):
    converter.load_lafan_rows(source)


def test_no_overlapping_output_tree(tmp_path):
  write_soma(tmp_path / "source/dance.csv")
  with pytest.raises(SystemExit):
    converter.main(["--input", str(tmp_path / "source"), "--output", str(tmp_path / "source/out")])
  assert not (tmp_path / "source/out").exists()
