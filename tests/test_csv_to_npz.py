"""Quaternion precision regressions without starting the GPU simulator."""
import ast
from pathlib import Path

import numpy as np
import torch


def motion_loader_class():
  # Import the converter's real tensor-only loader independently of GPU/GUI imports.
  source = Path(__file__).resolve().parents[1] / "scripts/csv_to_npz.py"
  node = next(n for n in ast.parse(source.read_text()).body
              if isinstance(n, ast.ClassDef) and n.name == "MotionLoader")
  namespace = {"torch": torch, "np": np}
  exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
  return namespace["MotionLoader"]


def test_csv_quaternions_normalized_preserving_position_and_joints(tmp_path):
  rows = np.arange(3 * 36, dtype=float).reshape(3, 36) / 100
  # CSV stores xyzw; rounded source data need not have exactly unit length.
  rows[:, 3:7] = [0.4, 0.2, 0.3, 0.5]
  source = tmp_path / "motion.csv"
  np.savetxt(source, rows, delimiter=",")
  cls = motion_loader_class()
  loader = cls.__new__(cls)
  loader.motion_file = str(source)
  loader.line_range = None
  loader.device = "cpu"
  loader.input_dt = 1 / 30
  loader._load_motion()
  expected = torch.tensor(rows[:, [6, 3, 4, 5]], dtype=torch.float32)
  expected /= expected.norm(dim=-1, keepdim=True)
  torch.testing.assert_close(loader.motion_base_rots_input, expected)
  torch.testing.assert_close(loader.motion_base_rots_input.norm(dim=-1), torch.ones(3))
  torch.testing.assert_close(loader.motion_base_poss_input, torch.tensor(rows[:, :3], dtype=torch.float32))
  torch.testing.assert_close(loader.motion_dof_poss_input, torch.tensor(rows[:, 7:], dtype=torch.float32))
  assert loader.input_frames == 3


def test_interpolated_quaternion_normalized_before_velocity_transforms():
  cls = motion_loader_class()
  loader = cls.__new__(cls)
  loader.device = "cpu"
  loader.input_fps, loader.output_fps = 30, 50
  loader.input_frames, loader.duration, loader.output_dt = 4, .1, .02
  loader.motion_base_poss_input = torch.arange(12, dtype=torch.float32).reshape(4, 3)
  loader.motion_dof_poss_input = torch.arange(4, dtype=torch.float32)[:, None].expand(-1, 29)
  q = torch.tensor([.1, .4, .7, .3])
  q /= q.norm()
  loader.motion_base_rots_input = q.expand(4, -1).clone()
  # Reproduce the tiny length error that remains after float32 interpolation.
  loader._slerp = lambda a, b, blend: q.expand_as(a) * 1.0000008
  loader._interpolate_motion()
  assert loader.output_frames == 5
  torch.testing.assert_close(loader.motion_base_rots.norm(dim=-1), torch.ones(5), rtol=0, atol=1e-7)
  torch.testing.assert_close(loader.motion_dof_poss[:, 0], torch.arange(5) * .6)

  # World -> body -> world angular velocity must pass the existing assertion.
  def rotate(q, vector):
    xyz = q[..., 1:]
    t = 2 * torch.linalg.cross(xyz, vector)
    return vector + q[..., :1] * t + torch.linalg.cross(xyz, t)
  velocity = torch.tensor([3., 4., 8.]).expand(5, -1)
  orientation = loader.motion_base_rots
  inverse = orientation * torch.tensor([1., -1., -1., -1.])
  restored = rotate(orientation, rotate(inverse, velocity))
  torch.testing.assert_close(restored, velocity)
