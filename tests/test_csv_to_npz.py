"""Quaternion precision regressions without starting the GPU simulator."""
import ast
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
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


def scalar_slerp(a, b, t):
  """Pre-batching mjlab scalar SLERP, used only as the regression reference."""
  if t == 0:
    return a
  if t == 1:
    return b
  dot = torch.dot(a, b)
  eps = torch.finfo(a.dtype).eps * 4
  if abs(abs(dot) - 1) < eps:
    return a
  if dot < 0:
    dot, b = -dot, -b
  angle = torch.acos(torch.clamp(dot, -1, 1))
  if abs(angle) < eps:
    return a
  inverse_sine = 1.0 / torch.sin(angle)
  return a * torch.sin((1 - t) * angle) * inverse_sine + b * torch.sin(t * angle) * inverse_sine


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
  not torch.cuda.is_available(), reason="requires CUDA"))])
def test_batched_slerp_preserves_scalar_results_and_inputs(device):
  generator = torch.Generator().manual_seed(17)
  a = torch.nn.functional.normalize(torch.randn(1024, 4, generator=generator), dim=-1)
  b = torch.nn.functional.normalize(torch.randn(1024, 4, generator=generator), dim=-1)
  t = torch.rand(1024, generator=generator)
  b[:3] = a[:3]              # identical
  b[3:6] = -a[3:6]          # opposite quaternion signs, same rotation
  b[6:9] = torch.nn.functional.normalize(a[6:9] + 1e-5, dim=-1)
  t[::7], t[1::7] = 0, 1   # preserve both endpoint signs
  expected = torch.stack([scalar_slerp(x, y, float(z)) for x, y, z in zip(a, b, t)])
  a, b, t = a.to(device), b.to(device), t.to(device)
  before = (a.clone(), b.clone(), t.clone())
  cls = motion_loader_class()
  loader = cls.__new__(cls)
  actual = loader._slerp(a, b, t)
  torch.testing.assert_close(actual.cpu(), expected, atol=1e-6, rtol=1e-5)
  for value, original in zip((a, b, t), before):
    torch.testing.assert_close(value, original, atol=0, rtol=0)
  assert torch.isfinite(actual).all()


@pytest.fixture
def conversion_runtime(monkeypatch):
  """Run the real converter with a small stateful simulator/renderer substitute.

  The substitute reuses mutable tensors like the real simulator, exposing stale
  state, block aliasing and lifecycle bugs without requiring mjlab/CUDA.
  """
  counts = SimpleNamespace(compiles=0, simulations=0, mappings=0, sim_resets=0,
                           scene_resets=0, renderers=0, renderer_closes=0, writers=[])

  class Robot:
    def __init__(self, n):
      self.data = SimpleNamespace(
        default_root_state=torch.zeros(1, 13),
        default_joint_pos=torch.zeros(1, n), default_joint_vel=torch.zeros(1, n),
        joint_pos=torch.zeros(1, n), joint_vel=torch.zeros(1, n),
        body_link_pos_w=torch.zeros(1, 3, 3), body_link_quat_w=torch.zeros(1, 3, 4),
        body_link_lin_vel_w=torch.zeros(1, 3, 3), body_link_ang_vel_w=torch.zeros(1, 3, 3),
      )
      self.root = self.data.default_root_state.clone()

    def find_joints(self, names, preserve_order):
      counts.mappings += 1
      assert preserve_order and len(names) == self.data.joint_pos.shape[1]
      return list(range(len(names))), names

    def write_root_state_to_sim(self, values):
      self.root.copy_(values)

    def write_joint_state_to_sim(self, pos, vel):
      self.data.joint_pos.copy_(pos)
      self.data.joint_vel.copy_(vel)

  class Scene:
    def __init__(self, cfg, device):
      self.robot = Robot(cfg)
      self.env_origins = torch.tensor([[5., -3., 0.]])
      assert device == "cpu"

    def __getitem__(self, key):
      assert key == "robot"
      return self.robot

    def compile(self):
      counts.compiles += 1
      return self.robot

    def initialize(self, *args):
      pass

    def reset(self):
      counts.scene_resets += 1

    def update(self, dt):
      assert dt > 0

  class Sim:
    def __init__(self, num_envs, cfg, model, device):
      counts.simulations += 1
      assert num_envs == 1
      self.robot = model
      self.device, self.model, self.data = device, model, object()
      self.mj_model = SimpleNamespace(opt=cfg.mujoco)
      self.bad_frame = None

    def reset(self):
      counts.sim_resets += 1
      self.frame = 0

    def forward(self):
      d, root = self.robot.data, self.robot.root
      d.body_link_pos_w.copy_(root[:, None, :3] + torch.arange(3)[None, :, None])
      d.body_link_quat_w.copy_(root[:, None, 3:7])
      d.body_link_lin_vel_w.copy_(root[:, None, 7:10])
      d.body_link_ang_vel_w.copy_(root[:, None, 10:13])
      if self.frame == self.bad_frame:
        d.body_link_ang_vel_w[0, 0, 1] += 1
      self.frame += 1

  class Progress:
    def __init__(self, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def update(self, n): pass
    def set_description(self, s): pass

  class Renderer:
    def __init__(self, **kwargs): counts.renderers += 1
    def initialize(self): pass
    def update(self, data): pass
    def render(self): return np.zeros((480, 640, 3), dtype=np.uint8)
    def close(self): counts.renderer_closes += 1

  class Writer:
    def __init__(self, path, **kwargs):
      self.path, self.frames, self.closed = Path(path), 0, False
      counts.writers.append(self)
    def __enter__(self): return self
    def __exit__(self, *args): self.closed = True
    def add_image(self, image):
      if self.path.stem == "bad_video":
        raise RuntimeError("video failure")
      self.frames += 1

  class ViewerConfig(SimpleNamespace):
    OriginType = SimpleNamespace(ASSET_ROOT="asset_root")

  source = Path(__file__).resolve().parents[1] / "scripts/csv_to_npz.py"
  tree = ast.parse(source.read_text())
  nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
  namespace = dict(torch=torch, np=np, Path=Path, ExitStack=ExitStack, Scene=Scene,
    Simulation=Sim, Entity=Robot, OffscreenRenderer=Renderer, ViewerConfig=ViewerConfig,
    Any=object, tqdm=Progress, SimulationCfg=lambda: SimpleNamespace(mujoco=SimpleNamespace()),
    unitree_g1_flat_tracking_env_cfg=lambda: SimpleNamespace(scene=29),
    unitree_g1_23dof_flat_tracking_env_cfg=lambda: SimpleNamespace(scene=23))
  exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
  monkeypatch.setitem(sys.modules, "mediapy", SimpleNamespace(VideoWriter=Writer))
  monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
  return namespace, counts


def prepared_motion(loader_class, frames, joints, offset):
  """Known per-frame inputs; exercise the real get_next_state and converter."""
  loader = loader_class.__new__(loader_class)
  loader.current_idx, loader.output_frames = 0, frames
  t = torch.arange(frames, dtype=torch.float32)[:, None]
  loader.motion_base_poss = (t * .01 + offset).expand(-1, 3).clone()
  loader.motion_base_rots = torch.tensor([1., 0., 0., 0.]).expand(frames, -1).clone()
  loader.motion_base_lin_vels = (t * .002 + .1).expand(-1, 3).clone()
  loader.motion_base_ang_vels = (t * .003 + .2).expand(-1, 3).clone()
  loader.motion_dof_poss = t * .001 + torch.arange(joints)[None, :] * .01 + offset
  loader.motion_dof_vels = t * .002 + torch.arange(joints)[None, :] * .02
  return loader


@pytest.mark.parametrize("robot,joints", [("g1", 29), ("g1_23dof", 23)])
def test_shared_converter_blocks_reset_and_video_lifecycle(conversion_runtime, tmp_path, robot, joints):
  ns, counts = conversion_runtime
  # Full blocks plus a short tail, then a shorter clip in the reused simulator.
  motions = {name: prepared_motion(ns["MotionLoader"], frames, joints, offset)
             for name, frames, offset in (("long", 513, .3), ("short", 7, -2.))}
  ns["MotionLoader"] = lambda motion_file, **kw: motions[Path(motion_file).stem]
  with ns["MotionConverter"](robot, device="cuda:0") as converter:
    for name, motion in motions.items():
      output = converter.convert(name + ".csv", output_dir=tmp_path, save_video=True,
                                 video_dir=tmp_path / "videos")
      with np.load(output) as data:
        assert set(data.files) == {"fps", "joint_pos", "joint_vel", "body_pos_w",
                                   "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"}
        np.testing.assert_array_equal(data["fps"], [50])
        np.testing.assert_array_equal(data["joint_pos"], motion.motion_dof_poss.numpy())
        np.testing.assert_array_equal(data["joint_vel"], motion.motion_dof_vels.numpy())
        expected_pos = motion.motion_base_poss[:, None, :] + torch.tensor([5., -3., 0.]) + torch.arange(3)[None, :, None]
        np.testing.assert_array_equal(data["body_pos_w"], expected_pos.numpy())
        for key, values in (("body_quat_w", motion.motion_base_rots),
                            ("body_lin_vel_w", motion.motion_base_lin_vels),
                            ("body_ang_vel_w", motion.motion_base_ang_vels)):
          np.testing.assert_array_equal(data[key], values[:, None, :].expand(-1, 3, -1).numpy())
      assert counts.writers[-1].closed
    assert counts.compiles == counts.simulations == counts.mappings == counts.renderers == 1
    assert counts.sim_resets == counts.scene_resets == 2
    assert counts.renderer_closes == 0
  assert counts.renderer_closes == 1
  assert [w.frames for w in counts.writers] == [513, 7]
  assert [w.path.name for w in counts.writers] == ["long.mp4", "short.mp4"]


def test_bad_frame_and_video_error_do_not_leak_into_next_clip(conversion_runtime, tmp_path):
  ns, counts = conversion_runtime
  loader_class = ns["MotionLoader"]
  ns["MotionLoader"] = lambda **kw: prepared_motion(loader_class, 258, 29, .4)
  with ns["MotionConverter"]("g1", device="cpu") as converter:
    converter.sim.bad_frame = 257
    with pytest.raises(AssertionError):
      converter.convert("bad_state.csv", output_dir=tmp_path)
    assert not (tmp_path / "bad_state.npz").exists()
    converter.sim.bad_frame = None
    with pytest.raises(RuntimeError, match="video failure"):
      converter.convert("bad_video.csv", output_dir=tmp_path, save_video=True)
    assert counts.writers[-1].closed
    converter.convert("good.csv", output_dir=tmp_path, save_video=True)
    assert counts.writers[-1].frames == 258
    assert counts.sim_resets == counts.scene_resets == 3
  assert counts.renderer_closes == 1 and all(w.closed for w in counts.writers)
  assert (tmp_path / "good.npz").is_file()
