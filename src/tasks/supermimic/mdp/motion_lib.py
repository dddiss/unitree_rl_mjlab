"""Multi-clip NPZ reference motions, independent of the simulation environment.

Files use the seven arrays written by ``scripts/csv_to_npz.py``. Body data stays
in the full robot body order; quaternions are scalar-first (w, x, y, z). No FK,
resampling, environment-origin translation, or episode bookkeeping happens here.

Example::

  motion = MotionLibrary("src/assets/motions/g1", device="cpu", step_dt=0.02)
  time_steps, start_bins = motion.sample_start_frames(4096)
  reference = motion.get_reference(time_steps)
  future = motion.future_frames(time_steps, 10)

``time_steps`` contains global reference frame indices, not episode durations.
The caller owns these indices and decides when to advance or reset them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

_STATE_FIELDS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


class MotionReference(NamedTuple):
  """Reference tensors with the query's leading dimensions and all robot bodies."""

  joint_pos: torch.Tensor
  joint_vel: torch.Tensor
  body_pos_w: torch.Tensor
  body_quat_w: torch.Tensor
  body_lin_vel_w: torch.Tensor
  body_ang_vel_w: torch.Tensor


class MotionLibrary:
  """Load one NPZ, an ordered list of NPZ files, or a directory of NPZ files.

  Directory discovery is sorted and nonrecursive. Explicit lists keep their
  order. ``motion_files`` stores resolved paths, so this tuple also defines the
  clip-ID manifest to save alongside future training checkpoints.

  Training files are assumed to have valid state arrays in the same joint/body
  order. Names and body counts are not stored or checked. Loading only checks
  that FPS matches ``1 / step_dt`` and each clip has at least two frames, so a
  sampled start has a following reference frame.

  ``clip_end`` is inclusive. Reset bins use global half-open intervals
  ``[bin_start, bin_stop)`` and exclude each clip's last frame, leaving at least
  one transition after reset. ``bin_duration_s / step_dt`` is rounded to the
  nearest integer frame count; the final bin may be shorter. ``frame_bin`` is
  -1 at terminal frames.
  """

  joint_pos: torch.Tensor
  joint_vel: torch.Tensor
  body_pos_w: torch.Tensor
  body_quat_w: torch.Tensor
  body_lin_vel_w: torch.Tensor
  body_ang_vel_w: torch.Tensor

  def __init__(
    self,
    motion_files: str | Path | Sequence[str | Path],
    device: str | torch.device = "cpu",
    *,
    step_dt: float = 0.02,
    bin_duration_s: float = 1.0,
  ) -> None:
    frames_per_bin = bin_duration_s / step_dt
    self.device = torch.device(device)
    self.step_dt = float(step_dt)
    self.fps = 1.0 / self.step_dt
    self.bin_frames = round(frames_per_bin)
    self.motion_files = _resolve_motion_files(motion_files)
    self.motion_names = tuple(Path(path).stem for path in self.motion_files)
    self.num_clips = len(self.motion_files)

    arrays: dict[str, list[np.ndarray]] = {name: [] for name in _STATE_FIELDS}
    lengths: list[int] = []
    for path in self.motion_files:
      data = _load_motion(path, self.fps)
      lengths.append(data["joint_pos"].shape[0])
      for name in _STATE_FIELDS:
        arrays[name].append(data[name])

    # Concatenate on CPU before transferring, without keeping a second full GPU
    # copy of the motion bank. All state tensors are float32 on the target device.
    for name in _STATE_FIELDS:
      values = arrays.pop(name)
      tensor = torch.from_numpy(np.concatenate(values, axis=0)).to(self.device)
      setattr(self, name, tensor)

    self.num_frames = sum(lengths)
    self.clip_len = torch.tensor(lengths, dtype=torch.long, device=self.device)
    self.clip_start = self.clip_len.cumsum(0) - self.clip_len
    self.clip_end = self.clip_start + self.clip_len - 1
    self.clip_id = torch.repeat_interleave(
      torch.arange(self.num_clips, device=self.device),
      self.clip_len,
      output_size=self.num_frames,
    )

    bin_clip, bin_start, bin_stop = [], [], []
    frame_bin = np.full(self.num_frames, -1, dtype=np.int64)
    offset = 0
    for clip, length in enumerate(lengths):
      # length - 1 is the exclusive upper bound for valid RESET starts. For a
      # 51-frame clip at 50 Hz this makes one bin, not an empty terminal-only bin.
      for local_start in range(0, length - 1, self.bin_frames):
        start = offset + local_start
        stop = offset + min(local_start + self.bin_frames, length - 1)
        frame_bin[start:stop] = len(bin_start)
        bin_clip.append(clip)
        bin_start.append(start)
        bin_stop.append(stop)
      offset += length
    self.bin_clip = torch.tensor(bin_clip, dtype=torch.long, device=self.device)
    self.bin_start = torch.tensor(bin_start, dtype=torch.long, device=self.device)
    self.bin_stop = torch.tensor(bin_stop, dtype=torch.long, device=self.device)
    self.num_bins = len(bin_start)
    self.frame_bin = torch.from_numpy(frame_bin).to(self.device)

  def get_reference(self, frame_ids: torch.Tensor) -> MotionReference:
    """Gather all six state arrays, preserving ``frame_ids.shape``.

    IDs must be valid global indices in [0, num_frames). For frequent training
    queries we check dtype/device only, avoiding a host/device synchronization.
    In particular, callers must not pass negative indices (PyTorch wraps them).
    Raw data has no per-environment origin added and includes every robot body.
    """
    self._check_index_tensor(frame_ids, "frame_ids")
    return MotionReference(*(getattr(self, name)[frame_ids] for name in _STATE_FIELDS))

  def get_clip_frames(
    self, motion_ids: torch.Tensor, local_frame_ids: torch.Tensor
  ) -> torch.Tensor:
    """Convert same-shaped clip/local-frame IDs to global frame IDs.

    This setup/reset helper validates bounds, including the local clip boundary;
    invalid frames raise instead of spilling into the next clip. Bounds checks
    synchronize when tensors are on a GPU, unlike the frequent query methods.
    """
    self._check_index_tensor(motion_ids, "motion_ids")
    self._check_index_tensor(local_frame_ids, "local_frame_ids")
    if motion_ids.shape != local_frame_ids.shape:
      raise ValueError("motion_ids and local_frame_ids must have the same shape")
    if torch.any((motion_ids < 0) | (motion_ids >= self.num_clips)).item():
      raise IndexError("motion_ids contains an invalid clip ID")
    if torch.any(
      (local_frame_ids < 0) | (local_frame_ids >= self.clip_len[motion_ids])
    ).item():
      raise IndexError("local_frame_ids contains a frame outside its clip")
    return self.clip_start[motion_ids] + local_frame_ids

  def future_frames(self, frame_ids: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Return f+1 through f+K, repeating each clip's last frame beyond its end.

    Input IDs have the same preconditions as ``get_reference``. The result has
    shape ``(*frame_ids.shape, num_frames)``. Padding never crosses clip borders
    and does not itself end, reset, or advance an environment.
    """
    self._check_index_tensor(frame_ids, "frame_ids")
    if (
      isinstance(num_frames, bool)
      or not isinstance(num_frames, int)
      or num_frames < 1
    ):
      raise ValueError("num_frames must be a positive integer")
    end = self.clip_end[self.clip_id[frame_ids]]
    offsets = torch.arange(1, num_frames + 1, device=self.device)
    return torch.minimum(frame_ids.unsqueeze(-1) + offsets, end.unsqueeze(-1))

  def sample_start_frames(
    self, num_samples: int, *, generator: torch.Generator | None = None
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly sample bins, then frames inside them; return (frames, bins).

    Bins are equally likely, including shorter final bins. Consequently this is
    neither uniform-over-clips nor uniform-over-all-frames sampling. ``generator``
    must match the library's device when supplied. No return-based weights or
    episode statistics are maintained here.
    """
    if (
      isinstance(num_samples, bool)
      or not isinstance(num_samples, int)
      or num_samples < 0
    ):
      raise ValueError("num_samples must be a nonnegative integer")
    bins = torch.randint(
      self.num_bins, (num_samples,), device=self.device, generator=generator
    )
    return self.sample_frames_from_bins(bins, generator=generator), bins

  def sample_frames_from_bins(
    self, bin_ids: torch.Tensor, *, generator: torch.Generator | None = None
  ) -> torch.Tensor:
    """Uniformly sample within selected bins, ready for a future weighted sampler.

    ``bin_ids`` must be valid indices in [0, num_bins). Like reference queries,
    this method checks dtype/device only. The result preserves the input shape.
    """
    self._check_index_tensor(bin_ids, "bin_ids")
    start = self.bin_start[bin_ids]
    width = self.bin_stop[bin_ids] - start
    fraction = torch.rand(
      bin_ids.shape, dtype=torch.float32, device=self.device, generator=generator
    )
    return start + (fraction * width).long()

  def _check_index_tensor(self, indices: torch.Tensor, name: str) -> None:
    if not isinstance(indices, torch.Tensor) or indices.dtype != torch.long:
      raise TypeError(f"{name} must be a torch.long tensor")
    if indices.device != self.device:
      raise ValueError(f"{name} must be on {self.device}, got {indices.device}")


def _resolve_motion_files(
  files: str | Path | Sequence[str | Path],
) -> tuple[str, ...]:
  if isinstance(files, (str, Path)):
    if not str(files).strip():
      raise ValueError("motion_files cannot be an empty path")
    path = Path(files).expanduser()
    paths = sorted(path.glob("*.npz")) if path.is_dir() else [path]
  else:
    paths = list(files)
  if not paths:
    raise ValueError("MotionLibrary requires at least one NPZ file")
  resolved = []
  seen = set()
  for path in paths:
    if not isinstance(path, (str, Path)) or not str(path).strip():
      raise ValueError("Each motion file must be a nonempty path")
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".npz":
      raise ValueError(f"Expected an NPZ file, got {path}")
    if path in seen:
      raise ValueError(f"Duplicate motion file: {path}")
    seen.add(path)
    resolved.append(str(path))
  return tuple(resolved)


def _load_motion(path: str, expected_fps: float) -> dict[str, np.ndarray]:
  with np.load(path, allow_pickle=False) as archive:
    fps = float(archive["fps"].item())
    if not math.isclose(fps, expected_fps, rel_tol=1e-6, abs_tol=1e-6):
      raise ValueError(
        f"{path}: fps={fps:g}, expected {expected_fps:g} (= 1 / step_dt). "
        "Convert the motion at the required FPS with scripts/csv_to_npz.py."
      )

    data = {
      name: np.ascontiguousarray(archive[name], dtype=np.float32)
      for name in _STATE_FIELDS
    }
    if data["joint_pos"].shape[0] < 2:
      raise ValueError(f"{path}: a motion must contain at least two frames")
    return data
