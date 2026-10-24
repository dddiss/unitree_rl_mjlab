from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from src.tasks.tracking.config.g1_23dof.env_cfgs import unitree_g1_23dof_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    input_fps: int,
    output_fps: int,
    device: torch.device | str,
    line_range: tuple[int, int] | None = None,
  ):
    self.motion_file = motion_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps
    self.current_idx = 0
    self.device = device
    self.line_range = line_range
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self):
    """Loads the motion from the csv file."""
    if self.line_range is None:
      motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
    else:
      motion = torch.from_numpy(
        np.loadtxt(
          self.motion_file,
          delimiter=",",
          skiprows=self.line_range[0] - 1,
          max_rows=self.line_range[1] - self.line_range[0] + 1,
        )
      )
    motion = motion.to(torch.float32).to(self.device)
    # motion[:, 2] -= 0.05
    self.motion_base_poss_input = motion[:, :3]
    self.motion_base_rots_input = motion[:, 3:7]
    self.motion_base_rots_input = self.motion_base_rots_input[
      :, [3, 0, 1, 2]
    ]  # convert to wxyz
    # CSV rounding can leave non-unit quaternions, distorting velocity transforms.
    self.motion_base_rots_input = torch.nn.functional.normalize(self.motion_base_rots_input, dim=-1)
    self.motion_dof_poss_input = motion[:, 7:]

    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

  def _interpolate_motion(self):
    """Interpolates the motion to the output fps."""
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    index_0, index_1, blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[index_0],
      self.motion_base_rots_input[index_1],
      blend,
    )
    self.motion_base_rots = torch.nn.functional.normalize(self.motion_base_rots, dim=-1)
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )
    print(
      f"Motion interpolated, input frames: {self.input_frames}, "
      f"input fps: {self.input_fps}, "
      f"output frames: {self.output_frames}, "
      f"output fps: {self.output_fps}"
    )

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Linear interpolation between two tensors."""
    return a * (1 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Batch SLERP with the same endpoint/shortest-path rules as quat_slerp."""
    tau = blend.unsqueeze(-1)
    dot = (a * b).sum(dim=-1, keepdim=True)
    aligned_b = torch.where(dot < 0, -b, b)
    angle = torch.acos(dot.abs().clamp(max=1.0))
    eps = torch.finfo(a.dtype).eps
    # Keep the unused near-zero branch finite, including identical/opposite q.
    inverse_sine = 1.0 / torch.sin(angle).clamp_min(eps)
    interpolated = (
      a * torch.sin((1.0 - tau) * angle) * inverse_sine
      + aligned_b * torch.sin(tau * angle) * inverse_sine
    )
    near = (dot.abs() - 1.0).abs() < eps * 4.0
    result = torch.where(near | (angle.abs() < eps * 4.0), a, interpolated)
    result = torch.where(tau == 0, a, result)
    return torch.where(tau == 1, b, result)

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the frame blend for the motion."""
    phase = times / self.duration
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    index_1 = (index_0 + 1).clamp(max=self.input_frames - 1)
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def _compute_velocities(self):
    """Computes the velocities of the motion."""
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_base_ang_vels = self._so3_derivative(
      self.motion_base_rots, self.output_dt
    )

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    """Computes the derivative of a sequence of SO3 rotations.

    Args:
      rotations: shape (B, 4).
      dt: time step.
    Returns:
      shape (B, 3).
    """
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
    omega = torch.cat(
      [omega[:1], omega, omega[-1:]], dim=0
    )  # repeat first and last sample
    return omega

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    """Gets the next state of the motion."""
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset_flag = False
    if self.current_idx >= self.output_frames:
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


def run_sim(
  sim: Simulation,
  scene: Scene,
  joint_names,
  input_file,
  input_fps,
  output_fps,
  output_path,
  line_range,
  renderer: OffscreenRenderer | None = None,
  video_writer: Any | None = None,
  robot_joint_indexes=None,
):
  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    line_range=line_range,
  )

  robot: Entity = scene["robot"]
  if robot_joint_indexes is None:
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

  # Reusing the simulator must not carry state or sensor history between clips.
  sim.reset()
  scene.reset()
  log: dict[str, Any] = {"fps": [output_fps]}
  device_buffers = {}
  chunk_frames = min(256, motion.output_frames)
  root_states = robot.data.default_root_state.clone()
  joint_pos = robot.data.default_joint_pos.clone()
  joint_vel = robot.data.default_joint_vel.clone()

  print(f"\nStarting simulation with {motion.output_frames} frames...")
  if video_writer is not None:
    print("Rendering enabled - generating video frames...")

  with tqdm(
    total=motion.output_frames,
    desc="Processing frames",
    unit="frame",
    ncols=100,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
  ) as pbar:
    for frame in range(motion.output_frames):
      (
        (
          motion_base_pos,
          motion_base_rot,
          motion_base_lin_vel,
          motion_base_ang_vel,
          motion_dof_pos,
          motion_dof_vel,
        ),
        _,
      ) = motion.get_next_state()

      root_states[:, 0:3] = motion_base_pos
      root_states[:, :2] += scene.env_origins[:, :2]
      root_states[:, 3:7] = motion_base_rot
      root_states[:, 7:10] = motion_base_lin_vel
      root_states[:, 10:] = motion_base_ang_vel
      robot.write_root_state_to_sim(root_states)

      joint_pos[:, robot_joint_indexes] = motion_dof_pos
      joint_vel[:, robot_joint_indexes] = motion_dof_vel
      robot.write_joint_state_to_sim(joint_pos, joint_vel)

      sim.forward()
      scene.update(sim.mj_model.opt.timestep)
      if video_writer is not None:
        assert renderer is not None
        renderer.update(sim.data)
        video_writer.add_image(renderer.render())

      values = {
        "joint_pos": robot.data.joint_pos[0],
        "joint_vel": robot.data.joint_vel[0],
        "body_pos_w": robot.data.body_link_pos_w[0],
        "body_quat_w": robot.data.body_link_quat_w[0],
        "body_lin_vel_w": robot.data.body_link_lin_vel_w[0],
        "body_ang_vel_w": robot.data.body_link_ang_vel_w[0],
      }
      slot = frame % chunk_frames
      for name, value in values.items():
        if frame == 0:
          device_buffers[name] = value.new_empty((chunk_frames, *value.shape))
        device_buffers[name][slot].copy_(value)

      frame_count = frame + 1
      if slot == chunk_frames - 1 or frame_count == motion.output_frames:
        start = frame - slot
        # Copy bounded blocks, not six tiny GPU -> CPU transfers every frame.
        for name, buffer in device_buffers.items():
          block = buffer[:slot + 1].cpu().numpy()
          if start == 0:
            log[name] = np.empty((motion.output_frames, *block.shape[1:]), dtype=block.dtype)
          log[name][start:frame_count] = block
        # Preserve both checks for every frame, now using blockwise CPU tensors.
        for name, expected in (
          ("body_lin_vel_w", motion.motion_base_lin_vels),
          ("body_ang_vel_w", motion.motion_base_ang_vels),
        ):
          torch.testing.assert_close(
            torch.from_numpy(log[name][start:frame_count, 0]),
            expected[start:frame_count].cpu(),
          )

      pbar.update(1)
      if frame_count % 100 == 0:
        pbar.set_description(f"Processing frames (t={frame_count / output_fps:.1f}s)")

  print("\nSaving data...")
  np.savez(output_path, **log)
  print(f"Motion saved to {output_path}")


class MotionConverter:
  """One robot/simulator shared by a single conversion or a whole directory."""

  def __init__(self, robot: str, output_fps: float = 50.0, device: str = "cuda:0"):
    if device.startswith("cuda") and not torch.cuda.is_available():
      print("[WARNING]: CUDA is not available. Falling back to CPU. This may be slow.")
      device = "cpu"

    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / output_fps
    if robot == "g1":    # 29 Dof
      scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
      joint_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
      ]
      default_output_dir = "./src/assets/motions/g1"
    elif robot == "g1_23dof":
      scene = Scene(unitree_g1_23dof_flat_tracking_env_cfg().scene, device=device)
      joint_names=[    # 23 Dof
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
      ]
      default_output_dir = "./src/assets/motions/g1_23dof"
    else:
      raise ValueError(f"Unsupported robot: {robot}")

    model = scene.compile()

    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)

    scene.initialize(sim.mj_model, sim.model, sim.data)

    self.scene = scene
    self.sim = sim
    self.joint_names = joint_names
    self.robot_joint_indexes = scene["robot"].find_joints(joint_names, preserve_order=True)[0]
    self.output_fps = output_fps
    self.default_output_dir = default_output_dir
    self.renderer = None
    self._resources = ExitStack()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()

  def close(self):
    try:
      self._resources.close()
    finally:
      self.renderer = None
      self.scene = None
      self.sim = None

  def _get_renderer(self):
    if self.renderer is None:
      viewer_cfg = ViewerConfig(
        height=480,
        width=640,
        origin_type=ViewerConfig.OriginType.ASSET_ROOT,
        entity_name="robot",
        distance=2.0,
        elevation=-5.0,
        azimuth=20,
      )
      renderer = OffscreenRenderer(
        model=self.sim.mj_model,
        cfg=viewer_cfg,
        scene=self.scene,
      )
      try:
        renderer.initialize()
      except BaseException:
        renderer.close()
        raise
      self._resources.callback(renderer.close)
      self.renderer = renderer
    return self.renderer

  def convert(
    self,
    input_file: str | Path,
    input_fps: float = 30.0,
    output_dir: str | Path | None = None,
    line_range: tuple[int, int] | None = None,
    save_video: bool = False,
    video_dir: str | Path | None = None,
  ):
    motion_name = Path(input_file).stem
    output_path = Path(output_dir or self.default_output_dir).expanduser() / f"{motion_name}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # A writer belongs to one motion; the renderer is shared across motions.
    with ExitStack() as stack:
      renderer = None
      video_writer = None
      if save_video:
        import mediapy as media

        video_output_dir = Path(video_dir).expanduser() if video_dir is not None else output_path.parent
        video_output_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_output_dir / f"{motion_name}.mp4"
        renderer = self._get_renderer()
        video_writer = stack.enter_context(media.VideoWriter(
          str(video_path), shape=(480, 640), fps=self.output_fps, codec="h264",
        ))
        print(f"Saving video to {video_path}")

      run_sim(
        sim=self.sim,
        scene=self.scene,
        joint_names=self.joint_names,
        robot_joint_indexes=self.robot_joint_indexes,
        input_fps=input_fps,
        input_file=str(input_file),
        output_fps=self.output_fps,
        output_path=output_path,
        line_range=line_range,
        renderer=renderer,
        video_writer=video_writer,
      )

    if save_video:
      print(f"Video saved to {video_path}")
    return output_path


def main(
  robot: str,
  input_file: str,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  save_video: bool = False,
  video_dir: str | None = None,
  line_range: tuple[int, int] | None = None,
  output_dir: str | None = None,
):
  """Convert a CSV motion to a same-stem NPZ and, optionally, an MP4 video.

  Args:
    robot: Robot model: g1 or g1_23dof.
    input_file: Path to the input CSV file.
    input_fps: Frame rate of the CSV file.
    output_fps: Desired output frame rate.
    device: Device to use.
    save_video: Whether to render and save one MP4 video for the motion.
    video_dir: Video output directory; defaults to the robot's NPZ output directory.
      The video filename is always the CSV stem followed by .mp4.
    line_range: Range of lines to process from the CSV file.
    output_dir: NPZ output directory; defaults to src/assets/motions/<robot>.
  """
  with MotionConverter(robot, output_fps=output_fps, device=device) as converter:
    converter.convert(
      input_file=input_file,
      input_fps=input_fps,
      output_dir=output_dir,
      line_range=line_range,
      save_video=save_video,
      video_dir=video_dir,
    )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
