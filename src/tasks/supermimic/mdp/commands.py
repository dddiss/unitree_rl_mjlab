from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

from .motion_lib import MotionLibrary, MotionReference
from .sampling import ReturnBasedSampler

if TYPE_CHECKING:
  from collections.abc import Callable
  from typing import Any

  import viser

  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.root_body_index = self.robot.body_names.index(self.cfg.body_names[0])
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    if cfg.motion_file and cfg.motion_files:
      raise ValueError("Set either motion_file or motion_files, not both")
    self._all_env_ids = torch.arange(self.num_envs, device=self.device)
    self.motion = MotionLibrary(
      cfg.motion_files or cfg.motion_file,
      device=self._all_env_ids.device,
      step_dt=env.step_dt,
    )
    # Global target frame for the action about to be executed.
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    # Retain k separately: a target clamped at the clip end cannot recover it.
    self._initialization_frames = torch.zeros_like(self.time_steps)
    self.start_bin = torch.zeros_like(self.time_steps)
    self._requested_frames = torch.full_like(self.time_steps, -1)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self.bin_count = self.motion.num_bins
    max_reward = sum(max(0.0, r.weight) for r in env.cfg.rewards.values()) * env.step_dt
    self.sampler = ReturnBasedSampler(self.motion, cfg.gamma, max_reward)

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  @property
  def motion_ids(self) -> torch.Tensor:
    return self.motion.clip_id[self.time_steps]

  def get_reference(self, frame_ids: torch.Tensor) -> MotionReference:
    """Reference for one frame per env, restricted to tracked bodies in world space."""
    bodies = self.body_indexes
    frames = frame_ids[:, None]
    return MotionReference(
      self.motion.joint_pos[frame_ids],
      self.motion.joint_vel[frame_ids],
      self.motion.body_pos_w[frames, bodies] + self._env.scene.env_origins[:, None, :],
      self.motion.body_quat_w[frames, bodies],
      self.motion.body_lin_vel_w[frames, bodies],
      self.motion.body_ang_vel_w[frames, bodies],
    )

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.time_steps]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps[:, None], self.body_indexes]
      + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps[:, None], self.body_indexes]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps[:, None], self.body_indexes]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps[:, None], self.body_indexes]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps, self.robot_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps, self.robot_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps, self.robot_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps, self.robot_anchor_body_index]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self, env_ids: torch.Tensor | None = None):
    ids = self._all_env_ids if env_ids is None else env_ids
    self.metrics["error_anchor_pos"][ids] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )[ids]
    self.metrics["error_anchor_rot"][ids] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )[ids]
    self.metrics["error_anchor_lin_vel"][ids] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )[ids]
    self.metrics["error_anchor_ang_vel"][ids] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )[ids]

    self.metrics["error_body_pos"][ids] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)[ids]
    self.metrics["error_body_rot"][ids] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)[ids]

    self.metrics["error_body_lin_vel"][ids] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)[ids]
    self.metrics["error_body_ang_vel"][ids] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)[ids]

    self.metrics["error_joint_pos"][ids] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )[ids]
    self.metrics["error_joint_vel"][ids] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )[ids]

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Clip joint positions and write root + joint state to sim."""
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      motion_ids = torch.randint(
        self.motion.num_clips, (len(env_ids),), device=self.device
      )
      frames = self.motion.clip_start[motion_ids]
      self.metrics["sampling_entropy"][env_ids] = 1.0
      self.metrics["sampling_top1_prob"][env_ids] = 1.0 / self.motion.num_clips
    elif self.cfg.sampling_mode == "return_based":
      frames, _ = self.sampler.sample(len(env_ids))
      probs = self.sampler.probabilities()
      self.metrics["sampling_entropy"][env_ids] = -(probs * probs.log()).sum()
      self.metrics["sampling_top1_prob"][env_ids] = probs.max()
    elif self.cfg.sampling_mode == "uniform":
      frames, _ = self.motion.sample_start_frames(len(env_ids))
      self.metrics["sampling_entropy"][env_ids] = 1.0
      self.metrics["sampling_top1_prob"][env_ids] = 1.0 / self.bin_count
    else:
      raise ValueError(f"Unknown sampling_mode: {self.cfg.sampling_mode}")
    self.metrics["sampling_top1_bin"][env_ids] = 0.5

    # GUI/external requests are applied during the env's reset, before its
    # forward(), sensor refresh, and observation-history initialization.
    requested = self._requested_frames[env_ids]
    exact = requested >= 0
    frames = torch.where(exact, requested, frames)
    self._requested_frames[env_ids] = -1
    self._initialization_frames[env_ids] = frames
    # Initialize the robot at k, but expose k+1 as the first control target.
    # Explicit playback requests may select the last frame, so clamp per clip.
    end = self.motion.clip_end[self.motion.clip_id[frames]]
    self.time_steps[env_ids] = torch.minimum(frames + 1, end)
    # Sampling statistics belong to the initialization frame k, not its target.
    self.start_bin[env_ids] = self.motion.frame_bin[frames]
    noise_mask = (~exact).to(torch.float32)[:, None]

    reference = self.motion.get_reference(frames)
    root_pos = (
      reference.body_pos_w[:, self.root_body_index].clone()
      + self._env.scene.env_origins[env_ids]
    )
    root_ori = reference.body_quat_w[:, self.root_body_index].clone()
    root_lin_vel = reference.body_lin_vel_w[:, self.root_body_index].clone()
    root_ang_vel = reference.body_ang_vel_w[:, self.root_body_index].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    ) * noise_mask
    root_pos += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori = quat_mul(orientations_delta, root_ori)
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    ) * noise_mask
    root_lin_vel += rand_samples[:, :3]
    root_ang_vel += rand_samples[:, 3:]

    joint_pos = reference.joint_pos.clone()
    joint_vel = reference.joint_vel

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    ) * noise_mask

    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_ori,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      joint_vel,
    )

  def get_relative_body_poses(
    self, reference: MotionReference, env_ids: torch.Tensor | None = None
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Align a reference to the current robot XY/yaw, retaining reference height.

    Used for visualization and metrics; shared MDP rewards align their own
    reference against the evaluated state.
    """
    ids = self._all_env_ids if env_ids is None else env_ids
    anchor = self.motion_anchor_body_index
    anchor_pos_w_repeat = reference.body_pos_w[ids, anchor, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = reference.body_quat_w[ids, anchor, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[ids, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[ids, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    body_quat = quat_mul(delta_ori_w, reference.body_quat_w[ids])
    body_pos = delta_pos_w + quat_apply(
      delta_ori_w, reference.body_pos_w[ids] - anchor_pos_w_repeat
    )
    return body_pos, body_quat

  def update_relative_body_poses(self, env_ids: torch.Tensor | None = None) -> None:
    ids = self._all_env_ids if env_ids is None else env_ids
    pos, quat = self.get_relative_body_poses(self.get_reference(self.time_steps), ids)
    self.body_pos_relative_w[ids] = pos
    self.body_quat_relative_w[ids] = quat

  def _resolve_env_ids(self, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None:
      return self._all_env_ids
    if isinstance(env_ids, slice):
      return self._all_env_ids[env_ids]
    return env_ids

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
    ids = self._resolve_env_ids(env_ids)
    if ids.numel() == 0:
      return {}
    extras = super().reset(ids)
    # Motion changes only at episode reset, never on a command timer.
    self.time_left[ids] = torch.inf
    # The environment forwards the newly written state before compute(dt=0).
    return extras

  def compute(
    self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    """Advance the target after evaluation; reset rows keep k+1 with dt=0.

    Deliberately bypass the base timer-driven resampling: the clip-end
    termination asks the environment to reset, including its action/history
    buffers. This method only advances references and refreshes derived poses.
    """
    ids = self._resolve_env_ids(env_ids)
    if ids.numel() == 0:
      return
    advance = (dt[ids] > 0).long() if isinstance(dt, torch.Tensor) else int(dt > 0)
    end = self.motion.clip_end[self.motion_ids[ids]]
    self.time_steps[ids] = torch.minimum(self.time_steps[ids] + advance, end)
    self._update_command(ids)
    self._update_metrics(ids)

  def _update_command(self, env_ids: torch.Tensor | None = None):
    self.update_relative_body_poses(env_ids)

  def request_frames(
    self, frames: torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    """Queue exact global initialization frames k for the selected envs' reset.

    The first control target is k+1, clamped to the clip's last frame. Unlike
    training sampling, playback may initialize at a final frame; its start_bin
    is -1 and it truncates on the next environment step.
    """
    ids = self._resolve_env_ids(env_ids)
    if torch.any((frames < 0) | (frames >= self.motion.num_frames)).item():
      raise IndexError("Requested frame is outside the motion library")
    self._requested_frames[ids] = frames

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    """Reset selected environments to an exact global frame through env.reset()."""
    frames = torch.full((len(env_ids),), frame, dtype=torch.long, device=self.device)
    self.request_frames(frames, env_ids)
    self._env.reset(env_ids=env_ids)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        # Build a ghost model with only visual geoms visible. Collision geoms (nonzero
        # contype/conaffinity) get alpha=0 so the viewer's alpha filter excludes them.
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        for gi in range(self._ghost_model.ngeom):
          if (
            self._ghost_model.geom_contype[gi] != 0
            or self._ghost_model.geom_conaffinity[gi] != 0
          ):
            self._ghost_model.geom_rgba[gi, 3] = 0
          else:
            self._ghost_model.geom_rgba[gi] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        frame = self.time_steps[batch]
        root_pos = (
          self.motion.body_pos_w[frame, self.root_body_index]
          + self._env.scene.env_origins[batch]
        )
        qpos[free_joint_q_adr[0:3]] = root_pos.cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = (
          self.motion.body_quat_w[frame, self.root_body_index].cpu().numpy()
        )
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

        visualizer.add_ghost_mesh(
          qpos,
          model=self._ghost_model,
          label=f"ghost_{batch}",
        )

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
        desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos,
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
        current_anchor_quat = self.robot_anchor_quat_w[batch]
        current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=current_anchor_pos,
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )

  def _get_gui_frame(self, env_idx: int) -> int:
    """Local initialization frame k for restarting at the playback position."""
    clip = self.motion_ids[env_idx]
    frame = torch.minimum(
      self._initialization_frames[env_idx] + self._env.episode_length_buf[env_idx],
      self.motion.clip_end[clip],
    )
    return int(frame - self.motion.clip_start[clip])

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Select an initialization frame k; its first control target is k+1."""
    labels = tuple(
      f"{index}: {motion_name}"
      for index, motion_name in enumerate(self.motion.motion_names)
    )
    lengths = self.motion.clip_len.cpu().tolist()
    env_idx = get_env_idx()
    clip = int(self.motion_ids[env_idx])

    with server.gui.add_folder(name.capitalize()):
      motion_selector = server.gui.add_dropdown(
        "Motion", options=labels, initial_value=labels[clip]
      )
      scrubber = server.gui.add_slider(
        "Frame",
        min=0,
        max=lengths[clip] - 1,
        step=1,
        initial_value=self._get_gui_frame(env_idx),
      )

      @motion_selector.on_update
      def _(_) -> None:
        scrubber.value = 0
        scrubber.max = lengths[labels.index(motion_selector.value)] - 1
        if on_change is not None:
          on_change()

      @scrubber.on_update
      def _(_) -> None:
        # Editing a control must not change the live command or robot state.
        if on_change is not None:
          on_change()

      all_envs_cb = server.gui.add_checkbox("All envs", initial_value=True)
      start_btn = server.gui.add_button("Start Here")

      @start_btn.on_click
      def _(_) -> None:
        if request_action is not None:
          ids = self._all_env_ids if all_envs_cb.value else self._all_env_ids[
            get_env_idx() : get_env_idx() + 1
          ]
          motion_ids = torch.full_like(ids, labels.index(motion_selector.value))
          local_frames = torch.full_like(ids, int(scrubber.value))
          frames = self.motion.get_clip_frames(motion_ids, local_frames)
          self.request_frames(frames, ids)
          request_action(
            "CUSTOM",
            {"type": "gui_reset", "all_envs": all_envs_cb.value},
          )

    self._scrubber_handles = (motion_selector, scrubber, all_envs_cb, start_btn)
    self._motion_labels = labels
    self._gui_env_index = get_env_idx
    self._set_scrubber_disabled(True)

  def _set_scrubber_disabled(self, disabled: bool) -> None:
    """Enable or disable the motion scrubber GUI controls."""
    for handle in self._scrubber_handles:
      handle.disabled = disabled

  def on_viewer_pause(self, paused: bool) -> None:
    if hasattr(self, "_scrubber_handles"):
      if paused:
        idx = self._gui_env_index()
        clip = int(self.motion_ids[idx])
        selector, scrubber, _, _ = self._scrubber_handles
        selector.value = self._motion_labels[clip]
        scrubber.value = 0
        scrubber.max = int(self.motion.clip_len[clip]) - 1
        scrubber.value = self._get_gui_frame(idx)
      self._set_scrubber_disabled(not paused)


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  # One file or directory, or an explicit ordered file list. Set only one.
  motion_file: str = ""
  motion_files: tuple[str, ...] = ()
  anchor_body_name: str
  body_names: tuple[str, ...]
  entity_name: str
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  # "uniform": uniform bins, then frames. "start": uniform clips, first frame.
  sampling_mode: Literal["uniform", "start", "return_based"] = "return_based"
  gamma: float = 0.99

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)
