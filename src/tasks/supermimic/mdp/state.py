"""Tensor-only state convention. Velocities are world-frame link-origin velocities."""
import torch

from .rotations import mat_to_quat, quat_to_mat, mat_to_6d, yaw_mat

NUM_J = 29
STATE_DIM = 71
INV_DIM = 71
XI_DIM = 33
OBS_DIM = 93
GOAL_FRAME_DIM = 64


def flatten_state(state):
  p, r, v, w, q, qd = state
  return torch.cat((p, mat_to_quat(r), v, w, q, qd), -1)


def unflatten_state(x):
  return (x[..., :3], quat_to_mat(x[..., 3:7]), x[..., 7:10],
          x[..., 10:13], x[..., 13:42], x[..., 42:71])


def invariant_features(state):
  p, r, v, w, q, qd = state
  rt = r.transpose(-1, -2)
  return torch.cat((p[..., 2:3], mat_to_6d(yaw_mat(r).transpose(-1, -2) @ r),
                    (rt @ v.unsqueeze(-1)).squeeze(-1),
                    (rt @ w.unsqueeze(-1)).squeeze(-1), q, qd), -1)


def normalize_xi(xi):
  return torch.cat(((xi[..., :1] - 0.75) / 0.45,
                    xi[..., 1:30] / 0.01, xi[..., 30:] / 0.05), -1)


def read_state(env):
  """Caller must forward the simulation after integration or state writes."""
  cmd = env.command_manager.get_term("motion")
  d = cmd.robot.data
  pose = d.body_link_pose_w[:, cmd.root_body_index]
  vel = d.body_link_vel_w[:, cmd.root_body_index]
  return (pose[:, :3] - env.scene.env_origins, quat_to_mat(pose[:, 3:]),
          vel[:, :3], vel[:, 3:], d.joint_pos, d.joint_vel)
