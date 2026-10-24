"""CPU coverage of target-timed diagnostic snapshots and reward accounting."""
import ast
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

import torch

from supermimic.mdp.logging_metrics import TRACKING_METRICS, capture_step_metrics


def command_at_target():
  # Execute the project's exact error definitions without importing GPU mjlab.
  path = Path(__file__).resolve().parents[2] / "src/tasks/supermimic/mdp/commands.py"
  cls = next(node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.ClassDef) and node.name == "MotionCommand")
  function = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_update_metrics")
  ns = {"torch": torch,
        "quat_error_magnitude": lambda a, b: 2 * (a.mul(b).sum(-1).abs().clamp(max=1)).acos()}
  exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), ns)
  command = NS(_all_env_ids=torch.arange(2), metrics={name: torch.full((2,), -99.) for name in TRACKING_METRICS})
  command._update_metrics = MethodType(ns["_update_metrics"], command)
  for reference, actual, shape in (
    ("anchor_pos_w", "robot_anchor_pos_w", (2, 3)),
    ("anchor_lin_vel_w", "robot_anchor_lin_vel_w", (2, 3)),
    ("anchor_ang_vel_w", "robot_anchor_ang_vel_w", (2, 3)),
    ("body_pos_relative_w", "robot_body_pos_w", (2, 14, 3)),
    ("body_lin_vel_w", "robot_body_lin_vel_w", (2, 14, 3)),
    ("body_ang_vel_w", "robot_body_ang_vel_w", (2, 14, 3)),
    ("joint_pos", "robot_joint_pos", (2, 29)),
    ("joint_vel", "robot_joint_vel", (2, 29)),
  ):
    setattr(command, reference, torch.zeros(shape))
    setattr(command, actual, torch.zeros(shape))
  for reference, actual, shape in (
    ("anchor_quat_w", "robot_anchor_quat_w", (2, 4)),
    ("body_quat_relative_w", "robot_body_quat_w", (2, 14, 4)),
  ):
    q = torch.zeros(shape); q[..., 0] = 1
    setattr(command, reference, q)
    setattr(command, actual, q.clone())
  return command


def test_target_matched_snapshot_survives_advance_and_partial_reset():
  cmd = command_at_target()
  cmd.anchor_pos_w[:, 0] = torch.tensor([1., 10.])
  cmd.robot_anchor_pos_w.copy_(cmd.anchor_pos_w)
  original_rng = torch.random.get_rng_state()
  snap = capture_step_metrics(cmd, {"tracking": torch.ones(2)}, {"tracking": 0.5}, {}, .02)
  assert all(value.eq(0).all() for value in snap["tracking"].values())
  assert torch.equal(original_rng, torch.random.get_rng_state())
  # Next target differs from the matched target; its refreshed diagnostics must
  # not overwrite the snapshot belonging to the just-finished transition.
  cmd.anchor_pos_w[:, 0] += 1
  cmd._update_metrics()
  assert cmd.metrics["error_anchor_pos"].tolist() == [1., 1.]
  assert snap["tracking"]["error_anchor_pos"].tolist() == [0., 0.]
  untouched = cmd.robot_anchor_pos_w[1].clone()
  cmd.robot_anchor_pos_w[0].zero_()
  for value in cmd.metrics.values(): value[0] = 0
  torch.testing.assert_close(cmd.robot_anchor_pos_w[1], untouched)
  assert cmd.metrics["error_anchor_pos"][1] == 1
  assert snap["tracking"]["error_anchor_pos"].tolist() == [0., 0.]


def test_metric_units_and_joint_l2_are_not_per_joint_rmse():
  cmd = command_at_target()
  cmd.anchor_pos_w[:, 0] = 2
  cmd.body_pos_relative_w[:, 0, 0] = 14
  cmd.joint_pos[:, :2] = torch.tensor([3., 4.])
  cmd.joint_vel[:, :2] = torch.tensor([6., 8.])
  cmd.anchor_quat_w[:] = torch.tensor([0., 1., 0., 0.])
  snap = capture_step_metrics(cmd, {}, {}, {}, .02)["tracking"]
  torch.testing.assert_close(snap["error_anchor_pos"], torch.full((2,), 2.))
  torch.testing.assert_close(snap["error_body_pos"], torch.ones(2))
  torch.testing.assert_close(snap["error_joint_pos"], torch.full((2,), 5.))
  torch.testing.assert_close(snap["error_joint_vel"], torch.full((2,), 10.))
  torch.testing.assert_close(snap["error_anchor_rot"], torch.full((2,), torch.pi))


def test_reward_weights_dt_nonfinite_and_termination_overlap():
  cmd = command_at_target()
  raw = {"tracking": torch.tensor([1., float("nan")], requires_grad=True),
         "action_rate_l2": torch.tensor([4., float("inf")]),
         "joint_limit": torch.tensor([.5, float("-inf")]),
         "disabled": torch.tensor([float("nan"), float("inf")])}
  weights = {"tracking": .5, "action_rate_l2": -.1, "joint_limit": -2., "disabled": 0.}
  termination = {"motion_end": torch.tensor([True, False]), "anchor_pos": torch.tensor([True, False])}
  snap = capture_step_metrics(cmd, raw, weights, termination, .02)
  torch.testing.assert_close(sum(snap["reward"].values()), torch.tensor([-.018, 0.]))
  assert all(not value.requires_grad for group in snap.values() for value in group.values())
  assert snap["termination"]["motion_end"][0] and snap["termination"]["anchor_pos"][0]
  for value in termination.values(): value.zero_()
  with torch.no_grad():
    for value in raw.values(): value.zero_()
  assert snap["termination"]["motion_end"].tolist() == [True, False]
  torch.testing.assert_close(snap["reward"]["tracking"], torch.tensor([.01, 0.]))
