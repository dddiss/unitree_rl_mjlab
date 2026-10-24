"""Read-only snapshots of the signals evaluated against the current target."""
from collections.abc import Mapping

import torch


TRACKING_METRICS = (
  "error_anchor_pos", "error_anchor_rot", "error_anchor_lin_vel", "error_anchor_ang_vel",
  "error_body_pos", "error_body_rot", "error_body_lin_vel", "error_body_ang_vel",
  "error_joint_pos", "error_joint_vel",
)


@torch.no_grad()
def capture_step_metrics(
  command,
  raw_reward_terms: Mapping[str, torch.Tensor],
  reward_weights: Mapping[str, float],
  termination_terms: Mapping[str, torch.Tensor],
  step_dt: float,
) -> dict[str, dict[str, torch.Tensor]]:
  """Capture per-environment values before advancing targets or resetting.

  The command already has post-physics relative body poses. Its metric update
  only refreshes diagnostic tensors; it neither advances frames nor samples.
  All returned tensors own their storage, so later command/reset writes cannot
  change a transition's log values. Reward scaling matches mjlab 1.6.0's
  default RewardManager, including zeroing non-finite weighted contributions.
  """
  command._update_metrics()
  return {
    "tracking": {name: command.metrics[name].detach().clone() for name in TRACKING_METRICS},
    "reward": {
      name: torch.nan_to_num(raw_reward_terms[name].detach() * weight * step_dt,
                             nan=0.0, posinf=0.0, neginf=0.0)
      for name, weight in reward_weights.items()
    },
    "termination": {name: value.detach().clone() for name, value in termination_terms.items()},
  }
