"""One differentiable imagined step and its shared reward/failure evaluation."""
import torch
from ..core.state import normalize_xi

def reward_term(self, state, f, action, a_prev, foot_costs=None, *, terrain_tolerance=None, height_offset=None):
  reward = self.mdp.reward(state, f, action, a_prev, terrain_tolerance=terrain_tolerance)
  if foot_costs is not None:
    for i, name in enumerate(self.foot_reward_names):
      reward = reward + self.mdp.cfg.weights[name] * foot_costs[..., i] * self.mdp.cfg.step_dt
  return reward, self.mdp.soft_termination(state, f, height_offset=height_offset)


def next_target(self, f):
  m = self.mdp.motion
  return torch.minimum(f + 1, m.clip_end[m.clip_id[f]])


def _imagine_step(self, state, history, f, xi, goal, depth_latent=None):
  if self.cfg.depth_enabled:
    action = self.policy(history, self.mdp.add_goal_noise(goal), depth_latent=depth_latent)
  else:
    action = self.policy(history, self.mdp.add_goal_noise(goal))
  if self.foot_reward_names:
    next_state, next_latent, costs = self.wm.step_with_foot_costs(
      state, action, normalize_xi(xi), depth_latent)
  elif self.cfg.depth_enabled:
    next_state, next_latent = self.wm.step_visual(state, action, normalize_xi(xi), depth_latent)
  else:
    next_state = self.wm.step(state, action, normalize_xi(xi))
  # Reward evaluates next_state against f, BEFORE changing the target.
  next_f = self.next_target(f)
  obs = self.mdp.add_noise(self.mdp.obs(next_state, action, xi, next_f))
  history = torch.cat((history[:, 1:], obs[:, None]), 1)
  # Reference orientations are relative to the predicted robot at this step.
  # Keep a clean goal and its gradient for the critic; only the policy adds noise.
  next_goal = self.mdp.goal(next_state, next_f, self.cfg.goal_frames)
  result = next_state, action, history, next_f, next_goal
  result = (*result, next_latent) if self.cfg.depth_enabled else result
  return (*result, costs) if self.foot_reward_names else result
