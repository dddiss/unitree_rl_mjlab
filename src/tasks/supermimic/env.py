"""mjlab 1.6 environment with explicit reset and post-integration RL signals."""
import torch

from mjlab.envs import ManagerBasedRlEnv

from .mdp.functional import FunctionalMDP, MdpConfig
from .mdp.kinematics import G1Kinematics
from .mdp.logging_metrics import capture_step_metrics
from .mdp.state import read_state, flatten_state


class SuperMimicEnv(ManagerBasedRlEnv):
  def __init__(self, cfg, **kwargs):
    if cfg.auto_reset:
      raise ValueError("SuperMimic requires auto_reset=False")
    self._functional_mdp = self._context = self._reward_terms = None
    self.last_transition = None
    self.last_step_metrics = None
    super().__init__(cfg, **kwargs)
    self._context = self.read_context().clone()
    self.reset_version = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

  @property
  def context(self):
    # ObservationManager probes shapes before startup randomization has run.
    return self.read_context() if self._context is None else self._context

  def read_context(self):
    cmd = self.command_manager.get_term("motion")
    robot = cmd.robot
    collision = next(i for i, n in enumerate(robot.geom_names) if n.endswith("_collision"))
    geom_id = robot.indexing.geom_ids[collision]
    torso = robot.indexing.body_ids[robot.body_names.index("torso_link")]
    mu = self.sim.model.geom_friction[:, geom_id, 0].expand(self.num_envs)
    com = self.sim.model.body_ipos[:, torso] - self.sim.get_default_field("body_ipos")[torso]
    return torch.cat((mu[:, None], robot.data.encoder_bias,
                      com.expand(self.num_envs, -1)), -1)

  @property
  def functional_mdp(self):
    if self._functional_mdp is None:
      cmd = self.command_manager.get_term("motion")
      robot = cmd.robot
      kin = G1Kinematics(self.sim.mj_model, robot.indexing.joint_ids.tolist(),
                         robot.indexing.body_ids[cmd.body_indexes].tolist(),
                         int(robot.indexing.body_ids[cmd.root_body_index]), self.device)
      cfg = MdpConfig(weights={k: v.weight for k, v in self.cfg.rewards.items()}, step_dt=self.step_dt)
      for key, attribute in (("motion_global_root_pos", "root_pos_std"),
                             ("motion_global_root_ori", "root_ori_std"),
                             ("motion_body_pos", "body_pos_std"), ("motion_body_ori", "body_ori_std"),
                             ("motion_body_lin_vel", "body_linvel_std"),
                             ("motion_body_ang_vel", "body_angvel_std")):
        setattr(cfg, attribute, self.cfg.rewards[key].params["std"])
      cfg.term_height = self.cfg.terminations["anchor_pos"].params["threshold"]
      cfg.term_tilt = self.cfg.terminations["anchor_ori"].params["threshold"]
      self._functional_mdp = FunctionalMDP(cmd.motion, kin, robot.data.default_joint_pos[0],
                                           cmd.body_indexes, cmd.root_body_index,
                                           robot.data.soft_joint_pos_limits[0], cfg)
    return self._functional_mdp

  def shared_reward_terms(self):
    if self._reward_terms is None:
      cmd = self.command_manager.get_term("motion")
      self._reward_terms = self.functional_mdp.reward_terms(
        read_state(self), cmd.time_steps, self.action_manager.action, self.action_manager.prev_action)
    return self._reward_terms

  @torch.no_grad()
  def reset(self, *, seed=None, env_ids=None, options=None):
    self._reward_terms = None
    result = super().reset(seed=seed, env_ids=env_ids, options=options)
    self.reset_version[slice(None) if env_ids is None else env_ids] += 1
    return result

  @torch.no_grad()
  def step(self, action):
    """Mirrors mjlab 1.6's manual-reset lifecycle, forwarding BEFORE rewards."""
    if torch.any(self._manual_reset_pending):
      raise RuntimeError("Reset finished environments before the next SuperMimic step")
    cmd = self.command_manager.get_term("motion")
    transition = {
      "s": flatten_state(read_state(self)).clone(), "f": cmd.time_steps.clone(),
      "a_prev": self.action_manager.action.clone(), "ep_step": self.episode_length_buf.clone(),
    }
    self.extras["log"] = {}
    self._reward_terms = None
    self.action_manager.process_action(action.to(self.device))
    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)
      self.metrics_manager.compute_substep()
    self.episode_length_buf += 1
    self.common_step_counter += 1
    self.sim.forward()
    cmd.update_relative_body_poses()
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs
    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    self.last_step_metrics = capture_step_metrics(
      cmd, self.shared_reward_terms(),
      {name: self.reward_manager.get_term_cfg(name).weight for name in self.reward_manager.active_terms},
      {name: self.termination_manager.get_term(name) for name in self.termination_manager.active_terms},
      self.step_dt,
    )
    self.metrics_manager.compute()
    transition.update(s1=flatten_state(read_state(self)).clone(),
                      a=self.action_manager.action.clone(), r=self.reward_buf.clone(),
                      terminated=self.reset_terminated.clone(), truncated=self.reset_time_outs.clone())
    self.last_transition = transition
    # The scoped configuration only contains startup events. Fail explicitly if
    # a later extension adds post-step perturbations without defining s1 timing.
    if any(mode in self.event_manager.available_modes for mode in ("step", "interval")):
      raise ValueError("SuperMimic step/interval events need an explicit transition convention")
    # A finished row keeps its evaluated target until the caller resets it.
    self.command_manager.compute(dt=torch.where(self.reset_buf, 0.0, self.step_dt))
    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)
    self._manual_reset_pending[self.reset_buf] = True
    self.recorder_manager.record_post_step()
    return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
