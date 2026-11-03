"""mjlab 1.6 environment with explicit reset and post-integration RL signals."""
import torch

from mjlab.envs import ManagerBasedRlEnv

from .core.functional import FunctionalMDP, MdpConfig
from .core.components import prediction_bindings, bind_analytic_manager_terms
from .tools.kinematics import G1Kinematics
from .rl.logging import capture_step_metrics
from .core.state import read_state, flatten_state, state_in_reference_frame
from .terrain.heightmap import TerrainHeightMap
from .perception.camera import DepthCamera, DepthPatternCfg


def tolerance_by_terrain_column(cfg):
  """mjlab 1.6 curriculum maps sub_terrains insertion order to columns.

  This identifies the configured family, not whether a particular point on a
  stair/platform happens to be level. The G1 factory names its flat family flat.
  """
  if cfg is None or cfg.terrain_type == "plane":
    return (False,)
  if cfg.terrain_type != "generator" or cfg.terrain_generator is None:
    raise ValueError("Terrain-dependent tolerance requires plane or generated terrain")
  generator = cfg.terrain_generator
  if not generator.curriculum:
    # Random generation can put different families in the same column; mjlab
    # 1.6 does not expose their per-tile family IDs on TerrainEntity.
    raise ValueError("Terrain-dependent tolerance requires curriculum=True (one family per column)")
  return tuple(name != "flat" for name in generator.sub_terrains)


class SuperMimicEnv(ManagerBasedRlEnv):
  def __init__(self, cfg, *, collect_foot_rewards=False,
               randomize_terrain_on_reset=False, **kwargs):
    if cfg.auto_reset:
      raise ValueError("SuperMimic requires auto_reset=False")
    self._functional_mdp = self._context = self._reward_terms = None
    self._terrain_tolerance_by_type = None
    self.last_transition = None
    self.last_step_metrics = None
    self._depth_camera = None
    self._depth_image = None
    self._randomize_terrain_on_reset = randomize_terrain_on_reset
    self.has_episode_time_limit = any(
      term is not None and term.time_out
      and getattr(term.func, "supermimic_truncation", None) == "time_limit"
      for term in cfg.terminations.values())
    bindings = prediction_bindings(cfg.rewards)
    self.foot_reward_names = tuple(name for name, _, _ in bindings) if collect_foot_rewards else ()
    self._sense_before_rewards = any(component.sense_before_reward and
      (term.weight != 0 or collect_foot_rewards) for _, term, component in bindings)
    bind_analytic_manager_terms(cfg)
    super().__init__(cfg, **kwargs)
    # The task owns sensor creation. Reuse its calibration for image readout.
    depth_camera = next((sensor.pattern.camera for sensor in cfg.scene.sensors
                         if isinstance(getattr(sensor, "pattern", None), DepthPatternCfg)), None)
    if depth_camera is not None:
      self._depth_camera = DepthCamera(depth_camera, self.scene, self.device)
    self._context = self.read_context().clone()
    self.reset_version = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

  def load_managers(self):
    # Managers probe observations/terminations during construction. Build the
    # shared map first, before any reset, replay collection or imagined rollout.
    self.terrain_height_map = None
    terrain = self.scene.terrain
    reset_cfg = self.cfg.commands["motion"].terrain_reset
    if terrain is not None and terrain.cfg.terrain_type == "generator":
      self.terrain_height_map = TerrainHeightMap.from_model(
        self.sim.mj_model, self.device, resolution=reset_cfg.heightmap_resolution)
    super().load_managers()

  @property
  def depth_enabled(self):
    return self._depth_camera is not None

  @property
  def depth_image(self):
    if self._depth_image is None:
      raise RuntimeError("Reset a depth-enabled environment before requesting depth")
    return self._depth_image

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
  def terrain_tolerance(self):
    """Per-environment reward flag; read current assignments after any reset."""
    terrain = self.scene.terrain
    if self._terrain_tolerance_by_type is None:
      flags = tolerance_by_terrain_column(None if terrain is None else terrain.cfg)
      self._terrain_tolerance_by_type = torch.tensor(flags, dtype=torch.bool, device=self.device)
    if terrain is None or terrain.cfg.terrain_type == "plane":
      return self._terrain_tolerance_by_type.expand(self.num_envs)
    return self._terrain_tolerance_by_type[terrain.terrain_types]

  @property
  def functional_mdp(self):
    if self._functional_mdp is None:
      cmd = self.command_manager.get_term("motion")
      robot = cmd.robot
      kin = G1Kinematics(self.sim.mj_model, robot.indexing.joint_ids.tolist(),
                         robot.indexing.body_ids[cmd.body_indexes].tolist(),
                         int(robot.indexing.body_ids[cmd.root_body_index]), self.device)
      cfg = MdpConfig.from_env_cfg(self.cfg.rewards, self.cfg.terminations, self.step_dt,
                                  body_names=cmd.cfg.body_names)
      self._functional_mdp = FunctionalMDP(cmd.motion, kin, robot.data.default_joint_pos[0],
                                           cmd.body_indexes, cmd.root_body_index,
                                           robot.data.soft_joint_pos_limits[0], cfg)
      terrain = self.scene.terrain
      if self._functional_mdp.requires_height_offset and terrain is not None and terrain.cfg.terrain_type == "generator":
        self._functional_mdp.termination_ground = self.terrain_height_map.height
    return self._functional_mdp

  def shared_reward_terms(self):
    if self._reward_terms is None:
      cmd = self.command_manager.get_term("motion")
      mdp = self.functional_mdp
      offset = (mdp.termination_height_offset(cmd.time_steps, self.scene.env_origins,
                                             cmd.reference_xy_offset)
                if mdp.requires_reward_height_offset else None)
      self._reward_terms = mdp.reward_terms(
        state_in_reference_frame(read_state(self), cmd.reference_xy_offset),
        cmd.time_steps, self.action_manager.action, self.action_manager.prev_action,
        terrain_tolerance=self.terrain_tolerance, height_offset=offset)
    return self._reward_terms

  def shared_failure_terms(self):
    cmd = self.command_manager.get_term("motion")
    mdp = self.functional_mdp
    offset = (mdp.termination_height_offset(cmd.time_steps, self.scene.env_origins,
                                           cmd.reference_xy_offset)
              if mdp.requires_height_offset else None)
    return mdp.failure_terms(state_in_reference_frame(read_state(self), cmd.reference_xy_offset),
                             cmd.time_steps, height_offset=offset)

  @torch.no_grad()
  def reset(self, *, seed=None, env_ids=None, options=None):
    self._reward_terms = None
    result = super().reset(seed=seed, env_ids=env_ids, options=options)
    if self._depth_camera is not None:
      frames = self._depth_camera.read()  # base reset already forwards and senses
      if self._depth_image is None:
        self._depth_image = frames
      else:
        ids = slice(None) if env_ids is None else env_ids
        self._depth_image[ids] = frames[ids]
    self.reset_version[slice(None) if env_ids is None else env_ids] += 1
    return result

  def _reset_idx(self, env_ids=None):
    # Play opts in here so even Viser's unwrapped Start Here reset is covered.
    # Base reset seeds before this hook; commands then use the new origin for k.
    if self._randomize_terrain_on_reset and self.scene.terrain is not None:
      ids = (torch.arange(self.num_envs, device=self.device)
             if env_ids is None else env_ids)
      if ids.numel():
        # mjlab samples difficulty rows and terrain columns independently and
        # uniformly. Plane terrain has no origin grid and is a no-op.
        self.scene.terrain.randomize_env_origins(ids)
    super()._reset_idx(env_ids)

  @torch.no_grad()
  def step(self, action):
    """Mirrors mjlab 1.6's manual-reset lifecycle, forwarding BEFORE rewards."""
    if torch.any(self._manual_reset_pending):
      raise RuntimeError("Reset finished environments before the next SuperMimic step")
    cmd = self.command_manager.get_term("motion")
    transition = {
      "s": flatten_state(read_state(self)).clone(), "f": cmd.time_steps.clone(),
      "a_prev": self.action_manager.action.clone(), "ep_step": self.episode_length_buf.clone(),
      "terrain_tolerance": self.terrain_tolerance.clone(),
      "reference_xy_offset": cmd.reference_xy_offset.clone(),
      "env_origin": self.scene.env_origins.clone(),
    }
    if getattr(self, "_depth_camera", None) is not None:
      transition["depth"] = self.depth_image.clone()
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
    if getattr(self, "_sense_before_rewards", False):
      # Foot support must query post-action ground, not the preceding step's rays.
      self.sim.sense()
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
      reward_contributions={
        name: self.reward_manager._step_reward[:, i] * self.step_dt
        for i, name in enumerate(self.reward_manager.active_terms)
      },
    )
    self.metrics_manager.compute()
    transition.update(s1=flatten_state(read_state(self)).clone(),
                      a=self.action_manager.action.clone(), r=self.reward_buf.clone(),
                      terminated=self.reset_terminated.clone(), truncated=self.reset_time_outs.clone())
    if getattr(self, "foot_reward_names", ()):
      # RewardManager skips weight=0, so evaluate raw sensor labels explicitly.
      # These include terminal contacts/terrain BEFORE target advance or reset.
      values = []
      for name in self.foot_reward_names:
        term = self.reward_manager.get_term_cfg(name)
        values.append(torch.nan_to_num(term.func(self, **term.params).float(),
                                       nan=0.0, posinf=0.0, neginf=0.0))
      transition["foot_costs"] = torch.stack(values, -1).clone()
    self.last_transition = transition
    # The scoped configuration only contains startup events. Fail explicitly if
    # a later extension adds post-step perturbations without defining s1 timing.
    if any(mode in self.event_manager.available_modes for mode in ("step", "interval")):
      raise ValueError("SuperMimic step/interval events need an explicit transition convention")
    # A finished row keeps its evaluated target until the caller resets it.
    self.command_manager.compute(dt=torch.where(self.reset_buf, 0.0, self.step_dt))
    if not getattr(self, "_sense_before_rewards", False):
      self.sim.sense()
    if getattr(self, "_depth_camera", None) is not None:
      self._depth_image = self._depth_camera.read()
      # Includes the terminal transition's real post-action, PRE-reset image.
      transition["depth_next"] = self._depth_image.clone()
    self.obs_buf = self.observation_manager.compute(update_history=True)
    self._manual_reset_pending[self.reset_buf] = True
    self.recorder_manager.record_post_step()
    return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
