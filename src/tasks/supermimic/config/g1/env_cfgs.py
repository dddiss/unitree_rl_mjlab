"""Shared G1 setup and explicit base/terrain SuperMimic task overrides."""

from dataclasses import replace
import math

from src.assets.robots.unitree_g1.g1_constants import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef
from mjlab.sensor.raycast_sensor import GridPatternCfg
from mjlab.managers.reward_manager import RewardTermCfg
from ...mdp import rewards as mdp
from ...mdp import MotionCommandCfg

from ...tracking_env_cfg import make_tracking_env_cfg
from .terrain_cfg import unitree_g1_terrain_cfg
from ...terrain.reset import TerrainResetCfg
from ...terrain.heightmap import mark_terrain_padding_ray_group
from ...terrain.foot_sensor import WorldUpTerrainHeightSensorCfg
from ...perception.camera import DepthCameraCfg


def _make_g1_env_cfg() -> ManagerBasedRlEnvCfg:
  """Apply robot-specific settings to the common tracking environment."""
  cfg = make_tracking_env_cfg()
  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "pelvis"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r".*_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.viewer.body_name = "torso_link"
  return cfg


def _configure_play(cfg):
  """Apply deterministic playback/evaluation settings after task overrides."""
  motion_cmd = cfg.commands["motion"]
  cfg.episode_length_s = int(1e9)
  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  motion_cmd.joint_position_range = (0.0, 0.0)
  motion_cmd.sampling_mode = "start"
  cfg.events.pop("base_com")
  cfg.events.pop("encoder_bias")
  cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
  # Ground friction is part of the deterministic evaluation setup too.
  cfg.scene.terrain = replace(cfg.scene.terrain, geoms=tuple(
    replace(g, friction=(1.0,)) for g in cfg.scene.terrain.geoms
  ))


def unitree_g1_env_cfg(play=False, terrain_type=None, collect_foot_rewards=False):
  """Main's original plane, eight rewards, and nonvisual training defaults."""
  cfg = _make_g1_env_cfg()
  # The shared tracking template already supplies this task's plane/rewards.
  if play:
    _configure_play(cfg)
  return cfg


def unitree_g1_terrain_env_cfg(play=False, terrain_type=None, collect_foot_rewards=False):
  """Current terrain/perception task, including its measured reward targets."""
  cfg = _make_g1_env_cfg()
  cfg.scene.sensors += (DepthCameraCfg().sensor_cfg(),)
  cfg.scene.terrain = unitree_g1_terrain_cfg(terrain_type)
  cfg.sim.nconmax = 128
  cfg.sim.njmax = 512

  # Pelvis has its own root rewards and must not dilute body-relative errors.
  # Lower-body tolerances apply only to non-flat terrain families; plane and
  # the generated flat column use zero tolerance in both real and imagined steps.
  upper_bodies = (
    "torso_link", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
  )
  lower_bodies = (
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
  )
  # Keep reward order/weights; only the anchor position target gains terrain Z.
  cfg.rewards = {
    "motion_global_root_pos": replace(
      cfg.rewards["motion_global_root_pos"], func=mdp.motion_terrain_anchor_position_error_exp),
    "motion_global_root_ori": cfg.rewards["motion_global_root_ori"],
    "motion_upper_body_pos": RewardTermCfg(
      func=mdp.motion_root_relative_body_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3,
              "body_names": upper_bodies, "tolerance": 0.0},
    ),
    "motion_lower_body_pos": RewardTermCfg(
      func=mdp.motion_root_relative_body_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3,
              "body_names": lower_bodies, "tolerance": 0.05},  # metres per body
    ),
    "motion_upper_body_ori": RewardTermCfg(
      func=mdp.motion_body_orientation_tolerance_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4,
              "body_names": upper_bodies, "tolerance": 0.0},
    ),
    "motion_lower_body_ori": RewardTermCfg(
      func=mdp.motion_body_orientation_tolerance_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4,
              "body_names": lower_bodies, "tolerance": math.radians(10.0)},
    ),
    "motion_body_lin_vel": cfg.rewards["motion_body_lin_vel"],
    "motion_body_ang_vel": cfg.rewards["motion_body_ang_vel"],
    "action_rate_l2": cfg.rewards["action_rate_l2"],
    "joint_limit": cfg.rewards["joint_limit"],
    # Measured P-BFM costs supervise two outputs of one WM head even at weight=0.
    # Keep these reward weights unchanged; use negative weights to penalize.
    "feet_stumble": RewardTermCfg(
      func=mdp.feet_stumble, weight=-10.0,
      params={"sensor_name": "foot_collision_contacts", "horizontal_ratio": 1.0,
              "force_threshold": 20.0, "force_scale": 100.0, "max_penalty": 1.0},
    ),
    "feet_at_plane": RewardTermCfg(
      func=mdp.feet_at_plane, weight=0.0,
      params={"sensor_name": "foot_height_scanners", "contact_sensor_name": "foot_support_contacts",
              "height_tolerance": 0.03, "transition_width": 0.005, "contact_threshold": 1.0},
    ),
  }

  # Task-owned sensors also support weight-zero reward supervision. The
  # collect_foot_rewards flag controls label capture in SuperMimicEnv only.
  feet = ("left_ankle_roll_link", "right_ankle_roll_link")
  cfg.scene.sensors += (
    ContactSensorCfg(
      name="foot_collision_contacts",
      primary=ContactMatch(mode="geom", pattern=r"(left|right)_foot.*_collision", entity="robot"),
      fields=("force", "normal", "tangent"), reduce="maxforce", num_slots=2,
      global_frame=True, history_length=cfg.decimation,
    ),
    ContactSensorCfg(
      name="foot_support_contacts",
      primary=ContactMatch(mode="body", pattern=feet, entity="robot"),
      fields=("force",), reduce="netforce", history_length=cfg.decimation,
    ),
    WorldUpTerrainHeightSensorCfg(
      name="foot_height_scanners",
      frame=tuple(ObjRef(type="body", name=name, entity="robot") for name in feet),
      pattern=GridPatternCfg(size=(0.23, 0.04), resolution=0.01),
      ray_alignment="base", local_center=(0.025, 0.0, -0.045),
      ray_start_height=20.0, max_distance=25.0,
      include_geom_groups=(5,) if cfg.scene.terrain.terrain_type == "generator" else (0,),
      debug_vis=False,
    ),
  )

  motion_cmd = cfg.commands["motion"]
  if cfg.scene.terrain.terrain_type == "generator":
    motion_cmd.terrain_reset = TerrainResetCfg(randomize_xy=not play)
    if cfg.scene.terrain.terrain_generator.border_width > 0:
      cfg.scene.spec_fn = mark_terrain_padding_ray_group

  if play:
    _configure_play(cfg)
  return cfg
