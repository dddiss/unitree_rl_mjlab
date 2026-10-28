"""Unitree G1 SuperMimic environment with P-BFM collision terrain."""

from src.assets.robots.unitree_g1.g1_constants import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from src.tasks.supermimic.mdp import MotionCommandCfg

from src.tasks.supermimic.tracking_env_cfg import make_tracking_env_cfg
from .terrain_cfg import unitree_g1_terrain_cfg
from ...terrain.pattern import RobotCentricGridPatternCfg
from ...terrain.reset import TerrainResetCfg
from ...terrain.sensor import PbfmTerrainHeightSensorCfg
from mjlab.sensor import ObjRef


def unitree_g1_flat_tracking_env_cfg(
  play: bool = False,
  depth_camera=None,
  terrain_type: str | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create G1 tracking configuration (legacy factory name retained)."""
  cfg = make_tracking_env_cfg()
  cfg.scene.terrain = unitree_g1_terrain_cfg(terrain_type)

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
  if depth_camera is not None:
    cfg.scene.sensors += (depth_camera.sensor_cfg(),)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if cfg.scene.terrain.terrain_type == "generator":
    motion_cmd.terrain_reset = TerrainResetCfg()
    cfg.scene.sensors += (PbfmTerrainHeightSensorCfg(
      name=motion_cmd.terrain_reset.sensor_name,
      frame=ObjRef(type="body", name="pelvis", entity="robot"),
      pattern=RobotCentricGridPatternCfg(), ray_alignment="yaw",
      include_geom_groups=(5,), reduction="none",
    ),)
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

  if play:
    cfg.episode_length_s = int(1e9)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.sampling_mode = "start"
    cfg.events.pop("base_com")
    cfg.events.pop("encoder_bias")
    cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
    # Ground friction is part of the deterministic evaluation setup too.
    from dataclasses import replace
    cfg.scene.terrain = replace(cfg.scene.terrain, geoms=tuple(
      replace(g, friction=(1.0,)) for g in cfg.scene.terrain.geoms
    ))

  return cfg
