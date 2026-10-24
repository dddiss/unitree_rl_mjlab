"""CPU tests import the tensor-only package without triggering the PPO registry."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
import mujoco

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/tasks"))
from supermimic.mdp.motion_lib import MotionLibrary
from supermimic.mdp.kinematics import G1Kinematics
from supermimic.mdp.functional import FunctionalMDP

BODIES = ('pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link',
          'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'torso_link',
          'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_yaw_link',
          'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_yaw_link')


@pytest.fixture(scope="session")
def robot():
  model = mujoco.MjModel.from_xml_path(str(ROOT / 'src/assets/robots/unitree_g1/xmls/g1.xml'))
  joints = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
  bodies = [model.body(n).id for n in BODIES]
  kin = G1Kinematics(model, joints, bodies, model.body('pelvis').id)
  return model, joints, bodies, kin


@pytest.fixture
def mdp(tmp_path, robot):
  model, joints, bodies, kin = robot
  for clip, length in enumerate((51, 7)):
    arrays = {k: [] for k in ('joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w')}
    for frame in range(length):
      d = mujoco.MjData(model)
      d.qpos[2] = 0.8
      d.qpos[0] = 0.002 * frame + clip * 10
      d.qvel[0] = 0.1
      mujoco.mj_forward(model,d)
      vel = np.zeros((model.nbody-1,6))
      for b in range(1, model.nbody):
        mujoco.mj_objectVelocity(model,d,mujoco.mjtObj.mjOBJ_XBODY,b,vel[b-1],0)
      values = (d.qpos[7:].copy(),d.qvel[6:].copy(),d.xpos[1:].copy(),d.xquat[1:].copy(),vel[:,3:].copy(),vel[:,:3].copy())
      for key,value in zip(arrays,values): arrays[key].append(value)
    np.savez(tmp_path/f'motion{clip}.npz',fps=[50],**{k:np.asarray(v,dtype=np.float32) for k,v in arrays.items()})
  motion = MotionLibrary(tmp_path)
  return FunctionalMDP(motion,kin,torch.zeros(29),[b-1 for b in bodies],model.body('pelvis').id-1)
