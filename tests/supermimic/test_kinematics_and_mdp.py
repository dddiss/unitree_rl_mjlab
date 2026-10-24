import numpy as np
import torch
import mujoco
from supermimic.mdp.rotations import mat_to_quat,quat_to_mat,so3_exp,so3_log
from supermimic.mdp.state import flatten_state,unflatten_state


def test_fk_against_mujoco(robot):
  model,joints,bodies,kin=robot
  rng=np.random.default_rng(1)
  for _ in range(12):
    d=mujoco.MjData(model)
    d.qpos[:3]=rng.normal(size=3)
    d.qpos[3:7]=rng.normal(size=4);d.qpos[3:7]/=np.linalg.norm(d.qpos[3:7])
    d.qpos[7:]=rng.uniform(-.3,.3,29)
    d.qvel[:]=rng.normal(size=model.nv)
    mujoco.mj_forward(model,d)
    root=model.body('pelvis').id
    vel=np.zeros(6);mujoco.mj_objectVelocity(model,d,mujoco.mjtObj.mjOBJ_XBODY,root,vel,0)
    ten=lambda x:torch.tensor(np.array(x),dtype=torch.float32)[None]
    state=(ten(d.xpos[root]),quat_to_mat(ten(d.xquat[root])),ten(vel[3:]),ten(vel[:3]),ten(d.qpos[7:]),ten(d.qvel[6:]))
    bp,br,bv,bw=kin.forward(*state)
    refvel=np.zeros((len(bodies),6))
    for i,b in enumerate(bodies):mujoco.mj_objectVelocity(model,d,mujoco.mjtObj.mjOBJ_XBODY,b,refvel[i],0)
    for pred,ref in ((bp,d.xpos[bodies]),(br,d.xmat[bodies].reshape(-1,3,3)),(bv,refvel[:,3:]),(bw,refvel[:,:3])):
      torch.testing.assert_close(pred,ten(ref),rtol=1e-4,atol=1e-5)
    torch.testing.assert_close(flatten_state(unflatten_state(flatten_state(state))),flatten_state(state))


def test_shared_mdp_and_gradients(mdp):
  f=torch.tensor([1,4])
  p,r,v,w=mdp.reference_root(f)
  q=mdp.motion.joint_pos[f].clone().requires_grad_()
  state=(p,r,v,w,q,mdp.motion.joint_vel[f])
  a=torch.zeros(2,29,requires_grad=True);xi=torch.zeros(2,33);xi[:,1:30]=.003
  obs=mdp.obs(state,a,xi,f)
  torch.testing.assert_close(obs[:,:29],q+.003)
  assert obs.shape==(2,93) and not mdp.termination(state,f).any()
  reward=mdp.reward(state,f,a,torch.ones_like(a)*.1)
  terms=mdp.reward_terms(state,f,a,torch.ones_like(a)*.1)
  torch.testing.assert_close(reward,sum(mdp.cfg.weights[k]*v for k,v in terms.items())*.02)
  reward.sum().backward()
  assert torch.isfinite(q.grad).all() and torch.isfinite(a.grad).all()
  assert a.grad.abs().sum()>0


def test_proprioception_is_independent_of_reference(mdp):
  f=torch.tensor([1,4])
  p,_,v,w=mdp.reference_root(f)
  rotation_vectors=torch.tensor([[0.,0.,torch.pi/2],[torch.pi/2,0.,0.]],requires_grad=True)
  state=(p,so3_exp(rotation_vectors),v,w,mdp.motion.joint_pos[f],mdp.motion.joint_vel[f])
  action=torch.linspace(-.2,.2,58).reshape(2,29)
  xi=torch.zeros(2,33)
  obs=mdp.obs(state,action,xi,f)
  goal=mdp.goal(state,f)
  # Change only the reference, keeping the physical state and previous action fixed.
  half_sqrt=2**-.5
  mdp.motion.body_quat_w[f,mdp.root_body_index]=torch.tensor(
    [[half_sqrt,half_sqrt,0.,0.],[half_sqrt,0.,0.,half_sqrt]])
  changed=mdp.obs(state,action,xi,f)
  torch.testing.assert_close(changed,obs)
  torch.testing.assert_close(mdp.obs(state,action,xi,f+1),obs)
  assert not torch.equal(mdp.goal(state,f),goal)
  assert obs.shape==(2,93)
  torch.testing.assert_close(obs[:,64:93],action)
  noise_bounds=torch.tensor([.01]*29+[.5]*29+[.2]*3+[.05]*3+[0.]*29)
  torch.testing.assert_close(mdp.obs_noise,noise_bounds)
  noisy=mdp.add_noise(obs)
  assert noisy.shape==obs.shape
  assert ((noisy-obs).abs()<=noise_bounds+1e-6).all()
  torch.testing.assert_close(noisy[:,64:93],action)


def test_ten_reference_orientations_use_same_current_robot(mdp):
  f=torch.tensor([2,20])
  m=mdp.motion
  frames=torch.arange(m.num_frames,dtype=torch.float32)
  reference_rotations=so3_exp(torch.stack((.2*frames.sin(),.3*frames.cos(),.07*frames),-1))
  m.body_quat_w[:,mdp.root_body_index]=mat_to_quat(reference_rotations)
  m.joint_pos[:]=frames[:,None]*.002+torch.arange(29)*.01
  m.joint_vel[:]=frames[:,None]*.03-torch.arange(29)*.02
  # Both robot rotations have roll, pitch and yaw; this rejects yaw-only bases.
  robot_angles=torch.tensor([[.3,-.2,.7],[-.4,.5,-.1]],requires_grad=True)
  p,_,v,w=mdp.reference_root(f-1)
  robot_rotation=so3_exp(robot_angles)
  state=(p,robot_rotation,v,w,m.joint_pos[f-1],m.joint_vel[f-1])
  goal=mdp.goal(state,f).reshape(2,10,64)
  targets=f[:,None]+torch.arange(10)
  torch.testing.assert_close(goal[:,:,:29],m.joint_pos[targets])
  torch.testing.assert_close(goal[:,:,29:58],m.joint_vel[targets])
  # A 6D rotation stores the reference x/y axes expressed in the robot frame.
  for batch in range(2):
    for offset in range(10):
      axes=reference_rotations[targets[batch,offset],:,:2]
      expected=robot_rotation[batch].T@axes
      torch.testing.assert_close(goal[batch,offset,58:].reshape(3,2),expected,atol=1e-6,rtol=1e-5)
  changed_state=(p,so3_exp(robot_angles+.1),v,w,state[4],state[5])
  changed=mdp.goal(changed_state,f).reshape(2,10,64)
  torch.testing.assert_close(changed[:,:,:58],goal[:,:,:58])
  assert not torch.allclose(changed[:,:,58:],goal[:,:,58:])
  (goal[:,:,58:]*torch.arange(1.,7.)).sum().backward()
  assert torch.isfinite(robot_angles.grad).all()
  assert (robot_angles.grad.abs().sum(-1)>0).all()


def test_rotation_gradients():
  for value in (0.,1e-7,3.141592-1e-5):
    x=torch.tensor([[value,0.,0.]],requires_grad=True)
    y=so3_log(so3_exp(x))
    y.sum().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()


def test_critic_features_include_one_previous_action_and_no_duplicate_rotation(mdp):
  f=torch.tensor([1,4])
  state=(*mdp.reference_root(f),mdp.motion.joint_pos[f],mdp.motion.joint_vel[f])
  xi=torch.linspace(-.3,.4,66).reshape(2,33)
  previous=torch.linspace(-.2,.5,58).reshape(2,29).requires_grad_()
  features=mdp.critic_features(state,f,xi,previous)
  # 71 state + 29 joint error + 42 body error + 3 root position error,
  # then 29 previous action and 33 context. Root rotation stays in goal only.
  assert features.shape==(2,207)
  torch.testing.assert_close(features[:,145:174],previous)
  torch.testing.assert_close(features[:,174:],xi)
  changed=mdp.critic_features(state,f,xi,previous+.25)
  torch.testing.assert_close(changed[:,:145],features[:,:145])
  torch.testing.assert_close(changed[:,174:],features[:,174:])
  torch.testing.assert_close(changed[:,145:174]-features[:,145:174],torch.full_like(previous,.25))
  features.sum().backward()
  torch.testing.assert_close(previous.grad,torch.ones_like(previous))


def test_goal_noise_only_changes_orientation_without_mutation_or_gradient_loss(mdp):
  # Include leading trajectory/batch axes and a non-default reference count.
  goal=torch.linspace(-2.,2.,2*3*4*64).reshape(2,3,4*64).requires_grad_()
  original=goal.detach().clone()
  noisy=mdp.add_goal_noise(goal)
  assert noisy.shape==goal.shape
  torch.testing.assert_close(goal,original,rtol=0,atol=0)
  delta=(noisy-goal).reshape(2,3,4,64)
  torch.testing.assert_close(delta[...,:58],torch.zeros_like(delta[...,:58]),rtol=0,atol=0)
  orientation_noise=delta[...,58:]
  assert orientation_noise.abs().max()<=.05+1e-6
  assert (orientation_noise.abs().sum(-1)>0).all()
  assert orientation_noise.min()<0 and orientation_noise.max()>0
  assert not torch.equal(noisy,mdp.add_goal_noise(goal))
  noisy.sum().backward()
  torch.testing.assert_close(goal.grad,torch.ones_like(goal))
