"""Checkpoint/collection/evaluation integration with a deterministic tensor environment."""
from types import SimpleNamespace as NS
import torch
import pytest
from supermimic.mdp.state import flatten_state,read_state
from supermimic.mdp.rotations import mat_to_6d,mat_to_quat,so3_exp
from supermimic.mdp.sampling import ReturnBasedSampler
from supermimic.mdp.logging_metrics import TRACKING_METRICS
from supermimic.rl.runner import SuperMimicRunner,PolicyController
from supermimic.rl.evaluation import evaluate,load_policy
from test_algorithm import small_config


class TensorEnv:
  """Controlled transitions; only the physics is fake, runner/algorithm are real."""
  device='cpu';num_envs=2;step_dt=.02
  def __init__(self,mdp):
    self.functional_mdp=mdp;self.context=torch.zeros(2,33);self.context[:,0]=.75
    self.scene=NS(env_origins=torch.tensor([[4.,5.,0.],[9.,2.,0.]]))
    self.reset_version=torch.zeros(2,dtype=torch.long)
    self.episode_length_buf=torch.zeros(2,dtype=torch.long)
    self.action_manager=NS(action=torch.zeros(2,29))
    self.robot=NS(data=NS())
    self.command=NS(robot=self.robot,root_body_index=0,motion=mdp.motion,time_steps=torch.ones(2,dtype=torch.long),
                    start_bin=torch.zeros(2,dtype=torch.long),sampler=ReturnBasedSampler(mdp.motion),motion_ids=torch.zeros(2,dtype=torch.long),
                    cfg=NS(sampling_mode='return_based'))
    self.command.request_frames=self.request_frames
    self.command_manager=NS(get_term=lambda _:self.command)
    self.reward_manager=NS(active_terms=tuple(mdp.cfg.weights))
    self.termination_manager=NS(active_terms=('time_out','motion_end','anchor_pos','anchor_ori'))
    self.requested=None;self.pending=torch.zeros(2,dtype=torch.bool)
    self.reset()
  def request_frames(self,frames,ids=None):
    self.requested=(frames.clone(),torch.arange(2) if ids is None else ids.clone())
  def set_state(self,f):
    m=self.functional_mdp
    p,r,v,w=m.reference_root(f)
    d=self.robot.data
    d.body_link_pose_w=torch.cat((p+self.scene.env_origins,mat_to_quat(r)),-1)[:,None]
    d.body_link_vel_w=torch.cat((v,w),-1)[:,None]
    d.joint_pos=m.motion.joint_pos[f];d.joint_vel=m.motion.joint_vel[f]
  def reset(self,env_ids=None):
    ids=torch.arange(2) if env_ids is None else env_ids
    frames=torch.zeros(len(ids),dtype=torch.long)
    if self.requested is not None:
      frames,ids=self.requested;self.requested=None
    self.command.time_steps[ids]=frames+1
    self.command.start_bin[ids]=self.command.motion.frame_bin[frames]
    self.command.motion_ids=self.command.motion.clip_id[self.command.time_steps]
    self.episode_length_buf[ids]=0;self.reset_version[ids]+=1;self.action_manager.action[ids]=0;self.pending[ids]=False
    # The fake environment uses exact reference states for its remaining rows.
    self.set_state(self.command.time_steps-1)
    return {},{}
  def step(self,a):
    assert not self.pending.any()
    cmd=self.command;m=self.functional_mdp;f=cmd.time_steps.clone()
    state=(*m.reference_root(f-1),m.motion.joint_pos[f-1],m.motion.joint_vel[f-1])
    state1=(*m.reference_root(f),m.motion.joint_pos[f],m.motion.joint_vel[f])
    steps=self.episode_length_buf.clone()
    terminal=torch.zeros(2,dtype=torch.bool)
    truncated=f>=m.motion.clip_end[m.motion.clip_id[f]]
    self.last_transition=dict(s=flatten_state(state),s1=flatten_state(state1),a=a.clone(),a_prev=self.action_manager.action.clone(),
                             f=f,r=m.reward(state1,f,a,self.action_manager.action),ep_step=steps,terminated=terminal,truncated=truncated)
    self.last_step_metrics=dict(
      tracking={name:torch.zeros(2) for name in TRACKING_METRICS},
      reward={key:value*m.cfg.weights[key]*self.step_dt
              for key,value in m.reward_terms(state1,f,a,self.action_manager.action).items()},
      termination={key:(truncated.clone() if key=='motion_end' else terminal.clone())
                   for key in self.termination_manager.active_terms})
    self.episode_length_buf+=1;self.action_manager.action=a.clone()
    cmd.time_steps=torch.minimum(f+1,m.motion.clip_end[m.motion.clip_id[f]])
    self.set_state(f);self.pending=truncated.clone()
    return {},self.last_transition['r'],terminal,truncated,{}


def test_train_resume_refills_buffer(mdp,tmp_path):
  cfg=small_config(iterations=2,warmup_steps=4,wm_only_iters=1,ckpt_every=1,log_every=1)
  env=TensorEnv(mdp);runner=SuperMimicRunner(env,cfg,tmp_path/'run')
  runner.learn()
  assert runner.iteration==2
  saved_pi={k:v.clone() for k,v in runner.algorithm.policy.state_dict().items()}
  cfg.iterations=3
  resumed=SuperMimicRunner(TensorEnv(mdp),cfg,tmp_path/'resume')
  resumed.load(tmp_path/'run/last.pt')
  assert resumed.iteration==2 and resumed.buffer.size==0
  for k,v in resumed.algorithm.policy.state_dict().items():torch.testing.assert_close(v,saved_pi[k])
  previous=resumed.env_steps;resumed.learn()
  assert resumed.iteration==3 and resumed.env_steps>=previous+cfg.warmup_steps
  assert resumed.buffer.size>=cfg.wm_window


def _track_updates(runner,monkeypatch):
  calls=[]
  for name in ('world_model_update','policy_update'):
    original=getattr(runner.algorithm,name)
    def update(buffer,name=name,original=original):
      calls.append((name,runner.iteration,runner.env_steps))
      return original(buffer)
    monkeypatch.setattr(runner.algorithm,name,update)
  return calls


def _track_collection(runner,monkeypatch):
  calls=[];original=runner.collect
  def collect(count):
    original(count)
    calls.append((count,runner.env_steps,runner.buffer.size))
  monkeypatch.setattr(runner,'collect',collect)
  return calls


def test_disabled_warmup_waits_for_window_then_updates_world_model_before_policy(mdp,tmp_path,monkeypatch):
  torch.set_num_threads(2)
  cfg=small_config(enable_warmup=False,iterations=4,wm_window=8,wm_batches=2,log_every=1)
  assert cfg.warmup_steps==64 and cfg.wm_only_iters==500
  runner=SuperMimicRunner(TensorEnv(mdp),cfg,tmp_path/'no_warmup')
  # The fixture has constant increments; give the WM an actual error to learn.
  with torch.no_grad():runner.algorithm.wm.net[-1].bias[0]=1
  before={name:[p.detach().clone() for p in getattr(runner.algorithm,name).parameters()]
          for name in ('wm','policy','critic')}
  updates=_track_updates(runner,monkeypatch)
  collected=_track_collection(runner,monkeypatch)
  runner.learn()
  assert collected==[(2,2,2),(3,5,5),(2,7,7),(3,10,10)]
  assert updates==[('world_model_update',3,10)]*cfg.wm_batches+[('policy_update',3,10)]
  for name,parameters in before.items():
    assert any(not torch.equal(old,new) for old,new in zip(parameters,getattr(runner.algorithm,name).parameters())),name
  assert runner.iteration==4 and runner.env_steps==10 and runner.step_acc==0


def test_enabled_warmup_keeps_prefill_and_world_model_only_phase(mdp,tmp_path,monkeypatch):
  cfg=small_config(iterations=2,wm_only_iters=1)
  assert cfg.enable_warmup is True and cfg.warmup_steps==64
  runner=SuperMimicRunner(TensorEnv(mdp),cfg,tmp_path/'warmup')
  updates=_track_updates(runner,monkeypatch)
  collected=_track_collection(runner,monkeypatch)
  runner.learn()
  assert [count for count,_,_ in collected]==[64,2,3]
  assert updates==[('world_model_update',0,66),('world_model_update',1,69),('policy_update',1,69)]
  assert runner.env_steps==69


def test_disabled_warmup_resume_preserves_collection_remainder_without_prefill(mdp,tmp_path,monkeypatch):
  cfg=small_config(enable_warmup=False,iterations=1,wm_window=8)
  runner=SuperMimicRunner(TensorEnv(mdp),cfg,tmp_path/'run')
  runner.learn()
  assert runner.env_steps==2 and runner.step_acc==.5
  cfg.iterations=3
  resumed=SuperMimicRunner(TensorEnv(mdp),cfg,tmp_path/'resume')
  resumed.load(tmp_path/'run/last.pt')
  assert resumed.iteration==1 and resumed.buffer.size==0 and resumed.step_acc==.5
  updates=_track_updates(resumed,monkeypatch)
  collected=_track_collection(resumed,monkeypatch)
  resumed.learn()
  assert collected==[(3,5,3),(2,7,5)]
  assert updates==[]
  assert resumed.iteration==3 and resumed.step_acc==.5
  assert all(not optimizer.state for optimizer in
             (resumed.algorithm.opt_wm,resumed.algorithm.opt_pi,resumed.algorithm.opt_v))


@pytest.mark.parametrize('saved_enable,current_enable',[(True,False),(False,True),(None,True),(None,False)])
def test_resume_can_switch_warmup_and_load_legacy_checkpoints(mdp,tmp_path,saved_enable,current_enable):
  cfg=small_config(enable_warmup=True if saved_enable is None else saved_enable)
  runner=SuperMimicRunner(TensorEnv(mdp),cfg,tmp_path/'run')
  runner.collect(2)
  runner.iteration=1;runner.step_acc=.5
  checkpoint=tmp_path/'checkpoint.pt'
  runner.save(checkpoint)
  if saved_enable is None:
    state=torch.load(checkpoint,weights_only=False)
    del state['config']['enable_warmup']
    torch.save(state,checkpoint)
    _,legacy_cfg,_=load_policy(checkpoint,TensorEnv(mdp))
    assert legacy_cfg.enable_warmup is True
  resumed=SuperMimicRunner(TensorEnv(mdp),small_config(enable_warmup=current_enable),tmp_path/'resume')
  resumed.load(checkpoint)
  assert resumed.cfg.enable_warmup is current_enable
  assert resumed.iteration==1 and resumed.env_steps==2 and resumed.step_acc==.5
  assert resumed.buffer.size==0
  for name,value in runner.algorithm.policy.state_dict().items():
    torch.testing.assert_close(resumed.algorithm.policy.state_dict()[name],value)


def test_disabled_warmup_updates_world_model_without_legal_policy_starts(mdp,tmp_path,monkeypatch):
  cfg=small_config(enable_warmup=False,iterations=1,horizon=100)
  runner=SuperMimicRunner(TensorEnv(mdp),cfg,tmp_path/'no_starts')
  with torch.no_grad():runner.algorithm.wm.net[-1].bias[0]=1
  before={name:[p.detach().clone() for p in getattr(runner.algorithm,name).parameters()]
          for name in ('wm','policy','critic')}
  updates=_track_updates(runner,monkeypatch)
  recorded=[];original=runner._record
  def record(key,value):
    recorded.append((key,value));original(key,value)
  monkeypatch.setattr(runner,'_record',record)
  runner.learn()
  assert updates==[('world_model_update',0,2),('policy_update',0,2)]
  assert ('policy_skipped',1.0) in recorded
  assert not runner.buffer.eligible.any()
  assert any(not torch.equal(old,new) for old,new in zip(before['wm'],runner.algorithm.wm.parameters()))
  for name in ('policy','critic'):
    assert all(torch.equal(old,new) for old,new in zip(before[name],getattr(runner.algorithm,name).parameters()))
  assert runner.algorithm.opt_wm.state
  assert not runner.algorithm.opt_pi.state and not runner.algorithm.opt_v.state


def test_evaluation_scores_each_clip_once(mdp):
  cfg=small_config()
  class ZeroPolicy:
    def __call__(self,h,g):return torch.zeros(h.shape[0],29)
  result=evaluate(TensorEnv(mdp),ZeroPolicy(),cfg)
  assert result['n_failed']==0
  assert [v['steps'] for v in result['per_clip']]==[50,6]
  # Zero joint angles have a small soft-limit penalty in this actual G1 XML.
  violation=(torch.relu(-mdp.soft_hi)+torch.relu(mdp.soft_lo)).sum().item()
  expected=(5-2*violation)/5
  assert result['norm_return']==pytest.approx(expected,abs=1e-5)


def test_failed_clip_uses_full_length_denominator(mdp):
  class EarlyFailure(TensorEnv):
    def step(self,a):
      result=super().step(a)
      self.last_transition['terminated'][0]=True
      self.pending[0]=True
      return result
  result=evaluate(EarlyFailure(mdp),lambda h,g:torch.zeros(2,29),small_config())
  assert result['n_failed']==1 and result['per_clip'][0]['steps']==1
  clip=result['per_clip'][0]
  assert clip['normalized_return']==pytest.approx(clip['return']/(50*.1))


def test_history_partial_reset(mdp):
  cfg=small_config();env=TensorEnv(mdp)
  inputs=[]
  def policy(h,g):
    inputs.append((h.clone(),g.clone()))
    return torch.zeros(2,29)
  controller=PolicyController(env,policy,cfg)
  controller();env.step(torch.ones(2,29));controller()
  other=controller.history[1].clone()
  env.reset(env_ids=torch.tensor([0]));controller()
  torch.testing.assert_close(controller.history[1],other)
  torch.testing.assert_close(controller.history[0,0],controller.history[0,-1])
  assert inputs[-1][0].shape==(2,cfg.hist,93)
  assert inputs[-1][1].shape==(2,64*cfg.goal_frames)
  torch.testing.assert_close(inputs[-1][1][1],inputs[-2][1][1])


def test_controller_encodes_measured_robot_rotation_and_first_reset_target(mdp):
  cfg=small_config(hist=10,goal_frames=10)
  m=mdp.motion
  m.joint_pos[:]=torch.arange(m.num_frames)[:,None]*.01
  env=TensorEnv(mdp)
  rotations=so3_exp(torch.tensor([[.3,-.2,.4],[-.1,.5,-.3]]))
  env.robot.data.body_link_pose_w[:,0,3:]=mat_to_quat(rotations)
  inputs=[]
  def policy(h,g):
    inputs.append((h.clone(),g.clone()))
    return torch.zeros(2,29)
  controller=PolicyController(env,policy,cfg)
  controller()
  history,goal=inputs[0]
  goal=goal.reshape(2,10,64)
  assert history.shape==(2,10,93)
  # reset(k=0) leaves the robot at 0, but the first goal starts at 1.
  torch.testing.assert_close(history[:,:,0],torch.zeros(2,10))
  torch.testing.assert_close(goal[:,:,:29],m.joint_pos[torch.arange(1,11)][None].expand(2,-1,-1))
  # Reference rotations are identity; each of the ten targets uses measured R^T.
  torch.testing.assert_close(goal[:,:,58:],mat_to_6d(rotations.transpose(-1,-2))[:,None].expand(-1,10,-1))


def test_evaluation_loader_restores_new_policy_dimensions(mdp,tmp_path):
  cfg=small_config(hist=10,goal_frames=10)
  env=TensorEnv(mdp)
  runner=SuperMimicRunner(env,cfg,tmp_path/'run')
  # Persist nontrivial running statistics along with the actor weights.
  runner.collect(3)
  checkpoint=tmp_path/'run/last.pt'
  runner.save(checkpoint)
  policy,loaded_cfg,_=load_policy(checkpoint,env)
  assert policy.obs_norm.mean.shape==(93,)
  assert policy.goal_norm.mean.shape==(640,)
  assert policy.enc[0].in_features==640 and policy.dec[0].in_features==962
  for name,tensor in runner.algorithm.policy.state_dict().items():
    torch.testing.assert_close(policy.state_dict()[name],tensor)
  expected=PolicyController(env,runner.algorithm.policy,cfg)()
  actual=PolicyController(env,policy,loaded_cfg)()
  torch.testing.assert_close(actual,expected)


def test_legacy_run_name_loads_for_evaluation_and_resume(mdp,tmp_path):
  cfg=small_config(run_name="original",log_dir=str(tmp_path/'run'))
  runner=SuperMimicRunner(TensorEnv(mdp),cfg,cfg.log_dir)
  checkpoint=tmp_path/'run/last.pt'
  runner.save(checkpoint)
  saved=torch.load(checkpoint,weights_only=False)
  saved['config'].pop('run_name')
  saved['config'].pop('log_dir')
  saved['config']['wandb_name']='legacy-experiment'
  torch.save(saved,checkpoint)
  _,loaded_cfg,_=load_policy(checkpoint,TensorEnv(mdp))
  assert loaded_cfg.run_name=='legacy-experiment'
  resumed_cfg=small_config(run_name='resumed',log_dir=str(tmp_path/'resumed'))
  resumed=SuperMimicRunner(TensorEnv(mdp),resumed_cfg,resumed_cfg.log_dir)
  resumed.load(checkpoint)
  assert resumed.cfg.run_name=='resumed' and resumed.log_dir==tmp_path/'resumed'
  for name,value in runner.algorithm.policy.state_dict().items():
    torch.testing.assert_close(resumed.algorithm.policy.state_dict()[name],value)


def test_collection_adds_goal_noise_before_normalization(mdp,tmp_path):
  cfg=small_config(hist=10,goal_frames=10)
  env=TensorEnv(mdp)
  runner=SuperMimicRunner(env,cfg,tmp_path/'noisy_goals')
  clean=mdp.goal(read_state(env),env.command.time_steps,cfg.goal_frames)
  before_norm=[];encoder_inputs=[]
  policy=runner.algorithm.policy
  norm_hook=policy.goal_norm.register_forward_pre_hook(
    lambda module,args:before_norm.append(args[0].clone()))
  enc_hook=policy.enc.register_forward_pre_hook(
    lambda module,args:encoder_inputs.append(args[0].clone()))
  try:
    runner.collect(1)
  finally:
    norm_hook.remove();enc_hook.remove()
  assert len(before_norm)==len(encoder_inputs)==1
  difference=(before_norm[0]-clean).reshape(2,10,64)
  torch.testing.assert_close(difference[...,:58],torch.zeros_like(difference[...,:58]),rtol=0,atol=0)
  assert difference[...,58:].abs().max()<=.05+1e-6
  assert (difference[...,58:].abs().sum(-1)>0).all()
  torch.testing.assert_close(policy.goal_norm.mean,before_norm[0].mean(0))
  torch.testing.assert_close(encoder_inputs[0],policy.goal_norm(before_norm[0]))


def test_evaluation_and_playback_controller_keep_clean_goals(mdp,monkeypatch):
  cfg=small_config(hist=10,goal_frames=10)
  env=TensorEnv(mdp)
  inputs=[]
  def policy(history,goal):
    expected=mdp.goal(read_state(env),env.command.time_steps,cfg.goal_frames)
    torch.testing.assert_close(goal,expected,rtol=0,atol=0)
    inputs.append(goal.clone())
    return torch.zeros(env.num_envs,29)
  def unexpected_noise(*args):
    raise AssertionError('Evaluation/playback must not add observation or goal noise')
  monkeypatch.setattr(mdp,'add_goal_noise',unexpected_noise)
  monkeypatch.setattr(mdp,'add_noise',unexpected_noise)
  # Playback and recording construct this controller with its default flags.
  controller=PolicyController(env,policy,cfg)
  controller();controller()
  torch.testing.assert_close(inputs[0],inputs[1],rtol=0,atol=0)
  result=evaluate(env,policy,cfg)
  assert result['n_failed']==0 and len(inputs)>2


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires Linux NVIDIA GPU and installed mjlab 1.6.0')
def test_real_gpu_train_save_resume(mdp,tmp_path):
  import sys
  from pathlib import Path
  sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
  from src.tasks.supermimic.config import make_env
  from src.tasks.supermimic.rl.runner import SuperMimicRunner as RealRunner
  from mjlab.tasks.registry import list_tasks
  assert len([n for n in list_tasks() if n.startswith('Unitree-')])==20
  cfg=small_config(num_envs=4,iterations=2,warmup_steps=8,wm_only_iters=0,ckpt_every=1,log_every=1)
  env=make_env(mdp.motion.motion_files,num_envs=4)
  try:
    runner=RealRunner(env,cfg,tmp_path/'gpu')
    runner.learn()
    assert runner.iteration==2
  finally:env.close()
  cfg.iterations=3
  env=make_env(mdp.motion.motion_files,num_envs=4)
  try:
    runner=RealRunner(env,cfg,tmp_path/'gpu_resume')
    runner.load(tmp_path/'gpu/last.pt')
    runner.learn()
    assert runner.iteration==3
  finally:env.close()
