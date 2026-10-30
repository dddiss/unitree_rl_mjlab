"""CPU tests run the real step/compute methods against observable fake physics.

The GPU test below additionally exercises the real mjlab managers and simulation.
"""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import torch

from supermimic.mdp.logging_metrics import TRACKING_METRICS, capture_step_metrics

ROOT=Path(__file__).resolve().parents[2]


def methods(path,class_name,names,namespace):
  tree=ast.parse(path.read_text())
  cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
  body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
  body += [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
  exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(path),'exec'),namespace)
  return namespace


def test_command_subset_dt_zero_and_last_frame():
  ns=methods(ROOT/'src/tasks/supermimic/mdp/commands.py','MotionCommand',
             {'compute','_resolve_env_ids'}, {'torch':torch})
  class Command:
    compute=ns['compute'];_resolve_env_ids=ns['_resolve_env_ids']
    def _update_command(self,ids): self.updated=ids.clone()
    def _update_metrics(self,ids): pass
  cmd=Command();cmd._all_env_ids=torch.arange(3);cmd.time_steps=torch.tensor([2,3,5])
  cmd.motion=NS(clip_end=torch.tensor([5]));cmd.motion_ids=torch.zeros(3,dtype=torch.long)
  cmd.compute(0.0,torch.tensor([1]))
  assert cmd.time_steps.tolist()==[2,3,5]
  cmd.compute(torch.tensor([.02,0.,.02]))
  assert cmd.time_steps.tolist()==[3,3,5]


@pytest.mark.parametrize('visual', [False, True])
@pytest.mark.parametrize('foot_support', [False, True])
def test_step_captures_state_and_target_before_command_advance(visual, foot_support):
  events=[]
  ns=methods(ROOT/'src/tasks/supermimic/env.py','SuperMimicEnv',{'step'},
             {'torch':torch,'flatten_state':lambda x:x,'read_state':lambda e:e.state,
              'capture_step_metrics':capture_step_metrics})
  class Env: step=ns['step']
  e=Env();e.state=torch.zeros(2,71);e.device='cpu';e._manual_reset_pending=torch.zeros(2,dtype=torch.bool)
  e._sim_step_counter=e.common_step_counter=0;e.physics_dt=.005;e.step_dt=.02
  e._sense_before_rewards=foot_support
  e.episode_length_buf=torch.zeros(2,dtype=torch.long);e.cfg=NS(decimation=4);e.extras={}
  cmd=NS(time_steps=torch.tensor([4,7]),update_relative_body_poses=lambda:events.append('align'))
  cmd.metrics={name:torch.zeros(2) for name in TRACKING_METRICS}
  def update_metrics():
    events.append('metrics')
    for value in cmd.metrics.values(): value.copy_((cmd.time_steps-torch.tensor([4,7])).abs())
  cmd._update_metrics=update_metrics
  def command_compute(dt):
    events.append('command');cmd.time_steps+=torch.as_tensor(dt).gt(0).long()
    update_metrics()
  e.command_manager=NS(get_term=lambda n:cmd,compute=command_compute)
  action=NS(action=torch.zeros(2,29),prev_action=torch.zeros(2,29),apply_action=lambda:None)
  def process(a): action.prev_action=action.action.clone();action.action=a.clone()
  action.process_action=process;e.action_manager=action
  def physics(): e.state+=1;events.append('physics')
  e.sim=NS(step=physics,forward=lambda:events.append('forward'),sense=lambda:events.append('sense'))
  if visual:
    Env.depth_image=property(lambda self:self._depth_image)
    e._depth_image=torch.zeros(2,18,32,dtype=torch.uint8)
    def read_depth():
      events.append('depth')
      return e.state[:,0,None,None].to(torch.uint8).expand(-1,18,32).clone()
    e._depth_camera=NS(read=read_depth)
  e.scene=NS(write_data_to_sim=lambda:None,update=lambda dt:None)
  e.metrics_manager=NS(compute_substep=lambda:None,compute=lambda:None)
  def term(): events.append('termination');return torch.tensor([True,False])
  e.termination_manager=NS(compute=term,terminated=torch.tensor([True,False]),time_outs=torch.zeros(2,dtype=torch.bool))
  e.termination_manager.active_terms=['anchor_pos']
  e.termination_manager.get_term=lambda name:e.termination_manager.terminated
  def reward(dt):
    events.append('reward');assert cmd.time_steps.tolist()==[4,7];return torch.tensor([.1,.1])
  e.reward_manager=NS(compute=reward,active_terms=['tracking', 'feet_at_plane'],
                      get_term_cfg=lambda name:NS(weight=5.0 if name=='tracking' else 0.0),
                      _step_reward=torch.tensor([[5., 0.], [5., 0.]]))
  if foot_support:
    e.foot_reward_names = ('feet_stumble', 'feet_at_plane')
    def measured(env, factor):
      events.append('foot_label')
      assert events[-2] != 'command'
      return env.state[:, 0] * factor + cmd.time_steps * .01
    e.reward_manager.get_term_cfg = lambda name: NS(
      weight=5. if name == 'tracking' else 0., func=measured,
      params={'factor': .1 if name == 'feet_stumble' else .2})
  e.shared_reward_terms=lambda:{'tracking':torch.ones(2)}
  e.event_manager=NS(available_modes=['startup'])
  e.observation_manager=NS(compute=lambda update_history:{})
  e.recorder_manager=NS(record_post_step=lambda:None)
  e.step(torch.ones(2,29))
  tr=e.last_transition
  assert events.index('forward')<events.index('termination')<events.index('reward')<events.index('command')
  assert tr['s'].eq(0).all() and tr['s1'].eq(4).all()
  assert tr['f'].tolist()==[4,7] and tr['a_prev'].eq(0).all()
  assert events.index('reward')<events.index('metrics')<events.index('command')
  assert e.last_step_metrics['tracking']['error_anchor_pos'].tolist()==[0.,0.]
  assert cmd.metrics['error_anchor_pos'].tolist()==[0.,1.]
  torch.testing.assert_close(e.last_step_metrics['reward']['tracking'],tr['r'])
  assert e.last_step_metrics['reward']['feet_at_plane'].eq(0).all()
  assert events.count('sense') == 1
  if foot_support:
    assert events.index('forward') < events.index('sense') < events.index('reward')
    assert events.index('reward') < events.index('foot_label') < events.index('command')
    torch.testing.assert_close(tr['foot_costs'], torch.tensor([[.44, .84], [.47, .87]]))
    # Raw labels are available although the actual foot reward is still zero.
    assert tr['r'].eq(.1).all()
  else:
    assert events.index('command') < events.index('sense')
  assert e._manual_reset_pending.tolist()==[True,False]
  if visual:
    assert events.index('sense')<events.index('depth')
    assert tr['depth'].eq(0).all() and tr['depth_next'].eq(4).all()
    # A terminal row's supervision must survive the subsequent reset image.
    e._depth_image[0].zero_()
    assert tr['depth_next'][0].eq(4).all()
  e.state.zero_()
  assert tr['s1'].eq(4).all()
  if foot_support:
    torch.testing.assert_close(tr['foot_costs'], torch.tensor([[.44, .84], [.47, .87]]))
  with pytest.raises(RuntimeError):e.step(torch.zeros(2,29))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires Linux NVIDIA GPU and installed mjlab 1.6.0')
def test_real_env_reset_and_reward_parity(mdp):
  import sys
  sys.path.insert(0,str(ROOT))
  from src.tasks.supermimic.config import make_env
  from src.tasks.supermimic.mdp.state import unflatten_state
  env=make_env(mdp.motion.motion_files,num_envs=4)
  try:
    cmd=env.command_manager.get_term('motion');env.reset()
    xi=env.context.clone()
    ids=torch.tensor([0,2],device=env.device)
    frames=torch.tensor([0,51],device=env.device)
    before=cmd.time_steps.clone()
    cmd.request_frames(frames,ids);env.reset(env_ids=ids)
    assert torch.equal(cmd.time_steps[ids],frames+1)
    assert torch.equal(cmd.time_steps[[1,3]],before[[1,3]])
    for _ in range(64):
      env.step(torch.zeros(4,29,device=env.device))
      tr=env.last_transition
      expected=env.functional_mdp.reward(unflatten_state(tr['s1']),tr['f'],tr['a'],tr['a_prev'])
      torch.testing.assert_close(tr['r'],expected,rtol=1e-4,atol=1e-5)
      assert torch.isfinite(tr['s1']).all()
      done=(tr['terminated']|tr['truncated']).nonzero().flatten()
      if len(done):env.reset(env_ids=done)
    torch.testing.assert_close(env.context,xi)
  finally:env.close()
