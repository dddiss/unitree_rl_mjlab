import pytest
import torch
from supermimic.mdp.functional import FunctionalMDP
from supermimic.mdp.kinematics import G1Kinematics
from supermimic.mdp.motion_lib import MotionLibrary
from supermimic.mdp.state import flatten_state,unflatten_state
from supermimic.mdp.rotations import mat_to_6d,so3_exp
from supermimic.rl.algorithm import SuperMimicAlgorithm
from supermimic.rl.buffer import ReplayBuffer
from supermimic.config.g1.rl_cfg import SuperMimicConfig


def small_config(**kwargs):
  values=dict(num_envs=2,buffer_steps=16,wm_window=2,wm_batch=4,wm_batches=1,pi_batch=4,
              horizon=3,hist=2,goal_frames=2,critic_epochs=1,critic_minibatches=2,
              wm_hidden=(24,24),policy_hidden=(24,24),critic_hidden=(24,),amp='none',compile=False,logger='csv')
  values.update(kwargs)
  return SuperMimicConfig(**values)


def filled_buffer(mdp,cfg):
  m=mdp.motion
  xi=torch.zeros(2,33);xi[:,0]=.7
  b=ReplayBuffer(cfg.buffer_steps,2,xi,m.clip_end,m.clip_id,cfg.horizon)
  transitions=[]
  for step in range(8):
    f=torch.full((2,),step+1,dtype=torch.long)
    make=lambda frame:flatten_state((*mdp.reference_root(frame),m.joint_pos[frame],m.joint_vel[frame]))
    tr=dict(s=make(f-1),s1=make(f),a=torch.zeros(2,29),a_prev=torch.zeros(2,29),f=f,
            r=torch.ones(2)*.1,ep_step=torch.full((2,),step),terminated=torch.zeros(2,dtype=torch.bool),
            truncated=torch.zeros(2,dtype=torch.bool))
    b.add(tr);transitions.append(tr)
  return b,transitions


def test_updates_and_freezing(mdp):
  torch.set_num_threads(2)
  cfg=small_config()
  alg=SuperMimicAlgorithm(mdp,cfg,'cpu')
  buffer,transitions=filled_buffer(mdp,cfg)
  for tr in transitions:alg.observe(tr,buffer.xi)
  wm_before=[p.detach().clone() for p in alg.wm.parameters()]
  metrics=alg.world_model_update(buffer)
  assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
  assert any(not torch.equal(a,b) for a,b in zip(wm_before,alg.wm.parameters()))
  wm_before=[p.detach().clone() for p in alg.wm.parameters()]
  pi_before=[p.detach().clone() for p in alg.policy.parameters()]
  v_before=[p.detach().clone() for p in alg.critic.parameters()]
  metrics=alg.policy_update(buffer)
  assert metrics['policy_skipped']==0 and metrics['pi_gradnorm']>0
  assert all(torch.equal(a,b) for a,b in zip(wm_before,alg.wm.parameters()))
  assert any(not torch.equal(a,b) for a,b in zip(pi_before,alg.policy.parameters()))
  assert any(not torch.equal(a,b) for a,b in zip(v_before,alg.critic.parameters()))
  assert all(p.requires_grad for p in alg.wm.parameters())


def test_missing_imagination_start(mdp):
  cfg=small_config(horizon=100)
  b,_=filled_buffer(mdp,cfg)
  alg=SuperMimicAlgorithm(mdp,cfg,'cpu')
  assert alg.policy_update(b)=={'policy_skipped':1.0}


def test_world_model_skips_unavailable_strict_window_without_update(mdp):
  alg=SuperMimicAlgorithm(mdp,small_config(),'cpu')
  before={name:value.clone() for name,value in alg.wm.state_dict().items()}
  rng=torch.get_rng_state().clone()
  class NoStrictWindow:
    def sample_windows(self,batch,length,*,strict=False):
      assert strict and batch==alg.cfg.wm_batch and length==alg.cfg.wm_window
      return None
  assert alg.world_model_update(NoStrictWindow(),strict=True) is None
  assert torch.equal(torch.get_rng_state(),rng)
  assert not alg.opt_wm.state
  assert all(torch.equal(value,before[name]) for name,value in alg.wm.state_dict().items())


def test_terminal_bootstrap_backpropagates_through_previous_action(mdp,monkeypatch):
  """Isolate the new action input: state/goal paths cannot explain this gradient."""
  cfg=small_config()
  alg=SuperMimicAlgorithm(mdp,cfg,'cpu')
  buffer,_=filled_buffer(mdp,cfg)
  with torch.no_grad():
    # A critic depending ONLY on the first previous-action component.
    for parameter in alg.critic.net.parameters():
      parameter.zero_()
    alg.critic.net[0].weight[0,145]=1
    alg.critic.net[-1].weight[0,0]=1
  monkeypatch.setattr(alg,'_reward_term',lambda state,f,a,a_prev:(
    torch.zeros(len(f)),torch.zeros(len(f))))
  bootstrap_previous=[]
  original=alg._critic_input
  def capture(state,f,xi_n,goal,a_prev):
    if torch.is_grad_enabled():
      a_prev.retain_grad()
      bootstrap_previous.append(a_prev)
    return original(state,f,xi_n,goal,a_prev)
  alg._critic_input=capture
  result=alg.policy_update(buffer)
  assert result['pi_gradnorm']>0
  assert len(bootstrap_previous)==1
  gradient=bootstrap_previous[0].grad
  assert torch.isfinite(gradient).all() and (gradient[:,0].abs()>0).all()
  torch.testing.assert_close(gradient[:,1:],torch.zeros_like(gradient[:,1:]))


def test_policy_uses_proprioception_and_relative_goal_dimensions(mdp):
  cfg=small_config(hist=10,goal_frames=10)
  alg=SuperMimicAlgorithm(mdp,cfg,'cpu')
  assert alg.policy.obs_norm.mean.shape==(93,)
  assert alg.policy.dec[0].in_features==962  # 93 * 10 history + 32 latent.
  assert alg.policy.enc[0].in_features==640
  assert alg.wm.in_dim==133
  assert alg.critic.net[0].in_features==847
  f=torch.tensor([1,4])
  state=(*mdp.reference_root(f),mdp.motion.joint_pos[f],mdp.motion.joint_vel[f])
  obs=mdp.obs(state,torch.zeros(2,29),torch.zeros(2,33),f).requires_grad_()
  goal=mdp.goal(state,f,cfg.goal_frames).requires_grad_()
  action=alg.policy(obs[:,None].expand(-1,cfg.hist,-1),goal)
  assert action.shape==(2,29) and torch.isfinite(action).all()
  action.sum().backward()
  assert torch.isfinite(obs.grad).all()
  assert (obs.grad.abs().sum(-1)>0).all()
  assert torch.isfinite(goal.grad).all()
  assert (goal.grad.reshape(2,10,64)[:,:,58:].abs().sum((1,2))>0).all()


def test_imagination_reencodes_goal_from_predicted_state_with_gradients(mdp):
  cfg=small_config(goal_frames=10)
  alg=SuperMimicAlgorithm(mdp,cfg,'cpu')
  f=torch.tensor([4,mdp.motion.clip_end[-1]])
  state=(*mdp.reference_root(f-1),mdp.motion.joint_pos[f-1],mdp.motion.joint_vel[f-1])
  xi=torch.zeros(2,33)
  obs=mdp.obs(state,torch.zeros(2,29),xi,f)
  history=obs[:,None].expand(-1,cfg.hist,-1)
  def controlled_dynamics(s,a,xi_n):
    # Distinct next rotations plus a differentiable action-to-rotation path.
    dr=so3_exp(a[:,:3]+a.new_tensor([.2,-.15,.3]))
    return (s[0],s[1]@dr,*s[2:])
  alg.wm.step=controlled_dynamics
  next_state,action,_,next_f,goal=alg._imagine_step(state,history,f,xi,mdp.goal(state,f))
  assert next_f.tolist()==[5,int(mdp.motion.clip_end[-1])]
  torch.testing.assert_close(goal,mdp.goal(next_state,next_f))
  assert not torch.allclose(goal,mdp.goal(state,next_f))
  action.retain_grad()
  # Use only goal orientation, so this tests the goal path rather than rewards.
  orientation=goal.reshape(2,10,64)[:,:,58:]
  (orientation*torch.arange(1.,7.)).sum().backward()
  assert torch.isfinite(action.grad).all()
  assert (action.grad[:,:3].abs().sum(-1)>0).all()


def test_imagination_uses_noisy_policy_goals_and_clean_critic_goals(mdp):
  cfg=small_config(pi_batch=3,goal_frames=3)
  alg=SuperMimicAlgorithm(mdp,cfg,'cpu')
  m=mdp.motion
  # Ensure each imagined state has a different, non-identity robot rotation.
  alg.wm.out_norm.mean[3:6]=torch.tensor([.1,-.06,.08])
  # The first two rows reach different clips' last frames on step 3; row 3
  # stays in the middle of its clip and must keep the ordinary goal convention.
  f=torch.tensor([48,55,7])
  xi=torch.zeros(3,33);xi[:,0]=.75
  make=lambda frame:flatten_state((*mdp.reference_root(frame),m.joint_pos[frame],m.joint_vel[frame]))
  hist_f=f[:,None]+torch.tensor([-1,0])
  start=dict(s=make(f-1),a_prev=torch.linspace(-.2,.3,3*29).reshape(3,29),f=f,xi=xi)
  history=dict(s=make(hist_f-1),a_prev=torch.zeros(3,2,29),f=hist_f,xi=xi[:,None].expand(-1,2,-1))
  class FixedStarts:
    def sample_starts(self,batch,hist):
      assert batch==3 and hist==2
      return start,history
  inputs=[];imagined=[];policy_goals=[];imagined_actions=[]
  original_imagine=alg._imag_step
  def capture_imagine(*args):
    result=original_imagine(*args)
    imagined.append((tuple(t.detach().clone() for t in result[0]),result[3].clone()))
    imagined_actions.append(result[1].detach().clone())
    return result
  alg._imag_step=capture_imagine
  hook=alg.critic.register_forward_pre_hook(lambda module,args:inputs.append(args[0].detach().clone()))
  policy_hook=alg.policy.goal_norm.register_forward_pre_hook(
    lambda module,args:policy_goals.append(args[0].detach().clone()))
  try:
    metrics=alg.policy_update(FixedStarts())
  finally:
    hook.remove()
    policy_hook.remove()
  assert metrics['policy_skipped']==0
  assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
  # First critic call is the actor bootstrap; second is the H+1 states used
  # for critic lambda-returns. Both must receive the same final encoded goal.
  assert inputs[1].shape[:2]==(cfg.horizon+1,3)
  torch.testing.assert_close(inputs[0],inputs[1][-1])
  # The first critic state receives replay a_prev; every subsequent state
  # receives exactly the action producing it, not a history or the next action.
  expected_previous=torch.stack([start['a_prev'],*imagined_actions])
  torch.testing.assert_close(inputs[1][...,145:174],expected_previous)
  goal_dim=64*cfg.goal_frames
  final_goals=inputs[0][:,-goal_dim:].reshape(3,cfg.goal_frames,64)
  ends=m.clip_end
  torch.testing.assert_close(final_goals[:2,:,:29],m.joint_pos[ends,None].expand(-1,cfg.goal_frames,-1))
  torch.testing.assert_close(final_goals[:2],final_goals[:2,:1].expand(-1,cfg.goal_frames,-1))
  assert not torch.allclose(final_goals[:2,:,58:],mat_to_6d(torch.eye(3)).expand(2,cfg.goal_frames,6))
  # Every lambda-return state uses its OWN robot rotation, including the final
  # bootstrap state. Reusing the last pre-step goal would fail this comparison.
  all_states=[(unflatten_state(start['s']),f),*imagined]
  assert len(policy_goals)==cfg.horizon
  for step,(state,target) in enumerate(all_states):
    targets=torch.minimum(target[:,None]+torch.arange(cfg.goal_frames),m.clip_end[m.clip_id[target]][:,None])
    _,reference_r,_,_=mdp.reference_root(targets)
    expected=torch.cat((m.joint_pos[targets],m.joint_vel[targets],
                        mat_to_6d(state[1].transpose(-1,-2)[:,None]@reference_r)),-1)
    torch.testing.assert_close(inputs[1][step,:,-goal_dim:],expected.flatten(1))
    if step<cfg.horizon:
      # Noise is present BEFORE the policy's RunningNorm, on orientation only.
      difference=policy_goals[step].reshape(3,cfg.goal_frames,64)-expected
      torch.testing.assert_close(difference[...,:58],torch.zeros_like(difference[...,:58]),rtol=0,atol=0)
      assert difference[...,58:].abs().max()<=.05+1e-6
      assert (difference[...,58:].abs().sum(-1)>0).all()
  # Critic normalization and minibatches also receive the clean trajectory.
  torch.testing.assert_close(alg.critic.norm.mean,inputs[1].flatten(0,1).mean(0))
  clean_rows=inputs[1][:-1].flatten(0,1)
  for batch in inputs[2:]:
    assert (batch[:,None]==clean_rows[None]).all(-1).any(-1).all()
  assert imagined[-1][1].tolist()==[50,57,10]
  assert not torch.allclose(inputs[1][-2,:2,-goal_dim:],inputs[1][-1,:2,-goal_dim:])


@pytest.mark.parametrize('helper', ['reward_term', 'critic_input'])
@pytest.mark.parametrize('device', [
  'cpu',
  pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='Requires NVIDIA GPU for Inductor compilation')),
])
def test_compiled_imagination_heads_match_values_and_gradients(mdp, robot, helper, device):
  """CPU checks full-graph capture/AOT autograd; CUDA also checks generated kernels."""
  torch.set_num_threads(2)
  if device != 'cpu':
    model, joints, bodies, _ = robot
    root = model.body('pelvis').id
    motion = MotionLibrary(mdp.motion.motion_files, device=device)
    kin = G1Kinematics(model, joints, bodies, root, device=device)
    mdp = FunctionalMDP(motion, kin, mdp.default_q.to(device),
                        [b - 1 for b in bodies], root - 1)
  alg = SuperMimicAlgorithm(mdp, small_config(), device)
  # Time-major rows include both clip boundaries and a third continuing clip row.
  f = torch.tensor([[48, 55, 7], [49, 56, 8], [50, 57, 9]], device=device).flatten()
  p, r, v, w = mdp.reference_root(f)
  batch = len(f)
  offsets = torch.linspace(-.12, .16, batch * 3, device=device).reshape(batch, 3)
  joints = torch.linspace(-.2, .3, batch * 29, device=device).reshape(batch, 29)
  values = (p + offsets + p.new_tensor([0., 0., .25]), r @ so3_exp(offsets),
            v + offsets, w - offsets, joints, joints * .7)
  state = tuple(t.detach().requires_grad_() for t in values)
  if helper == 'reward_term':
    action = (joints * .3).detach().requires_grad_()
    previous = (joints * -.2 + .1).detach().requires_grad_()
    inputs = (state, f, action, previous)
    leaves = (*state, action, previous)
  else:
    xi_n = torch.linspace(-.5, .5, batch * 33, device=device).reshape(batch, 33).requires_grad_()
    goal = mdp.goal(state, f, alg.cfg.goal_frames).detach().requires_grad_()
    previous = (joints * -.2 + .1).detach().requires_grad_()
    inputs = (state, f, xi_n, goal, previous)
    leaves = (*state, xi_n, goal, previous)
  function = getattr(alg, helper)
  # No silent graph-break fallback: both production helpers must capture whole.
  compiled = torch.compile(function, fullgraph=True,
                           backend='aot_eager' if device == 'cpu' else 'inductor')
  try:
    expected, actual = function(*inputs), compiled(*inputs)
    if isinstance(expected, torch.Tensor):
      expected, actual = (expected,), (actual,)
    for ref, got in zip(expected, actual):
      torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)
    def loss(outputs):
      return sum((t * torch.linspace(.2, 1., t.numel(), device=device).reshape(t.shape)).mean()
                 for t in outputs)
    expected_grad = torch.autograd.grad(loss(expected), leaves)
    actual_grad = torch.autograd.grad(loss(actual), leaves)
    for ref, got in zip(expected_grad, actual_grad):
      assert torch.isfinite(got).all()
      torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)
    assert actual_grad[-1].abs().sum() > 0  # Previous action in both helpers.
    if helper == 'critic_input':
      assert actual_grad[-2].abs().sum() > 0  # Clean goal retains its own path.
    # The same critic helper is also used without gradients for lambda-return data.
    with torch.no_grad():
      no_grad_output = compiled(*inputs)
      if isinstance(no_grad_output, torch.Tensor):
        no_grad_output = (no_grad_output,)
      for ref, got in zip(expected, no_grad_output):
        assert not got.requires_grad
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)
  finally:
    torch.compiler.reset()
