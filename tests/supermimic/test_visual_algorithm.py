import copy
import pytest
import torch
from supermimic.rl.algorithm import SuperMimicAlgorithm
from supermimic.rl.buffer import ReplayBuffer
from supermimic.mdp.state import unflatten_state
from test_algorithm import small_config, filled_buffer


def visual_config(**kw):
  opts=dict(depth_enabled=True,depth_frames=2,depth_skip=1,depth_latent_dim=16,
            depth_frame_dim=16,depth_checkpoint=False,pi_batch=2,wm_batch=2)
  opts.update(kw)
  return small_config(**opts)


def visual_buffer(mdp,cfg):
  _,transitions=filled_buffer(mdp,cfg)
  m=mdp.motion
  b=ReplayBuffer(cfg.buffer_steps,2,torch.zeros(2,33),m.clip_end,m.clip_id,cfg.horizon,
                  cfg.depth_history_length)
  for t,tr in enumerate(transitions):
    tr['depth']=torch.full((2,18,32),t*10,dtype=torch.uint8)
    tr['depth_next']=torch.full((2,18,32),(t+1)*10,dtype=torch.uint8)
    b.add(tr)
  return b,transitions


def test_eight_step_wm_reencodes_predictions_and_backpropagates(mdp,monkeypatch):
  torch.set_num_threads(2)
  cfg=visual_config(wm_window=8,depth_frames=8,depth_skip=5)
  alg=SuperMimicAlgorithm(mdp,cfg,'cpu')
  buffer,transitions=visual_buffer(mdp,cfg)
  for tr in transitions:alg.observe(tr,buffer.xi)
  window=buffer.sample_windows(2,8,strict=True)
  seen,images=[],[]
  step=alg.wm.step_visual
  def capture(*args,**kwargs):
    seen.append(kwargs['depth_features'].detach().clone())
    return step(*args,**kwargs)
  monkeypatch.setattr(alg.wm,'step_visual',capture)
  def save_image(m,a,o):
    o.retain_grad();images.append(o)
  h2=alg.wm.depth_decoder.register_forward_hook(save_image)
  try:
    states,predicted=alg._rollout(window['s'],window['a'],window['xi'],window['depth_history'])
    # Last-step image loss must reach the first generated image through re-encoding.
    predicted[:,-1].square().mean().backward()
  finally:h2.remove()
  assert len(seen)==len(images)==8
  with torch.no_grad():
    features=[alg.wm.depth_encoder.encode_frames(image.detach()) for image in images]
  for t in range(1,8): torch.testing.assert_close(seen[t][:,-1],features[t-1])
  torch.testing.assert_close(seen[6][:,-2],features[0])  # five-step spacing
  assert images[0].grad is not None and images[0].grad.abs().sum()>0
  assert all(torch.isfinite(x).all() for x in states)
  before=copy.deepcopy(alg.policy.state_dict())
  weights=alg.wm.depth_encoder.frame_cnn[0].weight.detach().clone()
  metrics=alg.world_model_update(buffer,strict=True)
  assert metrics['wm_depth_loss']>0 and metrics['wm_total_loss']>=metrics['wm_loss']
  assert not torch.equal(weights,alg.wm.depth_encoder.frame_cnn[0].weight)
  for k,v in before.items():torch.testing.assert_close(v,alg.policy.state_dict()[k])


def test_visual_actor_and_critic_update_without_mutating_wm(mdp):
  cfg=visual_config(depth_checkpoint=True,horizon=3)
  alg=SuperMimicAlgorithm(mdp,cfg,'cpu')
  b,transitions=visual_buffer(mdp,cfg)
  for tr in transitions:alg.observe(tr,b.xi)
  alg.world_model_update(b,strict=True)
  before=copy.deepcopy(alg.wm.state_dict())
  actor=alg.policy.depth_encoder.frame_cnn[0].weight.detach().clone()
  critic=alg.critic.depth_encoder.frame_cnn[0].weight.detach().clone()
  result=alg.policy_update(b)
  assert result['policy_skipped']==0
  assert all(torch.isfinite(torch.tensor(x)) for x in result.values())
  assert not torch.equal(actor,alg.policy.depth_encoder.frame_cnn[0].weight)
  assert not torch.equal(critic,alg.critic.depth_encoder.frame_cnn[0].weight)
  for k,v in before.items():torch.testing.assert_close(v,alg.wm.state_dict()[k])


def test_visual_short_motion_bootstrap_uses_unpadded_depth_history(mdp):
  cfg=visual_config(horizon=4,pi_batch=2)
  alg=SuperMimicAlgorithm(mdp,cfg,'cpu')
  b,_=visual_buffer(mdp,cfg)
  start,hist=b.sample_starts(2,cfg.hist)
  start['f']=torch.tensor([50,54])  # remaining lengths 1 and 4
  class Fixed:
    def sample_starts(self,*args):return start,hist
  seen=[]
  def capture(m,args,kwargs):
    features=kwargs.get('depth_features')
    if features is None:
      with torch.no_grad():features=m.depth_encoder.encode_frames(args[1])
    seen.append((args[0].detach().clone(),features.detach().clone()))
  hook=alg.critic.register_forward_pre_hook(capture,with_kwargs=True)
  try:alg.policy_update(Fixed())
  finally:hook.remove()
  # Bootstrap vs lambda-return values, which are evaluated one time slice at a time.
  torch.testing.assert_close(seen[0][1][0],seen[2][1][0])
  torch.testing.assert_close(seen[0][1][1],seen[5][1][1])
  for t in range(2,6):torch.testing.assert_close(seen[t][1][0],seen[2][1][0])


def test_visual_learning_rates_decay_independently(mdp):
  cfg=visual_config(actor_perception_lr=1e-4,wm_perception_lr=3e-4,
                    critic_perception_lr=4e-4,depth_decoder_lr=5e-4)
  alg=SuperMimicAlgorithm(mdp,cfg,'cpu')
  alg.set_lr(cfg.lr_decay_iters)
  for optimizer,rates in ((alg.opt_pi,[cfg.lr,1e-4]),(alg.opt_wm,[cfg.lr,3e-4,5e-4]),
                           (alg.opt_v,[cfg.lr,4e-4])):
    for group,base in zip(optimizer.param_groups,rates):assert group['lr']==pytest.approx(base*.03)
    ids=[id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(ids)==len(set(ids))


def test_visual_wm_supervises_terminal_image_but_masks_later_episode(mdp):
  cfg=visual_config(wm_window=8)
  alg=SuperMimicAlgorithm(mdp,cfg,'cpu')
  buffer,_=visual_buffer(mdp,cfg)
  window=buffer.sample_windows(2,8,strict=True)
  window['mask']=torch.tensor([[1.,1.,0.,0.,0.,0.,0.,0.]]).expand(2,-1)
  # Floating labels expose which true next images contribute to supervision.
  labels=window['depth_next'].float().div(255).requires_grad_()
  window['depth_next']=labels
  class Fixed:
    def sample_windows(self,*args,**kwargs):return window
  alg.world_model_update(Fixed())
  assert labels.grad[:,:2].abs().sum()>0
  assert labels.grad[:,1].abs().sum()>0  # terminal transition is included
  assert labels.grad[:,2:].eq(0).all()


def test_compiled_visual_rollout_matches_eager_gradients(mdp):
  """AOT autograd exercises the same checkpoint/re-encoding path used on CUDA."""
  cfg=visual_config(depth_checkpoint=True)
  alg=SuperMimicAlgorithm(mdp,cfg,'cpu')
  b,_=visual_buffer(mdp,cfg)
  w=b.sample_windows(2,cfg.wm_window,strict=True)
  image=w['depth_history'].float().div(255).requires_grad_()
  other=image.detach().clone().requires_grad_()
  compiled=torch.compile(alg._rollout,backend='aot_eager',fullgraph=True)
  try:
    state,depth=alg._rollout(w['s'],w['a'],w['xi'],image)
    got_state,got_depth=compiled(w['s'],w['a'],w['xi'],other)
    torch.testing.assert_close(got_depth,depth)
    for expected,actual in zip(state,got_state):torch.testing.assert_close(actual,expected)
    grad=torch.autograd.grad(depth.square().mean(),image)[0]
    got_grad=torch.autograd.grad(got_depth.square().mean(),other)[0]
    torch.testing.assert_close(got_grad,grad,rtol=1e-4,atol=1e-6)
    assert got_grad.abs().sum()>0
  finally:torch.compiler.reset()
