import copy
import pytest
import torch
from supermimic.rl.algorithm import SuperMimicAlgorithm
from supermimic.rl.buffer import ReplayBuffer
from supermimic.perception.history import select_history
from supermimic.perception.preprocessing import depth_float
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
    tr['depth']=torch.full((2,18,32),(t+1)*10,dtype=torch.uint8)
    tr['depth_next']=torch.full((2,18,32),(t+2)*10,dtype=torch.uint8)
    b.add(tr)
  return b,transitions


def test_eight_step_wm_predicts_history_latents_without_image_feedback(mdp,monkeypatch):
  torch.set_num_threads(2)
  cfg=visual_config(wm_window=8,depth_frames=8,depth_skip=5)
  alg=SuperMimicAlgorithm(mdp,cfg,'cpu')
  buffer,transitions=visual_buffer(mdp,cfg)
  for tr in transitions:alg.observe(tr,buffer.xi)
  window=buffer.sample_windows(2,8,strict=True)
  seen,predictions,decoded=[],[],[]
  step=alg.wm.step_visual
  def capture(state,action,xi,latent):
    seen.append(latent)
    result=step(state,action,xi,latent)
    result[1].retain_grad();predictions.append(result[1])
    return result
  monkeypatch.setattr(alg.wm,'step_visual',capture)
  hook=alg.wm.depth_decoder.register_forward_hook(lambda m,a,o:decoded.append(o))
  try:
    states,latents=alg._rollout(window['s'],window['a'],window['xi'],window['depth_history'])
    assert not decoded  # Decoder never participates in latent recurrence.
    images=alg.wm.decode_depth(latents)
    assert len(decoded)==1 and images.shape==(2,8,18,32)
    images[:,-1].square().mean().backward()
  finally:hook.remove()
  assert len(seen)==len(predictions)==8
  for t in range(1,8):assert seen[t] is predictions[t-1]
  assert predictions[0].grad.abs().sum()>0
  assert alg.wm.depth_encoder.frame_cnn[0].weight.grad.abs().sum()>0
  assert alg.wm.depth_encoder.gru.weight_hh_l0.grad.abs().sum()>0
  assert all(torch.isfinite(x).all() for x in states)
  before=copy.deepcopy(alg.policy.dec.state_dict())
  weights=alg.wm.depth_encoder.frame_cnn[0].weight.detach().clone()
  metrics=alg.world_model_update(buffer,strict=True)
  assert metrics['wm_depth_loss']>0 and metrics['wm_latent_loss']>0
  assert not torch.equal(weights,alg.wm.depth_encoder.frame_cnn[0].weight)
  for k,v in before.items():torch.testing.assert_close(v,alg.policy.dec.state_dict()[k])


def test_visual_actor_and_critic_update_only_heads_and_do_not_clip_shared_gradients(mdp):
  cfg=visual_config(depth_checkpoint=True,horizon=3)
  alg=SuperMimicAlgorithm(mdp,cfg,'cpu')
  b,transitions=visual_buffer(mdp,cfg)
  for tr in transitions:alg.observe(tr,b.xi)
  alg.world_model_update(b,strict=True)
  # Old WM gradients must not affect Actor's norm/clipping or be stepped by Critic.
  for p in alg.wm.depth_encoder.parameters():p.grad=torch.full_like(p,100.)
  gradients=[p.grad.clone() for p in alg.wm.depth_encoder.parameters()]
  before=copy.deepcopy(alg.wm.state_dict())
  actor=alg.policy.dec[0].weight.detach().clone()
  critic=alg.critic.net[0].weight.detach().clone()
  calls=[]
  hook=alg.wm.depth_decoder.register_forward_hook(lambda *args:calls.append(1))
  try:result=alg.policy_update(b)
  finally:hook.remove()
  assert not calls
  assert result['policy_skipped']==0 and result['pi_gradnorm']>0
  assert all(torch.isfinite(torch.tensor(x)) for x in result.values())
  assert not torch.equal(actor,alg.policy.dec[0].weight)
  assert not torch.equal(critic,alg.critic.net[0].weight)
  for k,v in before.items():torch.testing.assert_close(v,alg.wm.state_dict()[k],rtol=0,atol=0)
  for p,g in zip(alg.wm.depth_encoder.parameters(),gradients):
    torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
    assert p.requires_grad


def test_visual_short_motion_bootstrap_uses_true_endpoint_latent(mdp):
  cfg=visual_config(horizon=4,pi_batch=2)
  alg=SuperMimicAlgorithm(mdp,cfg,'cpu')
  b,_=visual_buffer(mdp,cfg)
  start,hist=b.sample_starts(2,cfg.hist)
  start['f']=torch.tensor([50,54])  # remaining lengths 1 and 4
  class Fixed:
    def sample_starts(self,*args):return start,hist
  seen=[]
  hook=alg.critic.register_forward_pre_hook(
    lambda m,args,kwargs:seen.append(kwargs['depth_latent'].detach().clone()),with_kwargs=True)
  try:alg.policy_update(Fixed())
  finally:hook.remove()
  # First call is bootstrap. Subsequent calls are V(s_0)...V(s_H).
  torch.testing.assert_close(seen[0][0],seen[2][0])
  torch.testing.assert_close(seen[0][1],seen[5][1])
  for t in range(2,6):torch.testing.assert_close(seen[t][0],seen[2][0])


def test_shared_parameters_have_exactly_one_optimizer_owner_and_lr(mdp):
  cfg=visual_config(wm_perception_lr=3e-4,depth_decoder_lr=5e-4)
  alg=SuperMimicAlgorithm(mdp,cfg,'cpu')
  assert alg.wm.depth_encoder is alg.policy.depth_encoder is alg.critic.depth_encoder
  shared={id(p) for p in alg.wm.depth_encoder.parameters()}
  owners=[]
  alg.set_lr(cfg.lr_decay_iters)
  for optimizer,rates in ((alg.opt_pi,[cfg.lr]),(alg.opt_wm,[cfg.lr,3e-4,5e-4]),(alg.opt_v,[cfg.lr])):
    assert len(optimizer.param_groups)==len(rates)
    for group,base in zip(optimizer.param_groups,rates):assert group['lr']==pytest.approx(base*.03)
    ids=[id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(ids)==len(set(ids));owners.append(set(ids))
  assert shared<=owners[1]
  assert not (owners[0]&owners[1] or owners[0]&owners[2] or owners[1]&owners[2])


def test_targets_are_real_strided_histories_at_next_time_and_stop_gradient(mdp):
  cfg=visual_config(wm_window=8,depth_frames=8,depth_skip=5)
  alg=SuperMimicAlgorithm(mdp,cfg,'cpu')
  history=torch.rand(2,36,18,32,requires_grad=True)
  future=torch.rand(2,8,18,32,requires_grad=True)
  actual=alg.depth_targets(history,future)
  assert not actual.requires_grad
  bank=torch.cat((history,future),1)
  with torch.no_grad():
    expected=torch.stack([alg.encode_depth(bank[:,t+1:36+t+1]) for t in range(8)],1)
  torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
  assert history.grad is None and future.grad is None


def test_visual_wm_supervises_terminal_image_but_masks_later_episode(mdp):
  cfg=visual_config(wm_window=8)
  alg=SuperMimicAlgorithm(mdp,cfg,'cpu')
  buffer,_=visual_buffer(mdp,cfg)
  window=buffer.sample_windows(2,8,strict=True)
  window['mask']=torch.tensor([[1.,1.,0.,0.,0.,0.,0.,0.]]).expand(2,-1)
  labels=window['depth_next'].float().div(255).requires_grad_()
  window['depth_next']=labels
  class Fixed:
    def sample_windows(self,*args,**kwargs):return window
  alg.world_model_update(Fixed())
  assert labels.grad[:,:2].abs().sum()>0
  assert labels.grad[:,1].abs().sum()>0  # includes true next image BEFORE reset
  assert labels.grad[:,2:].eq(0).all()


def test_loss_weights_preserve_masks_and_log_unweighted_components(mdp):
  cfg=visual_config(state_loss_weight=2.,latent_loss_weight=.3,depth_loss_weight=.7)
  alg=SuperMimicAlgorithm(mdp,cfg,'cpu')
  b,_=visual_buffer(mdp,cfg)
  metrics=alg.world_model_update(b,strict=True)
  assert metrics['wm_total_loss']==pytest.approx(
    2*metrics['wm_loss']+.3*metrics['wm_latent_loss']+.7*metrics['wm_depth_loss'],rel=1e-6)


@pytest.mark.parametrize('name',['state_loss_weight','latent_loss_weight','depth_loss_weight'])
def test_loss_weight_validation(name):
  visual_config(**{name:0.})
  for value in (-1.,float('nan'),float('inf')):
    with pytest.raises(ValueError,match=name):visual_config(**{name:value})


def test_actor_gradient_passes_through_predicted_h_with_frozen_perception(mdp,monkeypatch):
  cfg=visual_config(horizon=3)
  alg=SuperMimicAlgorithm(mdp,cfg,'cpu')
  b,_=visual_buffer(mdp,cfg)
  saved=[]
  def dynamics(s,a,xi,h):
    next_h=h+a[:,:cfg.depth_latent_dim]
    next_h.retain_grad();saved.append(next_h)
    return s,next_h
  monkeypatch.setattr(alg.wm,'step_visual',dynamics)
  monkeypatch.setattr(alg,'_reward_term',lambda s,f,a,prev:(torch.zeros(len(f)),torch.zeros(len(f))))
  # Only the visual endpoint contributes to Actor's return.
  alg._critic_forward_chunk=lambda x,h=None: h.sum(-1)+alg.critic.net[-1].bias.sum()*0
  result=alg.policy_update(b)
  assert result['pi_gradnorm']>0
  assert len(saved)==cfg.horizon and saved[0].grad.abs().sum()>0
  assert all(p.grad is None for p in alg.wm.depth_encoder.parameters())


def test_compiled_visual_rollout_and_decoder_match_eager_gradients(mdp):
  cfg=visual_config(depth_checkpoint=True)
  alg=SuperMimicAlgorithm(mdp,cfg,'cpu')
  b,_=visual_buffer(mdp,cfg)
  w=b.sample_windows(2,cfg.wm_window,strict=True)
  image=depth_float(w['depth_history']).requires_grad_()
  other=image.detach().clone().requires_grad_()
  compiled=torch.compile(alg._rollout,backend='aot_eager',fullgraph=True)
  decode=torch.compile(alg.wm.decode_depth,backend='aot_eager',fullgraph=True)
  targets=torch.compile(alg.depth_targets,backend='aot_eager',fullgraph=True)
  try:
    state,latent=alg._rollout(w['s'],w['a'],w['xi'],image)
    got_state,got_latent=compiled(w['s'],w['a'],w['xi'],other)
    depth=alg.wm.decode_depth(latent);got_depth=decode(got_latent)
    torch.testing.assert_close(got_depth,depth)
    for expected,actual in zip(state,got_state):torch.testing.assert_close(actual,expected)
    leaves=(image,*alg.wm.parameters());other_leaves=(other,*alg.wm.parameters())
    gradients=torch.autograd.grad(depth.square().mean()+latent.square().mean(),leaves,allow_unused=True)
    got=torch.autograd.grad(got_depth.square().mean()+got_latent.square().mean(),other_leaves,allow_unused=True)
    for a,b in zip(got,gradients):
      if a is None:assert b is None
      else:torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-6)
    assert got[0].abs().sum()>0
    torch.testing.assert_close(targets(image,w['depth_next']),alg.depth_targets(image,w['depth_next']))
  finally:torch.compiler.reset()
