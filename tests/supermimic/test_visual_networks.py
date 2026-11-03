import copy
from dataclasses import replace
import pytest
import torch
from supermimic.networks import Policy, WorldModel, Critic
from supermimic.networks.depth_encoder import DepthEncoder, WindowGRU
from supermimic.networks.world_model import WM_OUT
from supermimic.config.g1.rl_cfg import unitree_g1_runner_cfg, unitree_g1_terrain_runner_cfg


def test_base_architecture_has_no_perception_modules():
  cfg=unitree_g1_runner_cfg()
  w=WorldModel((16,),depth_cfg=cfg)
  p=Policy(93,640,hidden=(16,),depth_cfg=cfg)
  v=Critic(847,(16,),depth_cfg=cfg)
  assert p.dec[0].in_features==962
  assert w.net[0].in_features==133 and v.net[0].in_features==847
  assert not any(hasattr(m,'depth_encoder') for m in (w,p,v))
  assert not hasattr(w,'depth_decoder') and w.foot_head is None
  with torch.no_grad():
    assert p(torch.zeros(1,10,93),torch.zeros(1,640)).shape==(1,29)


def test_default_architecture_with_shared_encoder():
  cfg=unitree_g1_terrain_runner_cfg()
  w=WorldModel((16,),depth_cfg=cfg)
  p=Policy(93,640,hidden=(16,),depth_cfg=cfg,depth_encoder=w.depth_encoder)
  v=Critic(847,(16,),depth_cfg=cfg,depth_encoder=w.depth_encoder)
  assert p.dec[0].in_features==1218
  assert w.net[0].in_features==389 and v.net[0].in_features==1103
  encoders=[m.depth_encoder for m in (p,w,v)]
  for e in encoders:
    assert e.frame_cnn[0].stride==(1,1)
    assert e.frame_projection[1].out_features==256
    assert e.gru.input_size==e.gru.hidden_size==256 and e.gru.num_layers==1
  sets=[{id(x) for x in e.parameters()} for e in encoders]
  assert encoders[0] is encoders[1] is encoders[2]
  assert sets[0]==sets[1]==sets[2]
  with torch.no_grad():
    image=torch.randint(256,(1,8,18,32),dtype=torch.uint8)
    assert p(torch.zeros(1,10,93),torch.zeros(1,640),image).shape==(1,29)


@pytest.mark.parametrize('with_costs',[False,True])
def test_wm_prediction_bounds_only_h_and_preserves_state_deltas_and_costs(with_costs):
  cfg=replace(unitree_g1_terrain_runner_cfg(),depth_frame_dim=16,depth_latent_dim=16)
  wm=WorldModel((16,),depth_cfg=cfg,foot_reward_names=('feet_stumble','feet_at_plane'))
  raw_h=torch.linspace(-3.,3.,cfg.depth_latent_dim)
  raw_delta=torch.full((WM_OUT,),2.)
  raw_delta[3:6]=.1  # Keep the rotation increment inside SO(3)'s principal branch.
  with torch.no_grad():
    wm.net[-1].weight.zero_()
    wm.net[-1].bias.copy_(torch.cat((raw_delta,raw_h)))
    wm.out_norm.mean.fill_(.1);wm.out_norm.var.fill_(.25)
    wm.foot_head[0][-1].weight.zero_();wm.foot_head[0][-1].bias.fill_(2.)
  state=(torch.zeros(2,3),torch.eye(3).expand(2,-1,-1),torch.zeros(2,3),
         torch.zeros(2,3),torch.zeros(2,29),torch.zeros(2,29))
  args=(state,torch.zeros(2,29),torch.zeros(2,33),torch.zeros(2,cfg.depth_latent_dim))
  if with_costs:
    next_state,h,costs=wm.step_with_foot_costs(*args)
    torch.testing.assert_close(costs,torch.nn.functional.softplus(torch.full((2,2),2.)))
    assert costs.gt(1).all()  # The independent reward head is not bounded by tanh.
  else:
    next_state,h=wm.step_visual(*args)
  expected_delta=raw_delta*wm.out_norm.std()+wm.out_norm.mean
  torch.testing.assert_close(wm.targets(state,next_state),expected_delta.expand(2,-1))
  assert next_state[0].gt(1).all()  # Physical increments still use their original units.
  torch.testing.assert_close(h,raw_h.tanh().expand(2,-1))
  assert h.abs().le(1).all()
  h.sum().backward()
  # Overshooting raw predictions must retain gradients (unlike hard clipping).
  gradient=wm.net[-1].bias.grad
  torch.testing.assert_close(gradient[WM_OUT:],2*(1-raw_h.tanh().square()))
  assert gradient[WM_OUT:].gt(0).all() and gradient[:WM_OUT].eq(0).all()


def test_chunked_checkpointed_encoder_preserves_outputs_and_input_gradients():
  torch.set_num_threads(2)
  e=DepthEncoder(16,16,False,batch_size=10)
  other=copy.deepcopy(e);other.batch_size=2;other.checkpoint_activations=True
  x=torch.rand(5,3,18,32,requires_grad=True)
  y=x.detach().clone().requires_grad_()
  out=e(x);got=other(y)
  torch.testing.assert_close(got,out)
  out.square().sum().backward();got.square().sum().backward()
  torch.testing.assert_close(y.grad,x.grad,atol=1e-6,rtol=1e-4)
  for p,q in zip(e.parameters(),other.parameters()):
    torch.testing.assert_close(q.grad,p.grad,atol=1e-6,rtol=1e-4)
  other.requires_grad_(False);other.eval()
  z=y.detach().clone().requires_grad_()
  other(z).sum().backward()
  assert z.grad.abs().sum()>0  # Frozen critic still propagates to imagined images.


@pytest.mark.parametrize('frozen',[False,True])
def test_window_gru_matches_torch_gru_outputs_input_and_parameter_gradients(frozen):
  torch.manual_seed(17)
  expected=torch.nn.GRU(7,11,batch_first=True).double()
  torch.manual_seed(17)
  actual=WindowGRU(7,11).double()
  assert list(actual.state_dict())==list(expected.state_dict())
  for key,value in expected.state_dict().items():
    torch.testing.assert_close(actual.state_dict()[key],value,rtol=0,atol=0)
  # Existing nn.GRU checkpoints load strictly, without renaming any keys.
  actual.load_state_dict(expected.state_dict(),strict=True)
  if frozen:
    actual.requires_grad_(False);expected.requires_grad_(False)
    actual.eval();expected.eval()
  x=torch.randn(3,8,7,dtype=torch.float64,requires_grad=True)
  y=x.detach().clone().requires_grad_()
  out,last=expected(x);got,got_last=actual(y)
  torch.testing.assert_close(got,out,rtol=1e-10,atol=1e-12)
  torch.testing.assert_close(got_last,last,rtol=1e-10,atol=1e-12)
  (out.square().mean()+last.square().mean()).backward()
  (got.square().mean()+got_last.square().mean()).backward()
  torch.testing.assert_close(y.grad,x.grad,rtol=1e-9,atol=1e-12)
  for p,q in zip(expected.parameters(),actual.parameters()):
    if frozen:assert p.grad is None and q.grad is None
    else:torch.testing.assert_close(q.grad,p.grad,rtol=1e-9,atol=1e-12)


def test_visual_encoder_compiles_fullgraph_with_checkpoint_and_input_gradients():
  torch.set_num_threads(2)
  encoder=DepthEncoder(16,16,True,batch_size=2)
  compiled=torch.compile(encoder,backend='aot_eager',fullgraph=True)
  x=torch.rand(3,8,18,32,requires_grad=True)
  y=x.detach().clone().requires_grad_()
  try:
    expected=encoder(x);got=compiled(y)
    torch.testing.assert_close(got,expected)
    expected_grad=torch.autograd.grad(expected.square().sum(),(x,*encoder.parameters()))
    actual_grad=torch.autograd.grad(got.square().sum(),(y,*encoder.parameters()))
    for a,b in zip(actual_grad,expected_grad):
      torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-4)
  finally:torch.compiler.reset()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires NVIDIA GPU for Inductor')
def test_cuda_inductor_visual_critic_forward_and_gradients_match_eager():
  """Exercise the real CUDA compiler, not just CPU AOT tracing, when available."""
  torch.manual_seed(23)
  cfg=replace(unitree_g1_terrain_runner_cfg(),depth_frame_dim=16,depth_latent_dim=16,depth_batch_size=4)
  model=Critic(17,(16,),depth_cfg=cfg).cuda()
  compiled=torch.compile(model,fullgraph=True,dynamic=True)
  try:
    for frozen in (True,False):
      model.requires_grad_(not frozen)
      # Include a partial chunk; image gradients are required in frozen mode.
      for batch in (4,3):
        x=torch.randn(batch,17,device='cuda')
        a=torch.rand(batch,8,18,32,device='cuda',requires_grad=True)
        b=a.detach().clone().requires_grad_()
        expected=model(x,a);actual=compiled(x,b)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
        parameters=() if frozen else tuple(model.parameters())
        eg=torch.autograd.grad(expected.square().sum(),(a,*parameters))
        ag=torch.autograd.grad(actual.square().sum(),(b,*parameters))
        for p,q in zip(ag,eg):torch.testing.assert_close(p,q,rtol=1e-3,atol=1e-5)
  finally:torch.compiler.reset()
