import copy
import pytest
import torch
from supermimic.rl.networks import Policy, WorldModel, Critic
from supermimic.rl.depth_encoder import DepthEncoder, WindowGRU
from supermimic.config.g1.rl_cfg import SuperMimicConfig


def test_default_architecture_with_shared_encoder():
  cfg=SuperMimicConfig()
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
  cfg=SuperMimicConfig(depth_frame_dim=16,depth_latent_dim=16,depth_batch_size=4)
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
