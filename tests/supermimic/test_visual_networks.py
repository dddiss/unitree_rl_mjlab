import copy
import torch
from supermimic.rl.networks import Policy, WorldModel, Critic
from supermimic.rl.depth_encoder import DepthEncoder
from supermimic.config.g1.rl_cfg import SuperMimicConfig


def test_default_architecture_and_three_independent_encoders():
  cfg=SuperMimicConfig()
  p=Policy(93,640,hidden=(16,),depth_cfg=cfg)
  w=WorldModel((16,),depth_cfg=cfg)
  v=Critic(847,(16,),depth_cfg=cfg)
  assert p.dec[0].in_features==1218
  assert w.net[0].in_features==389 and v.net[0].in_features==1103
  encoders=[m.depth_encoder for m in (p,w,v)]
  for e in encoders:
    assert e.frame_cnn[0].stride==(1,1)
    assert e.frame_projection[1].out_features==256
    assert e.gru.input_size==e.gru.hidden_size==256 and e.gru.num_layers==1
  sets=[{id(x) for x in e.parameters()} for e in encoders]
  assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
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
