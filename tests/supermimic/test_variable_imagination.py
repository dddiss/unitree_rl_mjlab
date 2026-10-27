"""Closed-form checks of clip-tail bootstraps and masked critic fitting."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from supermimic.config.g1.rl_cfg import SuperMimicConfig
from supermimic.mdp.state import flatten_state
from supermimic.rl.algorithm import SuperMimicAlgorithm
from supermimic.rl.networks import RunningNorm
from test_algorithm import small_config


class LinearValue(nn.Module):
  def __init__(self):
    super().__init__()
    self.slope = nn.Parameter(torch.tensor(.4))
    self.bias = nn.Parameter(torch.tensor(.7))
    self.norm = RunningNorm(1)
    self.fit_inputs = []

  def forward(self, x):
    if torch.is_grad_enabled() and self.slope.requires_grad:
      self.fit_inputs.append(x.detach().clone())
    # Deliberately analytic; norm is still updated by the production learner.
    return self.slope * x[..., 0] + self.bias


def analytic_learner(mdp, monkeypatch, frames, failure, **config):
  cfg = small_config(horizon=32, pi_batch=len(frames), critic_minibatches=1, **config)
  alg = SuperMimicAlgorithm(mdp, cfg, 'cpu')
  alg.policy = nn.Module()
  alg.policy.register_parameter('gain', nn.Parameter(torch.tensor(1.)))
  alg.critic = LinearValue()
  # Keep parameters fixed so losses, targets and gradients have exact oracles.
  alg.opt_pi = torch.optim.SGD(alg.policy.parameters(), lr=0.)
  alg.opt_v = torch.optim.SGD(alg.critic.parameters(), lr=0.)
  f = torch.tensor(frames)
  n = len(f)
  state = (torch.zeros(n,3), torch.eye(3).expand(n,-1,-1),
           torch.zeros(n,3), torch.zeros(n,3), torch.zeros(n,29), torch.zeros(n,29))
  start = dict(s=flatten_state(state), f=f, xi=torch.zeros(n,33), a_prev=torch.zeros(n,29))
  hist = {key:value[:,None].expand((-1,cfg.hist)+value.shape[1:])
          for key,value in start.items()}
  buffer = SimpleNamespace(sample_starts=lambda *args:(start,hist))
  candidates = []

  def step(s, history, frame, xi, goal):
    a = alg.policy.gain.expand(n,29)
    pos = s[0] + a[:,:3]
    pos.retain_grad()
    candidates.append(pos)
    return (pos,*s[1:]), a, history, alg.next_target(frame), goal

  def reward(s, frame, a, previous):
    return 2 + s[0][:,0], torch.tensor(failure).repeat(cfg.horizon)

  bootstrap = []
  def critic_input(s, frame, xi, goal, previous):
    if torch.is_grad_enabled():
      bootstrap.append((s[0].detach().clone(),frame.clone(),previous.detach().clone()))
    return s[0][:,:1]

  monkeypatch.setattr(alg,'_imag_step',step)
  monkeypatch.setattr(alg,'_reward_term',reward)
  monkeypatch.setattr(alg,'_critic_input',critic_input)
  return alg, buffer, candidates, bootstrap


@pytest.mark.parametrize('failure', [[0.,0.,0.], [0.,.25,0.], [1.,0.,0.]])
def test_short_rollouts_bootstrap_at_own_endpoint_and_mask_padding(mdp,monkeypatch,failure):
  # Different clips: 1 step, 2 steps, and a full 32-step rollout.
  alg,buffer,candidates,bootstrap = analytic_learner(mdp,monkeypatch,[50,56,10],failure)
  lengths = [1,2,32]
  c = alg.cfg
  result = alg.policy_update(buffer)
  gain = torch.tensor(1.,requires_grad=True)
  objectives,targets,positions = [],[],[]
  for length,d in zip(lengths,failure):
    reward = 2 + torch.arange(1,length+1) * gain
    value = .4 * torch.arange(length+1) * gain + .7
    discount_survival = (c.gamma*(1-d)) ** torch.arange(length)
    objectives.append((discount_survival*reward).sum() + (c.gamma*(1-d))**length*value[-1])
    ret = value[-1]
    row = []
    for t in reversed(range(length)):
      ret = reward[t] + c.gamma*(1-d)*((1-c.lam)*value[t+1]+c.lam*ret)
      row.append(ret)
    targets.append(torch.stack(row[::-1]).detach())
    positions.extend(range(length+1))  # Unique valid states, including bootstrap.
  expected_loss = -torch.stack(objectives).mean()/32
  expected_grad, = torch.autograd.grad(expected_loss,gain)
  torch.testing.assert_close(result['pi_loss'],float(expected_loss.detach()))
  torch.testing.assert_close(result['pi_gradnorm'],float(expected_grad.abs()))
  assert result['policy_skipped']==0
  assert len(bootstrap)==1
  torch.testing.assert_close(bootstrap[0][0][:,0],torch.tensor(lengths,dtype=torch.float32))
  assert bootstrap[0][1].tolist()==[50,57,42]
  torch.testing.assert_close(bootstrap[0][2],torch.ones(3,29))

  # No gradient through steps after a row's motion end. The endpoint bootstrap
  # still contributes through the final VALID transition.
  for t,candidate in enumerate(candidates):
    assert candidate.grad is not None and torch.isfinite(candidate.grad).all()
    for b,length in enumerate(lengths):
      if t>=length:
        assert torch.count_nonzero(candidate.grad[b])==0
      elif failure[b]==0:
        assert candidate.grad[b,0].abs()>0

  assert alg.critic.norm.count.item()==sum(lengths)+len(lengths)
  torch.testing.assert_close(alg.critic.norm.mean,torch.tensor(positions,dtype=torch.float32).mean().reshape(1))
  assert sum(len(x) for x in alg.critic.fit_inputs)==sum(lengths)
  expected_value_loss = torch.cat([
    (.4*torch.arange(length)+.7-target).square()
    for length,target in zip(lengths,targets)]).mean()
  torch.testing.assert_close(result['v_loss'],float(expected_value_loss))
  expected_survival = sum((1-d)**length for length,d in zip(lengths,failure))/len(lengths)
  assert result['imag_survival']==pytest.approx(expected_survival)


def test_actor_clip_settings_leave_critic_clip_unchanged(mdp,monkeypatch):
  alg,buffer,_,_ = analytic_learner(mdp,monkeypatch,[50],[0.],pi_clip=.02,pi_clip_value=.003)
  norm_clip, value_clip = nn.utils.clip_grad_norm_, nn.utils.clip_grad_value_
  calls = []
  def record_norm(parameters,bound):
    parameters=list(parameters)
    calls.append(('norm',bound))
    return norm_clip(parameters,bound)
  def record_value(parameters,bound):
    calls.append(('value',bound))
    return value_clip(parameters,bound)
  monkeypatch.setattr(nn.utils,'clip_grad_norm_',record_norm)
  monkeypatch.setattr(nn.utils,'clip_grad_value_',record_value)
  result=alg.policy_update(buffer)
  assert calls==[('norm',.02),('value',.003),('norm',1.0)]
  assert result['pi_gradnorm']>.02
  assert alg.policy.gain.grad.abs().item()==pytest.approx(.003)


def test_actor_clip_defaults():
  cfg=SuperMimicConfig()
  assert cfg.pi_clip==1/32 and cfg.pi_clip_value==.1/32


@pytest.mark.parametrize('field',['pi_clip','pi_clip_value'])
@pytest.mark.parametrize('value',[0.,-1.,float('nan'),float('inf')])
def test_actor_clip_config_rejects_invalid_bounds(field,value):
  with pytest.raises(ValueError,match=field):
    SuperMimicConfig(**{field:value})
