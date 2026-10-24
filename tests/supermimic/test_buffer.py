import torch
from supermimic.rl.buffer import ReplayBuffer


def transition(f, step, done=False):
  n=len(f)
  return dict(s=torch.full((n,71),float(step)),s1=torch.full((n,71),float(step+1)),
              a=torch.zeros(n,29),a_prev=torch.zeros(n,29),f=torch.tensor(f),
              r=torch.ones(n),ep_step=torch.full((n,),step),
              terminated=torch.full((n,),done),truncated=torch.zeros(n,dtype=torch.bool))


def test_windows_and_histories_do_not_cross_reset(mdp):
  m=mdp.motion
  b=ReplayBuffer(4,1,torch.zeros(1,33),m.clip_end,m.clip_id,2)
  b.add(transition([1],0));b.add(transition([2],1,True));b.add(transition([3],0));b.add(transition([4],1))
  w=b.sample_windows(100,4)
  assert w['mask'].tolist()==[[1,1,0,0]]*100
  start,h=b.sample_starts(500,5)
  for row,states in zip(start['s'],h['s']):
    assert states[:,0].tolist()==([0,0,0,0,0] if row[0]==0 else [0,0,0,0,1])


def test_ring_and_no_eligible_start(mdp):
  m=mdp.motion
  b=ReplayBuffer(3,1,torch.zeros(1,33),m.clip_end,m.clip_id,32)
  for step in range(6): b.add(transition([step+1],step))
  _,h=b.sample_starts(200,10)
  assert h['s'][...,0].min()>=3
  for step in range(3): b.add(transition([55],step))
  assert b.sample_starts(1,10) is None
