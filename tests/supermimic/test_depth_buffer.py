import torch
from supermimic.rl.buffer import ReplayBuffer


def test_successor_sharing_terminal_pending_and_wrap():
  b=ReplayBuffer(4,2,torch.zeros(2,33),torch.tensor([999]),torch.zeros(1000,dtype=torch.long),32,4)
  def image(values): return torch.tensor(values,dtype=torch.uint8)[:,None,None].expand(2,18,32)
  def add(values,next_values,steps,done=(False,False)):
    b.add(dict(s=torch.zeros(2,71),s1=torch.ones(2,71),a=torch.zeros(2,29),a_prev=torch.zeros(2,29),
               r=torch.ones(2),f=torch.ones(2,dtype=torch.long),ep_step=torch.tensor(steps),
               terminated=torch.tensor(done),truncated=torch.zeros(2,dtype=torch.bool),
               depth=image(values),depth_next=image(next_values)))
  add([10,20],[11,21],[0,0])
  assert b.depth.next_frames(torch.tensor([0,0]),torch.arange(2))[:,0,0].tolist()==[11,21]
  add([11,21],[12,22],[1,1],(True,False))
  add([100,22],[101,23],[0,2])  # Row 2 is reset in env 0, continuing in env 1.
  rows=torch.tensor([[0,1,2],[0,1,2]])
  envs=torch.tensor([[0],[1]])
  assert b.depth.next_frames(rows,envs)[...,0,0].tolist()==[[11,12,101],[21,22,23]]
  hist=b.depth.history(b,torch.tensor([2,2]),torch.arange(2),4)
  assert hist[...,0,0].tolist()==[[100]*4,[20,20,21,22]]
  add([101,23],[102,24],[1,3])
  add([102,24],[103,25],[2,4])  # Overwrite row 0; oldest retained row is 1.
  assert b.depth.next_frames(torch.tensor([1]),torch.tensor([0]))[0,0,0]==12
  hist=b.depth.history(b,torch.tensor([3,3]),torch.arange(2),4)
  assert hist[...,0,0].tolist()==[[100,100,101,102],[21,22,23,24]]
  add([103,25],[104,26],[3,5])  # Overwrite terminal row and recycle its pool slot.
  assert b.depth.terminal_index[1].eq(0).all()
  assert len(b.depth.free)==len(b.depth.pool)-1


def test_sampled_visual_windows_keep_real_terminal_targets():
  b=ReplayBuffer(8,1,torch.zeros(1,33),torch.tensor([9]),torch.zeros(10,dtype=torch.long),32,4)
  for t in range(4):
    b.add(dict(s=torch.zeros(1,71),s1=torch.ones(1,71),a=torch.zeros(1,29),a_prev=torch.zeros(1,29),
               r=torch.ones(1),f=torch.ones(1,dtype=torch.long),ep_step=torch.tensor([t%2]),
               terminated=torch.tensor([t%2==1]),truncated=torch.tensor([False]),
               depth=torch.full((1,18,32),10*t,dtype=torch.uint8),
               depth_next=torch.full((1,18,32),10*(t+1) if t%2==0 else 199,dtype=torch.uint8)))
  w=b.sample_windows(20,2,strict=True)
  assert w['depth_next'][:,-1].eq(199).all()
  assert w['mask'].eq(1).all()
  assert (w['depth_history'][:,0]==w['depth_history'][:,-1]).all()
