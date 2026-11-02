import torch
from supermimic.rl.buffer import DepthReplay, ReplayBuffer
from test_buffer import transition


def test_optional_depth_storage_preserves_state_sampling_and_rng():
  args = (4, 2, torch.zeros(2, 33), torch.tensor([9]), torch.zeros(10, dtype=torch.long), 32)
  plain = ReplayBuffer(*args)
  visual = ReplayBuffer(*args, depth_history_length=4)
  assert plain.depth is None
  assert isinstance(visual.depth, DepthReplay)
  for t in range(6):
    tr = transition([t + 1] * 2, t % 3, done=t % 3 == 2)
    plain.add(tr)  # Nonvisual transitions need no image fields.
    visual.add({**tr,
                "depth": torch.full((2, 18, 32), t, dtype=torch.uint8),
                "depth_next": torch.full((2, 18, 32), 199 if t % 3 == 2 else t + 1,
                                         dtype=torch.uint8)})

  # Ring wrap and episode boundaries must produce identical state samples.
  for sample in (lambda b: (b.sample_windows(20, 2, strict=True),),
                 lambda b: b.sample_starts(20, 4)):
    torch.manual_seed(17)
    expected = sample(plain)
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(17)
    actual = sample(visual)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    for plain_sample, visual_sample in zip(expected, actual):
      assert "depth_history" not in plain_sample and "depth_next" not in plain_sample
      for key, value in plain_sample.items():
        assert torch.equal(value, visual_sample[key]), key
    assert "depth_history" in actual[0]


def test_successor_sharing_terminal_pending_and_wrap():
  b=ReplayBuffer(4,2,torch.zeros(2,33),torch.tensor([999]),torch.zeros(1000,dtype=torch.long),32,4)
  def image(values): return torch.tensor(values,dtype=torch.uint8)[:,None,None].expand(2,18,32)
  def add(values,next_values,steps,done=(False,False)):
    b.add(dict(s=torch.zeros(2,71),s1=torch.ones(2,71),a=torch.zeros(2,29),a_prev=torch.zeros(2,29),
               r=torch.ones(2),f=torch.ones(2,dtype=torch.long),ep_step=torch.tensor(steps),
               terminated=torch.tensor(done),truncated=torch.zeros(2,dtype=torch.bool),
               depth=image(values),depth_next=image(next_values),terrain_tolerance=torch.zeros(2,dtype=torch.bool),
               reference_xy_offset=torch.zeros(2,2),env_origin=torch.zeros(2,3)))
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
               terminated=torch.tensor([t%2==1]),truncated=torch.tensor([False]),terrain_tolerance=torch.zeros(1,dtype=torch.bool),
               reference_xy_offset=torch.zeros(1,2),env_origin=torch.zeros(1,3),
               depth=torch.full((1,18,32),10*t,dtype=torch.uint8),
               depth_next=torch.full((1,18,32),10*(t+1) if t%2==0 else 199,dtype=torch.uint8)))
  w=b.sample_windows(20,2,strict=True)
  assert w['depth_next'][:,-1].eq(199).all()
  assert w['mask'].eq(1).all()
  assert (w['depth_history'][:,0]==w['depth_history'][:,-1]).all()
