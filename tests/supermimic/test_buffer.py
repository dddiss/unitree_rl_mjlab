import torch
import pytest
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


def test_ring_and_short_imagination_starts(mdp):
  m=mdp.motion
  b=ReplayBuffer(3,1,torch.zeros(1,33),m.clip_end,m.clip_id,32)
  assert b.sample_starts(1,10) is None
  for step in range(6): b.add(transition([step+1],step))
  _,h=b.sample_starts(200,10)
  assert h['s'][...,0].min()>=3
  for step in range(3): b.add(transition([57],0,True))
  start,h=b.sample_starts(100,10)
  # A one-step clip tail remains usable; histories stay inside each reset.
  assert torch.all(start['f']==57)
  assert torch.all(h['s']==0)


def test_imagination_samples_short_and_long_clips_without_unwritten_slots(mdp):
  m=mdp.motion
  b=ReplayBuffer(20,3,torch.zeros(3,33),m.clip_end,m.clip_id,32)
  b.add(transition([1,56,57],0))
  torch.manual_seed(41)
  start,_=b.sample_starts(600,2)
  assert set(start['f'].tolist())=={1,56,57}
  assert torch.all(start['s']==0)
  counts=torch.stack([(start['f']==f).sum() for f in (1,56,57)])
  assert torch.all((counts>150)&(counts<250))


def test_strict_windows_allow_terminal_or_truncated_last_step(mdp):
  m = mdp.motion
  b = ReplayBuffer(8, 2, torch.zeros(2, 33), m.clip_end, m.clip_id, 2)
  for step in range(8):
    tr = transition([step + 1] * 2, step)
    if step == 7:
      tr["terminated"][0] = True
      tr["truncated"][1] = True
    b.add(tr)
  w = b.sample_windows(100, 8, strict=True)
  assert torch.all(w["mask"] == 1)
  assert torch.all((w["terminated"] | w["truncated"])[:, -1])
  assert torch.equal(w["s"][..., 0], torch.arange(8).expand(100, -1))
  assert b._window_choices[8].tolist() == [[0, 0], [0, 1]]


def test_strict_windows_exclude_early_end_and_unmarked_reset(mdp):
  m = mdp.motion
  b = ReplayBuffer(8, 3, torch.zeros(3, 33), m.clip_end, m.clip_id, 2)
  for step in range(8):
    tr = transition([step + 1] * 3, step)
    tr["terminated"][0] = step == 3
    # Environment 1 is manually reset without a termination flag.
    tr["ep_step"][1] = step if step < 4 else step - 4
    tr["s"][:, 1] = torch.arange(3)
    b.add(tr)
  w = b.sample_windows(20, 8, strict=True)
  assert torch.all(w["s"][..., 1] == 2)
  assert b._window_choices[8].tolist() == [[0, 2]]


def test_strict_windows_without_legal_start_return_none(mdp):
  m = mdp.motion
  b = ReplayBuffer(8, 1, torch.zeros(1, 33), m.clip_end, m.clip_id, 2)
  b.add(transition([1], 0))
  assert b.sample_windows(10, 8, strict=True) is None
  with pytest.raises(ValueError, match="Not enough collected steps"):
    b.sample_windows(10, 8)
  for step in range(1, 8):
    b.add(transition([step + 1], step % 4, step == 3))
  assert b.sample_windows(10, 8, strict=True) is None
  # Warmup still samples this window and masks the post-terminal episode.
  w = b.sample_windows(10, 8)
  assert w["mask"].tolist() == [[1, 1, 1, 1, 0, 0, 0, 0]] * 10


def test_strict_windows_follow_chronology_through_ring_wrap(mdp):
  m = mdp.motion
  b = ReplayBuffer(5, 1, torch.zeros(1, 33), m.clip_end, m.clip_id, 2)
  for step in range(8):
    b.add(transition([step + 1], step))
  w = b.sample_windows(200, 3, strict=True)
  states = w["s"][..., 0]
  assert torch.all(states[:, 0] >= 3)
  assert torch.all(states[:, -1] <= 7)
  assert torch.all(states[:, 1:] - states[:, :-1] == 1)
  assert b._window_choices[3].tolist() == [[0, 0], [1, 0], [2, 0]]
  # A full-capacity window can cross physical slot zero, never newest->oldest.
  w = b.sample_windows(10, 5, strict=True)
  assert w["s"][..., 0].tolist() == [[3, 4, 5, 6, 7]] * 10


def test_strict_window_cache_reused_until_add(mdp):
  m = mdp.motion
  b = ReplayBuffer(8, 1, torch.zeros(1, 33), m.clip_end, m.clip_id, 2)
  for step in range(8):
    b.add(transition([step + 1], step, step == 7))
  assert b.sample_windows(1, 8, strict=True) is not None
  choices = b._window_choices[8]
  b.sample_windows(1, 8, strict=True)
  assert b._window_choices[8] is choices
  b.add(transition([1], 0))
  assert b._window_choices == {}
  assert b.sample_windows(1, 8, strict=True) is None
  # A one-transition window is valid even if its transition ends an episode.
  assert b.sample_windows(10, 1, strict=True) is not None
  assert len(b._window_choices[1]) == 8
