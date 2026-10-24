from types import SimpleNamespace

import pytest
import torch

from supermimic.mdp.sampling import EpisodeBinReturns, ReturnBasedSampler


def motion():
  return SimpleNamespace(
    device=torch.device('cpu'), clip_len=torch.tensor([6, 4]), bin_frames=2,
    num_bins=5, frame_bin=torch.tensor([0, 0, 1, 1, 2, -1, 3, 3, 4, -1]),
  )


@pytest.mark.parametrize('terminated', [True, False])
def test_each_visited_bin_receives_its_return_to_go(terminated):
  episodes = EpisodeBinReturns(motion(), 1, gamma=.5)
  # Starting in the middle of bin 0 still attributes its return to that start.
  for bin_id, reward in ((0, 1.), (0, 2.), (1, 3.), (1, 4.)):
    episodes.add(torch.tensor([bin_id]), torch.tensor([reward]))
  bins, values = episodes.finish(torch.tensor([0]), torch.tensor([terminated]))
  assert bins.tolist() == [0, 1]
  expected = torch.tensor([1 + .5*2 + .25*3 + .125*4, 3 + .5*4])
  if not terminated:
    # Both bins use the mean of the full episode (2.5), not a suffix mean.
    expected += torch.tensor([.5**4, .5**2]) * 2.5 / (1-.5)
  torch.testing.assert_close(values, expected)
  assert episodes.num_entries.tolist() == [0]


def test_initial_bin_retains_previous_bootstrap_formula():
  gamma = .99
  episodes = EpisodeBinReturns(motion(), 1, gamma)
  rewards = torch.tensor([.1, .2, -.3, .4, .5])
  for index, reward in enumerate(rewards):
    episodes.add(torch.tensor([index // 2]), reward[None])
  bins, values = episodes.finish(torch.tensor([0]), torch.tensor([False]))
  expected = (rewards * gamma ** torch.arange(5)).sum()
  expected += gamma**5 * rewards.mean() / (1-gamma)
  assert bins.tolist() == [0, 1, 2]
  torch.testing.assert_close(values[0], expected)


def test_multiple_episodes_update_shared_bin_once_after_averaging():
  m = motion()
  episodes = EpisodeBinReturns(m, 2, gamma=.5)
  sampler = ReturnBasedSampler(m, gamma=.5)
  sampler.bin_return[0] = 10
  episodes.add(torch.tensor([0, 0]), torch.tensor([1., 3.]))
  episodes.add(torch.tensor([1, 1]), torch.tensor([2., 4.]))
  bins, returns = episodes.finish(torch.tensor([0, 1]), torch.tensor([True, True]))
  sampler.update_returns(bins, returns)
  # First bin has returns 2 and 5; second bin has returns 2 and 4.
  torch.testing.assert_close(sampler.bin_return, torch.tensor([8.7, .6, 0., 0., 0.]))


def test_partial_finish_retains_other_episode_and_new_clip_is_independent():
  episodes = EpisodeBinReturns(motion(), 2, gamma=.5)
  episodes.add(torch.tensor([0, 0]), torch.tensor([1., 10.]))
  episodes.finish(torch.tensor([0]), torch.tensor([True]))
  assert episodes.steps.tolist() == [0, 1]
  # Environment 0 starts in a different clip, while environment 1 continues.
  episodes.add(torch.tensor([3, 1]), torch.tensor([2., 20.]))
  episodes.add(torch.tensor([4, 1]), torch.tensor([4., 30.]))
  bins, returns = episodes.finish(torch.tensor([0, 1]), torch.tensor([True, True]))
  assert bins.tolist() == [3, 0, 4, 1]
  torch.testing.assert_close(returns, torch.tensor([4., 27.5, 4., 35.]))
  assert episodes.steps.tolist() == [0, 0]


def test_final_transition_uses_source_bin_without_crossing_clip_boundary():
  m = motion()
  episodes = EpisodeBinReturns(m, 2, gamma=.5)
  # f is each clip's final goal frame; f - 1 remains a legal reset source.
  target_frames = torch.tensor([5, 9])
  assert m.frame_bin[target_frames].tolist() == [-1, -1]
  episodes.add(m.frame_bin[target_frames - 1], torch.tensor([1., 2.]))
  bins, returns = episodes.finish(torch.tensor([0, 1]), torch.tensor([False, False]))
  assert bins.tolist() == [2, 4]
  torch.testing.assert_close(returns, torch.tensor([2., 4.]))


def test_empty_finish_and_restore_reset_discard_old_accumulators():
  episodes = EpisodeBinReturns(motion(), 2, gamma=.5)
  bins, returns = episodes.finish(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.bool))
  assert bins.numel() == returns.numel() == 0
  episodes.add(torch.tensor([0, 1]), torch.tensor([9., 8.]))
  episodes.reset()
  episodes.add(torch.tensor([3, 3]), torch.tensor([1., 2.]))
  bins, returns = episodes.finish(torch.tensor([0, 1]), torch.tensor([True, True]))
  assert bins.tolist() == [3, 3]
  torch.testing.assert_close(returns, torch.tensor([1., 2.]))


def test_long_episode_does_not_divide_by_underflowed_prefix_discount():
  m = SimpleNamespace(device=torch.device('cpu'), clip_len=torch.tensor([301]), bin_frames=50)
  episodes = EpisodeBinReturns(m, 1, gamma=.5)
  for frame in range(300):
    episodes.add(torch.tensor([frame // 50]), torch.tensor([1.]))
  bins, values = episodes.finish(torch.tensor([0]), torch.tensor([False]))
  assert bins.tolist() == list(range(6))
  # With reward 1 and the retained mean-reward tail, every entry has return 2,
  # even though gamma ** 300 underflows in float32.
  torch.testing.assert_close(values, torch.full((6,), 2.))
