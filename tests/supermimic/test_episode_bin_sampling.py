from types import SimpleNamespace

import pytest
import torch

from supermimic.mdp.sampling import EpisodeBinReturns, ReturnBasedSampler
from supermimic.config.g1.rl_cfg import SuperMimicConfig


def motion(lengths=(6, 4), bin_frames=2):
  lengths = torch.tensor(lengths)
  start = lengths.cumsum(0) - lengths
  end = start + lengths - 1
  frame_bin = torch.full((int(lengths.sum()),), -1, dtype=torch.long)
  count = 0
  for offset, length in zip(start.tolist(), lengths.tolist()):
    for frame in range(offset, offset + length - 1, bin_frames):
      frame_bin[frame:min(frame + bin_frames, offset + length - 1)] = count
      count += 1
  return SimpleNamespace(
    device=torch.device('cpu'), clip_len=lengths, clip_start=start, clip_end=end,
    clip_id=torch.repeat_interleave(torch.arange(len(lengths)), lengths),
    bin_frames=bin_frames, num_bins=count, frame_bin=frame_bin,
  )


def test_crossing_bins_updates_only_the_initial_bin():
  episodes = EpisodeBinReturns(motion(), 1, gamma=.5, max_step_reward=4.)
  # Initialize at k=1, in the middle of bin 0. Targets 2..5 cross bins 1 and 2.
  for frame, reward in zip(range(2, 6), (1., 2., 3., 4.)):
    episodes.add(torch.tensor([frame]), torch.tensor([reward]))
  assert episodes.horizon.tolist() == [4]
  bins, scores = episodes.finish(torch.tensor([0]))
  assert bins.tolist() == [0]
  expected = (1 + .5*2 + .25*3 + .125*4) / (4 * (1 + .5 + .25 + .125))
  torch.testing.assert_close(scores, torch.tensor([expected]))
  assert episodes.steps.tolist() == [0]


@pytest.mark.parametrize('steps', [1, 4, 500])
def test_full_clip_constant_reward_has_same_score_regardless_of_length(steps):
  episodes = EpisodeBinReturns(motion((steps + 1,)), 1, max_step_reward=.1)
  for frame in range(1, steps + 1):
    episodes.add(torch.tensor([frame]), torch.tensor([.03]))
  bins, scores = episodes.finish(torch.tensor([0]))
  assert bins.tolist() == [0]
  torch.testing.assert_close(scores, torch.tensor([.3]))


@pytest.mark.parametrize('time_limit, horizon', [(3, 3), (None, 5), (10, 5)])
def test_time_limit_caps_potential_only_when_enabled(time_limit, horizon):
  episodes = EpisodeBinReturns(motion(), 1, gamma=.5, max_episode_steps=time_limit)
  for frame in range(1, 4):
    episodes.add(torch.tensor([frame]), torch.tensor([.1]))
  assert episodes.horizon.tolist() == [horizon]
  _, scores = episodes.finish(torch.tensor([0]))
  # With a 3-step timeout this is a complete episode; otherwise it is an early failure.
  torch.testing.assert_close(scores, torch.tensor([(1 - .5**3) / (1 - .5**horizon)]))


def test_early_failure_keeps_original_potential_horizon_and_has_no_tail():
  episodes = EpisodeBinReturns(motion(), 1, gamma=.5)
  for frame in (1, 2):
    episodes.add(torch.tensor([frame]), torch.tensor([.1]))
  assert episodes.steps.tolist() == [2] and episodes.horizon.tolist() == [5]
  _, scores = episodes.finish(torch.tensor([0]))
  torch.testing.assert_close(scores, torch.tensor([(1 - .5**2) / (1 - .5**5)]))
  assert scores.item() < 1


@pytest.mark.parametrize('bin_ema', [0., .1, .2, 1.])
def test_multiple_episodes_normalize_separately_then_average_shared_start_bin(bin_ema):
  m = motion()
  episodes = EpisodeBinReturns(m, 2, gamma=.5)
  sampler = ReturnBasedSampler(m, gamma=.5, bin_ema=bin_ema)
  sampler.bin_return[0] = .7
  episodes.add(torch.tensor([1, 2]), torch.tensor([.05, .1]))
  episodes.add(torch.tensor([2, 3]), torch.tensor([.1, .05]))
  # Different start frames inside bin 0 have different potential horizons, 5 and 4.
  bins, scores = episodes.finish(torch.tensor([0, 1]))
  expected = torch.tensor([.1 / (.1*(1-.5**5)/.5), .125 / (.1*(1-.5**4)/.5)])
  assert bins.tolist() == [0, 0]
  torch.testing.assert_close(scores, expected)
  sampler.update_returns(bins, scores)
  torch.testing.assert_close(sampler.bin_return[0], (1-bin_ema)*torch.tensor(.7) + bin_ema*expected.mean())
  assert torch.count_nonzero(sampler.bin_return) == 1


def test_partial_finish_retains_other_episode_and_new_clip_is_independent():
  episodes = EpisodeBinReturns(motion(), 2, gamma=.5, max_step_reward=30.)
  episodes.add(torch.tensor([1, 1]), torch.tensor([1., 10.]))
  episodes.finish(torch.tensor([0]))
  assert episodes.steps.tolist() == [0, 1]
  # Row 0 initializes at k=6, in the second clip; row 1 keeps the first episode.
  episodes.add(torch.tensor([7, 2]), torch.tensor([2., 20.]))
  episodes.add(torch.tensor([8, 3]), torch.tensor([4., 30.]))
  assert episodes.horizon.tolist() == [3, 5]
  bins, scores = episodes.finish(torch.tensor([0, 1]))
  assert bins.tolist() == [3, 0]
  torch.testing.assert_close(scores, torch.tensor([
    4 / (30*(1-.5**3)/.5), 27.5 / (30*(1-.5**5)/.5),
  ]))
  assert episodes.steps.tolist() == [0, 0]


def test_last_transition_has_horizon_one_and_never_crosses_clip_boundary():
  m = motion()
  episodes = EpisodeBinReturns(m, 2, gamma=.5)
  targets = m.clip_end
  assert m.frame_bin[targets].tolist() == [-1, -1]
  episodes.add(targets, torch.tensor([.06, .08]))
  assert episodes.horizon.tolist() == [1, 1]
  bins, scores = episodes.finish(torch.tensor([0, 1]))
  assert bins.tolist() == [2, 4]
  torch.testing.assert_close(scores, torch.tensor([.6, .8]))


def test_empty_finish_and_restore_reset_discard_old_accumulators():
  episodes = EpisodeBinReturns(motion(), 2, gamma=.5)
  bins, scores = episodes.finish(torch.empty(0, dtype=torch.long))
  assert bins.numel() == scores.numel() == 0
  episodes.add(torch.tensor([1, 3]), torch.tensor([.09, .08]))
  episodes.reset()
  episodes.add(torch.tensor([9, 9]), torch.tensor([.01, .02]))
  bins, scores = episodes.finish(torch.tensor([0, 1]))
  assert bins.tolist() == [4, 4]
  torch.testing.assert_close(scores, torch.tensor([.1, .2]))


def test_long_episode_remains_finite_after_discount_underflows():
  episodes = EpisodeBinReturns(motion((301,), bin_frames=50), 1, gamma=.5)
  for frame in range(1, 301):
    episodes.add(torch.tensor([frame]), torch.tensor([.1]))
  bins, scores = episodes.finish(torch.tensor([0]))
  assert bins.tolist() == [0]
  torch.testing.assert_close(scores, torch.ones(1))


def test_sampler_default_return_ema_matches_config_and_preserves_history_ema():
  sampler = ReturnBasedSampler(motion(), gamma=.99)
  assert sampler.bin_ema == SuperMimicConfig().bin_ema == .1
  sampler.update_returns(torch.tensor([0, 1]), torch.tensor([.8, .4]))
  torch.testing.assert_close(sampler.bin_return[:2], torch.tensor([.08, .04]))
  expected = .5/5 + .4*torch.softmax(-25*sampler.bin_return, 0) + .1/5
  torch.testing.assert_close(sampler.probabilities(), expected)
  history = sampler.past_softmax.clone()
  soft = torch.softmax(-25*sampler.bin_return, 0)
  sampler.probabilities(update_history=True)
  torch.testing.assert_close(sampler.past_softmax, .999*history + .001*soft)


@pytest.mark.parametrize('bin_ema', [-.1, 1.1, float('nan'), float('inf')])
def test_config_rejects_invalid_return_ema(bin_ema):
  with pytest.raises(ValueError, match='bin_ema'):
    SuperMimicConfig(bin_ema=bin_ema)


def test_normalized_sampler_checkpoint_round_trip():
  sampler = ReturnBasedSampler(motion())
  sampler.update_returns(torch.tensor([0, 1]), torch.tensor([.8, .4]))
  sampler.probabilities(update_history=True)
  saved = {key: value.clone() for key, value in sampler.state_dict().items()}
  restored = ReturnBasedSampler(motion())
  restored.load_state_dict(saved)
  for key, value in saved.items():
    torch.testing.assert_close(restored.state_dict()[key], value)
  torch.testing.assert_close(restored.probabilities(), sampler.probabilities())


def test_old_bootstrapped_checkpoint_resets_scores_and_history_explicitly():
  sampler = ReturnBasedSampler(motion())
  sampler.bin_return.fill_(.5)
  sampler.past_softmax.copy_(torch.tensor([.8, .05, .05, .05, .05]))
  with pytest.warns(UserWarning, match='resetting sampling statistics'):
    sampler.load_state_dict({'bin_return': torch.arange(5.), 'past_softmax': torch.ones(5)/5})
  torch.testing.assert_close(sampler.bin_return, torch.zeros(5))
  torch.testing.assert_close(sampler.past_softmax, torch.ones(5)/5)
