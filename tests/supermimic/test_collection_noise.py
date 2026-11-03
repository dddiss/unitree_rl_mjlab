"""Fixed exploration groups, actual-action histories, and checkpoint compatibility."""
import json

import pytest
import torch

from supermimic.config.g1.rl_cfg import SuperMimicConfig
from supermimic.rl.runner import SuperMimicRunner, _sample_action_noise_free_mask
from test_algorithm import small_config
from test_train_resume_eval import TensorEnv


def test_default_noise_partition():
  cfg = SuperMimicConfig()
  assert cfg.action_noise == .1
  assert cfg.collect_det_frac == .25
  assert round(cfg.collect_det_frac * cfg.num_envs) == 1024


@pytest.mark.parametrize("num_envs,fraction,count", [
  (4096, .25, 1024), (10, .25, 2), (6, .25, 2), (8, 0., 0), (8, 1., 8),
])
def test_random_partition_size(num_envs, fraction, count):
  mask = _sample_action_noise_free_mask(num_envs, fraction, seed=0, device="cpu")
  assert mask.shape == (num_envs,)
  assert mask.dtype == torch.bool and mask.device == torch.device("cpu")
  assert mask.sum().item() == count


def test_partition_seed_is_reproducible_and_independent_of_training_rng():
  with torch.random.fork_rng(devices=[]):
    before = torch.get_rng_state().clone()
    first = _sample_action_noise_free_mask(4096, .25, seed=7, device="cpu")
    assert torch.equal(torch.get_rng_state(), before)
    torch.rand(100)
    after_training_draws = torch.get_rng_state().clone()
    repeated = _sample_action_noise_free_mask(4096, .25, seed=7, device="cpu")
    different = _sample_action_noise_free_mask(4096, .25, seed=8, device="cpu")
    assert torch.equal(torch.get_rng_state(), after_training_draws)
  assert torch.equal(first, repeated)
  assert not torch.equal(first, different)
  # Both contiguous environment groups contain noisy and noise-free rows.
  assert 0 < first[:1756].sum().item() < 1756
  assert 0 < first[1756:].sum().item() < 2340


@pytest.mark.parametrize("fraction", [-.01, 1.01, float("nan"), float("inf")])
def test_invalid_noise_fraction(fraction):
  with pytest.raises(ValueError, match="collect_det_frac"):
    SuperMimicConfig(collect_det_frac=fraction)


@pytest.mark.parametrize("fraction,std", [(0., .1), (.5, .1), (.5, .3), (1., .1), (.5, 0.)])
def test_fixed_partition_survives_partial_resets_and_stores_executed_actions(
  mdp, tmp_path, monkeypatch, fraction, std,
):
  # Two rows let both groups run through different episode boundaries.
  cfg = small_config(collect_det_frac=fraction, action_noise=std, seed=1)
  env = TensorEnv(mdp)
  runner = SuperMimicRunner(env, cfg, tmp_path)
  mask = runner._action_noise_free_mask.clone()
  if fraction == .5:
    assert mask.tolist() == [False, True]  # Exercise a non-prefix partition.
  env.command.request_frames(mdp.motion.clip_start.clone())
  env.reset()
  assert torch.equal(runner._action_noise_free_mask, mask)
  clean_actions, draws, history_actions, initial_rows = [], [], [], []
  controller = runner.controller
  assert controller.noise is True and controller.update_norm is True
  original_controller = runner.controller
  original_randn = torch.randn_like
  original_step = env.step

  def capture_controller():
    initial_rows.append(env.episode_length_buf == 0)
    history_actions.append(env.action_manager.action.clone())
    clean = original_controller()
    # Observation noise never overwrites last_action, including after resets.
    torch.testing.assert_close(controller.history[:, -1, -29:], history_actions[-1])
    clean_actions.append(clean.clone())
    return clean

  def capture_noise(action):
    draw = original_randn(action)
    draws.append(draw.clone())
    return draw

  def short_first_clip(action):
    result = original_step(action)
    # Row 0 fails every three steps; row 1 completes its six-step motion.
    if env.episode_length_buf[0] == 3:
      env.last_transition["terminated"][0] = True
      env.pending[0] = True
    return result

  monkeypatch.setattr(runner, "controller", capture_controller)
  monkeypatch.setattr(torch, "randn_like", capture_noise)
  monkeypatch.setattr(env, "step", short_first_clip)
  runner.collect(8)
  assert torch.equal(runner._action_noise_free_mask, mask)
  for i, (clean, draw) in enumerate(zip(clean_actions, draws)):
    expected_noise = std * draw
    expected_noise[mask] = 0
    expected = (clean + expected_noise).clamp(-20, 20)
    torch.testing.assert_close(runner.buffer.data["a"][i], expected, rtol=0, atol=0)
    torch.testing.assert_close(runner.buffer.data["a_prev"][i], history_actions[i])
    if i:
      continuing = ~initial_rows[i]
      torch.testing.assert_close(history_actions[i][continuing], runner.buffer.data["a"][i-1][continuing])
  assert len(draws) == 8 and runner.buffer.size == 8
  assert [rows.tolist() for rows in initial_rows[:7]] == [
    [True, True], [False, False], [False, False], [True, False],
    [False, False], [False, False], [True, True],
  ]
  assert json.loads((tmp_path / "config.json").read_text())["collect_det_frac"] == fraction


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("saved_fraction", [0., .5])
def test_resume_uses_current_partition_and_saves_it(mdp, tmp_path, legacy, saved_fraction):
  original = SuperMimicRunner(
    TensorEnv(mdp), small_config(collect_det_frac=saved_fraction, seed=1), tmp_path / "original",
  )
  original.collect(2)
  checkpoint = tmp_path / "checkpoint.pt"
  original.save(checkpoint)
  if legacy:
    saved = torch.load(checkpoint, weights_only=False)
    del saved["config"]["collect_det_frac"]
    torch.save(saved, checkpoint)
  cfg = small_config(collect_det_frac=.5, seed=1)
  resumed = SuperMimicRunner(TensorEnv(mdp), cfg, tmp_path / "resumed")
  mask = resumed._action_noise_free_mask.clone()
  resumed.load(checkpoint)
  assert torch.equal(resumed._action_noise_free_mask, mask)
  assert mask.tolist() == [False, True]
  if saved_fraction == .5:
    assert torch.equal(mask, original._action_noise_free_mask)
  assert resumed.cfg.collect_det_frac == .5
  assert resumed.cfg.action_noise == original.cfg.action_noise
  assert resumed.buffer.size == 0 and resumed.env_steps == original.env_steps
  for name, tensor in original.algorithm.policy.state_dict().items():
    torch.testing.assert_close(resumed.algorithm.policy.state_dict()[name], tensor)
  resumed.save(tmp_path / "resumed.pt")
  saved = torch.load(tmp_path / "resumed.pt", weights_only=False)
  assert saved["config"]["collect_det_frac"] == .5
