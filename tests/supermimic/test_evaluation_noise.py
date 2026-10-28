"""Noise diagnostics exercise the real controller, scoring, and CLI reports."""
import csv
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

from supermimic.mdp.state import read_state
from supermimic.rl.evaluation import evaluate
from supermimic.rl.networks import Policy
from test_algorithm import small_config
from test_train_resume_eval import TensorEnv


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/evaluate_supermimic.py"


class RecordingEnv(TensorEnv):
  def __init__(self, mdp):
    self.actions, self.seeds = [], []
    super().__init__(mdp)

  def reset(self, env_ids=None, *, seed=None):
    if seed is not None:
      self.seeds.append(seed)
      torch.manual_seed(seed)  # mjlab's seeded reset also seeds PyTorch.
    return super().reset(env_ids=env_ids)

  def step(self, action):
    self.actions.append(action.clone())
    return super().step(action)


@pytest.mark.parametrize("observation_noise,action_noise", [
  (False, 0.), (False, .1), (True, 0.), (True, .1),
])
def test_noise_modes_keep_timing_history_and_scoring(mdp, observation_noise, action_noise):
  env = RecordingEnv(mdp)
  cfg = small_config()
  obs_differences, goal_differences, first_targets = [], [], []

  def policy(history, goal):
    f = env.command.time_steps
    state = read_state(env)
    clean_obs = mdp.obs(state, env.action_manager.action, env.context, f)
    obs_differences.append(history[:, -1] - clean_obs)
    goal_differences.append((goal - mdp.goal(state, f, cfg.goal_frames)).reshape(2, -1, 64))
    first_targets.append(f.clone())
    # The executed action, including exploration noise, enters the next history.
    torch.testing.assert_close(history[:, -1, -29:], env.action_manager.action)
    return torch.zeros(2, 29)

  result = evaluate(env, policy, cfg, observation_noise=observation_noise,
                    action_noise=action_noise, seed=17)
  assert env.seeds == [17]
  torch.testing.assert_close(first_targets[0], mdp.motion.clip_start + 1)
  assert [clip["steps"] for clip in result["per_clip"]] == [50, 6]
  assert result["n_failed"] == 0
  for clip, length in zip(result["per_clip"], (50, 6)):
    assert clip["normalized_return"] == pytest.approx(clip["return"] / (length * .1))
  obs_delta, goal_delta = torch.stack(obs_differences), torch.stack(goal_differences)
  if observation_noise:
    assert obs_delta.abs().sum() > 0 and goal_delta[..., 58:].abs().sum() > 0
    assert (obs_delta.abs() <= mdp.obs_noise + 1e-6).all()
    assert goal_delta[..., 58:].abs().max() <= .05 + 1e-6
  else:
    assert obs_delta.count_nonzero() == goal_delta.count_nonzero() == 0
  assert goal_delta[..., :58].count_nonzero() == 0
  generator = torch.Generator().manual_seed(17)
  for action in env.actions:
    expected = action_noise * torch.randn(2, 29, generator=generator) if action_noise else torch.zeros_like(action)
    torch.testing.assert_close(action, expected)


def test_noise_repeats_by_seed_and_does_not_update_policy_or_normalizers(mdp):
  env = RecordingEnv(mdp)
  cfg = small_config()
  policy = Policy(93, 64 * cfg.goal_frames, cfg.hist, hidden=cfg.policy_hidden).eval()
  before = {name: value.clone() for name, value in policy.state_dict().items()}
  result = evaluate(env, policy, cfg, observation_noise=True, action_noise=.1, seed=3)
  actions = torch.stack(env.actions)
  env.actions.clear()
  repeated = evaluate(env, policy, cfg, observation_noise=True, action_noise=.1, seed=3)
  assert result == repeated
  torch.testing.assert_close(torch.stack(env.actions), actions, rtol=0, atol=0)
  env.actions.clear()
  evaluate(env, policy, cfg, observation_noise=True, action_noise=.1, seed=4)
  assert not torch.equal(torch.stack(env.actions), actions)
  for name, value in policy.state_dict().items():
    torch.testing.assert_close(value, before[name], rtol=0, atol=0)
  assert all(p.grad is None for p in policy.parameters())


@pytest.mark.parametrize("std", [0., .1])
def test_action_clipping_is_opt_in_and_applies_after_noise(mdp, std):
  policy = lambda h, g: torch.full((2, 29), 25.)
  cfg = small_config()
  unclipped = RecordingEnv(mdp)
  evaluate(unclipped, policy, cfg, action_noise=std, seed=0)
  clipped = RecordingEnv(mdp)
  evaluate(clipped, policy, cfg, action_noise=std, seed=0, action_clip=20.)
  assert torch.stack(unclipped.actions).min() > 20
  torch.testing.assert_close(torch.stack(clipped.actions), torch.stack(unclipped.actions).clamp(-20, 20))


@pytest.fixture
def cli(monkeypatch):
  monkeypatch.setattr(sys, "path", list(sys.path))
  return runpy.run_path(str(SCRIPT))


@pytest.mark.parametrize("flags", [
  ["--action-noise", "-0.1"], ["--action-noise", "nan"],
  ["--action-clip", "0"], ["--action-clip", "inf"],
  ["--seeds", "-1"], ["--seeds", "4294967296"], ["--seeds", "1", "1"],
])
def test_invalid_noise_arguments(cli, flags):
  with pytest.raises(SystemExit):
    cli["parse_args"](["--motion-file", "motions", "--checkpoint-file", "last.pt", *flags])


def test_all_modes_reports_and_cached_run_directory(cli, tmp_path):
  args = cli["parse_args"](["--motion-file", "motions", "--run-dir", str(tmp_path),
                            "--noise-mode", "all", "--seeds", "2", "4"])
  cfg, ck = small_config(action_noise=.2), {"iteration": 64000}
  checkpoint = tmp_path / "ckpt_064000.pt"
  calls = []

  def score(env, policy, config, record_dir, **options):
    calls.append(options)
    return dict(norm_return=options["seed"] / 10, norm_return_framew=.4,
                n_failed=options["seed"], failed=["clip"], per_clip=[])

  run = cli["evaluate_noise_modes"]
  run(None, None, cfg, ck, checkpoint, tmp_path, args, score)
  assert len(calls) == 8
  for index, mode in enumerate(cli["NOISE_MODES"]):
    for offset, seed in enumerate((2, 4)):
      assert calls[index * 2 + offset] == dict(seed=seed, observation_noise=mode in ("observation", "both"),
        action_noise=.2 if mode in ("action", "both") else 0., action_clip=None)
  root = tmp_path / "noise_evaluation" / checkpoint.stem
  with (root / "eval.csv").open() as handle:
    rows = list(csv.DictReader(handle))
  assert len(rows) == 8 and {row["noise_mode"] for row in rows} == set(cli["NOISE_MODES"])
  summary = json.loads((root / "summary.json").read_text())
  assert summary["seeds"] == [2, 4]
  for mode in summary["modes"]:
    assert mode["norm_return_mean"] == pytest.approx(.3)
    assert mode["norm_return_std"] == pytest.approx(.1)
    assert mode["n_failed_mean"] == 3 and mode["n_failed_std"] == 1
  assert not (tmp_path / "eval.csv").exists()
  run(None, None, cfg, ck, checkpoint, tmp_path, args, score)
  assert len(calls) == 8  # Cached reports, no duplicate evaluations/CSV rows.
  with (root / "eval.csv").open() as handle:
    assert len(list(csv.DictReader(handle))) == 8
  args.action_noise = .3
  with pytest.raises(ValueError, match="different diagnostic settings"):
    run(None, None, cfg, ck, checkpoint, tmp_path, args, score)


@pytest.mark.parametrize("diagnostic", [False, True])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("visual", [False, True])
def test_cli_keeps_legacy_default_and_closes_environment(cli, tmp_path, monkeypatch, diagnostic, fail, visual):
  calls, env_options = [], []
  closed = []
  env = NS(close=lambda: closed.append(True))
  config_module = ModuleType("src.tasks.supermimic.config")
  def make_env(*args, **kwargs):
    env_options.append(kwargs)
    return env
  config_module.make_env = make_env
  motion_module = ModuleType("src.tasks.supermimic.mdp.motion_lib")
  motion_module._resolve_motion_files = lambda path: ["a.npz", "b.npz"]
  eval_module = ModuleType("src.tasks.supermimic.rl.evaluation")
  eval_module.load_policy = lambda *args: (None, small_config(), {"iteration": 42})
  eval_module.read_checkpoint_config = lambda *args: small_config(depth_enabled=visual)
  def score(*args, **kwargs):
    calls.append(kwargs)
    if fail:
      raise RuntimeError("evaluation failed")
    return dict(norm_return=.8, norm_return_framew=.7, n_failed=0, failed=[], per_clip=[])
  eval_module.evaluate = score
  for module in (config_module, motion_module, eval_module):
    monkeypatch.setitem(sys.modules, module.__name__, module)
  flags = [str(SCRIPT), "--motion-file", "motions", "--checkpoint-file", str(tmp_path / "ckpt.pt"),
           "--output-dir", str(tmp_path), "--device", "cpu"]
  if diagnostic:
    flags += ["--noise-mode", "all", "--action-noise", ".15", "--action-clip", "20"]
  monkeypatch.setattr(sys, "argv", flags)
  if fail:
    with pytest.raises(RuntimeError, match="evaluation failed"):
      cli["main"]()
  else:
    cli["main"]()
    if diagnostic:
      assert len(calls) == 12  # Four modes × default seeds 0, 1, 2.
      assert {call["seed"] for call in calls} == {0, 1, 2}
      assert all(call["action_clip"] == 20 for call in calls)
      assert {call["action_noise"] for call in calls} == {0., .15}
      assert not (tmp_path / "eval.csv").exists()
    else:
      assert calls == [{}]
      assert (tmp_path / "ckpt.json").exists() and (tmp_path / "eval.csv").exists()
  assert env_options == [dict(num_envs=2, device="cpu", evaluation=True, depth_enabled=visual)]
  assert closed == [True]
