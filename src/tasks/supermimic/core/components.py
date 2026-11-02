"""Reward/failure contracts and startup assembly; no simulator dependency.

Register a measured reward with ``reward_component`` (or a failure with
``failure_component``), then put that function in the normal mjlab term config.
Names in the config are labels, not algorithm dispatch keys. New sensor targets
automatically become replay columns, WM outputs, losses and logged metrics.
"""
from dataclasses import dataclass, field, replace
from typing import Callable

import torch


@dataclass(frozen=True)
class Prediction:
  """Supervise the RAW scalar; weighting by reward * dt happens only afterward."""
  loss_weight: float = 1.0
  loss_weight_field: str | None = None  # Compatibility with existing rl_cfg keys.
  nonnegative: bool = True

  def weight(self, cfg, name):
    overrides = getattr(cfg, "reward_loss_weights", {})
    return overrides.get(name, getattr(cfg, self.loss_weight_field, self.loss_weight)
                         if self.loss_weight_field else self.loss_weight)


@dataclass(frozen=True)
class RewardComponent:
  tensor: Callable | None = None
  prepare: Callable | None = None
  prediction: Prediction | None = None
  sense_before_reward: bool = False

  def __post_init__(self):
    if (self.tensor is None) == (self.prediction is None):
      raise ValueError("Declare exactly one imagined reward source: tensor formula or prediction")


@dataclass(frozen=True)
class RewardBinding:
  name: str
  component: RewardComponent
  params: dict = field(default_factory=dict)

  def value(self, context):
    return self.component.tensor(context, **self.params)


@dataclass(frozen=True)
class FailureComponent:
  error: Callable
  soft_width: float
  prepare: Callable | None = None
  requires_height: bool = False


@dataclass(frozen=True)
class FailureBinding:
  name: str
  component: FailureComponent
  threshold: float
  params: dict = field(default_factory=dict)

  def hard(self, context):
    return self.component.error(context, **self.params) > self.threshold

  def soft(self, context):
    return torch.sigmoid((self.component.error(context, **self.params) - self.threshold)
                         / self.component.soft_width)


def reward_component(tensor=None, *, prepare=None, prediction=None,
                     sense_before_reward=False):
  """Attach the imagined implementation/label metadata to a manager function."""
  def decorate(measured):
    measured.supermimic_reward = RewardComponent(tensor, prepare, prediction,
                                                sense_before_reward)
    return measured
  return decorate


def failure_component(error, *, soft_width, prepare=None, requires_height=False):
  def decorate(measured):
    measured.supermimic_failure = FailureComponent(error, soft_width, prepare, requires_height)
    return measured
  return decorate


def component_of(term):
  return getattr(getattr(term, "func", None), "supermimic_reward", None)


def reward_binding(name, term, body_names):
  component = component_of(term)
  if component is None:
    return None
  params = dict(term.params)
  if component.prepare is not None:
    params = component.prepare(params, body_names)
  else:
    params.pop("command_name", None)
    params.pop("asset_cfg", None)
  return RewardBinding(name, component, params)


def failure_binding(name, term, body_names):
  component = getattr(getattr(term, "func", None), "supermimic_failure", None)
  if component is None:
    return None
  params = dict(term.params)
  threshold = params.pop("threshold")
  if component.prepare is not None:
    params = component.prepare(params, body_names)
  else:
    params.pop("command_name", None)
    params.pop("asset_cfg", None)
  return FailureBinding(name, component, threshold, params)


def prediction_bindings(rewards):
  """Keep config order, including weight-zero terms that still supervise WM."""
  return tuple((name, term, component_of(term)) for name, term in rewards.items()
               if term is not None and component_of(term) is not None
               and component_of(term).prediction is not None)


def bind_analytic_manager_terms(cfg):
  """Resolve aliases at startup, preserving native measured body formulas.

  Pure tensor functions opt in with analytic_reward/analytic_failure. Existing
  manager functions keep their measured path unless explicitly marked.
  """
  for name, term in cfg.rewards.items():
    if term is not None and getattr(term.func, "use_tensor_measurement", False):
      original = term.func
      # A bound function carries its component metadata for from_env_cfg().
      def measured(env, component_name=name, **_):
        return env.shared_reward_terms()[component_name]
      measured.supermimic_reward = original.supermimic_reward
      cfg.rewards[name] = replace(term, func=measured)
  for name, term in cfg.terminations.items():
    if term is not None and getattr(term.func, "use_tensor_failure", False):
      original = term.func
      def measured_failure(env, component_name=name, **_):
        return env.shared_failure_terms()[component_name]
      measured_failure.supermimic_failure = original.supermimic_failure
      cfg.terminations[name] = replace(term, func=measured_failure)


def analytic_reward(function):
  """Convenience: define one ``function(StepContext, **params)`` for both paths.

  Use the decorated function directly in RewardTermCfg; make_env binds its
  configured name before managers are constructed.
  """
  function.supermimic_reward = RewardComponent(tensor=function)
  function.use_tensor_measurement = True
  return function


def analytic_failure(*, soft_width, requires_height=False):
  """Define one error(ctx, **params); use threshold in TerminationTermCfg.

  The real path compares error > threshold; imagination applies the same
  sigmoid convention as existing height/tilt failures. Never use for timeouts.
  """
  def decorate(function):
    function.supermimic_failure = FailureComponent(function, soft_width,
                                                  requires_height=requires_height)
    function.use_tensor_failure = True
    return function
  return decorate
