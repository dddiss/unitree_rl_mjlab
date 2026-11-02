"""Lazy manager adapters; tensor-only modules also work without mjlab."""
from importlib import import_module

_MODULES = ("commands", "observations", "rewards", "terminations")

def __getattr__(name):
  if name.startswith("_"):
    raise AttributeError(name)
  if name in _MODULES:
    return import_module(f"{__name__}.{name}")
  for module in _MODULES:
    mod = import_module(f"{__name__}.{module}")
    if hasattr(mod, name):
      value = getattr(mod, name)
      globals()[name] = value
      return value
  mod = import_module("mjlab.envs.mdp")
  if hasattr(mod, name):
    return getattr(mod, name)
  raise AttributeError(name)
