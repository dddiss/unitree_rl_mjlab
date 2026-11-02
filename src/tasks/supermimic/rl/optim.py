"""Optimizer ownership and parameter freezing (input gradients stay enabled)."""
from contextlib import contextmanager

def parameter_groups(model, lr, visual_groups=(), excluded=()):
  """Keep each parameter in exactly one optimizer group with its own base LR."""
  groups, excluded = [], {id(p) for p in excluded}
  for module, rate in visual_groups:
    parameters = list(module.parameters())
    excluded.update(id(p) for p in parameters)
    groups.append({"params": parameters, "lr": rate, "base_lr": rate})
  return [{"params": [p for p in model.parameters() if id(p) not in excluded],
           "lr": lr, "base_lr": lr}, *groups]


@contextmanager
def frozen(*models):
  flags = [[p.requires_grad for p in m.parameters()] for m in models]
  try:
    for model in models:
      model.requires_grad_(False)
    yield
  finally:
    for model, previous in zip(models, flags):
      for p, flag in zip(model.parameters(), previous):
        p.requires_grad_(flag)
