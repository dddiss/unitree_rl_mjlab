"""Ordered auxiliary reward labels, shared by environment, replay and learner.

The storage key/old public attribute remain compatible with existing foot-head
checkpoints. The schema itself supports any registered scalar learned reward.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardTargetSchema:
  names: tuple[str, ...] = ()
  predictions: tuple = ()
  field: str = "foot_costs"

  @classmethod
  def from_mdp(cls, mdp):
    bindings = mdp.prediction_components
    return cls(tuple(b.name for b in bindings),
               tuple(b.component.prediction for b in bindings))

  def loss_weights(self, cfg):
    return {name: prediction.weight(cfg, name)
            for name, prediction in zip(self.names, self.predictions)}

  @property
  def nonnegative(self):
    return tuple(prediction.nonnegative for prediction in self.predictions)

  @property
  def log_fields(self):
    return tuple("wm_" + name + "_loss" for name in self.names)

  def metadata(self):
    return {"names": list(self.names), "nonnegative": list(self.nonnegative)}
