"""Tensor-only inputs shared by measured and imagined MDP components.

One context belongs to one evaluation, including a whole [time, batch] call.
Lazy quantities are cached only inside that call, never across optimizer steps.
Nothing here detaches a predicted state from the BPTT graph.
"""
from dataclasses import dataclass

from ..tools.rotations import quat_to_mat


@dataclass
class StepContext:
  mdp: object
  state: tuple
  frame: object
  action: object = None
  previous_action: object = None
  terrain_tolerance: object = None
  height_offset: object = None
  _bodies: object = None
  _reference_root: object = None
  _aligned_reference: object = None
  _reference_bodies: object = None

  @property
  def bodies(self):
    if self._bodies is None:
      self._bodies = self.mdp.kin.forward(*self.state)
    return self._bodies

  @property
  def reference_root(self):
    if self._reference_root is None:
      self._reference_root = self.mdp.reference_root(self.frame)
    return self._reference_root

  @property
  def aligned_reference(self):
    if self._aligned_reference is None:
      self._aligned_reference = self.mdp.aligned_reference(self.state, self.frame)
    return self._aligned_reference

  @property
  def reference_bodies(self):
    m, f, ids = self.mdp.motion, self.frame, self.mdp.body_indexes
    if self._reference_bodies is None:
      self._reference_bodies = (m.body_pos_w[f][..., ids, :], quat_to_mat(m.body_quat_w[f][..., ids, :]),
                               m.body_lin_vel_w[f][..., ids, :], m.body_ang_vel_w[f][..., ids, :])
    return self._reference_bodies
