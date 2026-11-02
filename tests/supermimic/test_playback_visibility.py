"""mjlab 1.6 viewer adapters set display flags before Viser creates its GUI."""
from contextlib import nullcontext
import runpy
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np

from conftest import ROOT


def test_native_and_viser_geometry_and_checkbox_defaults(monkeypatch):
  module = ModuleType('mjlab.viewer')
  mask = (1, 1, 1, 0, 1, 1)
  # Geometry and sensor selection must stay unchanged: only display flags change.
  model_groups = np.array([1, 4, 5])
  depth_groups = (0, 2, 5)

  class Base:
    def __init__(self):
      self.cfg = NS(geom_group=mask)

  class Native(Base):
    def setup(self):
      self.viewer = NS(lock=nullcontext, opt=NS(geomgroup=np.array([1, 1, 1, 0, 0, 0])))
      self.vopt = NS(geomgroup=np.array([1, 1, 1, 0, 0, 0]))

  class Scene:
    def __init__(self):
      self.geom_groups_visible = [True, True, True, False, False, False]
      self._sync_visibilities()

    def _sync_visibilities(self):
      self.mesh_visible = [self.geom_groups_visible[g] for g in model_groups]

    def create_groups_gui(self):
      self.checkboxes = self.geom_groups_visible.copy()

  class Viser(Base):
    def setup(self):
      # Same lifecycle ordering as the pinned mjlab 1.6 implementation.
      self._scene = Scene()
      self._scene.create_groups_gui()

  module.NativeMujocoViewer, module.ViserPlayViewer = Native, Viser
  monkeypatch.setitem(sys.modules, module.__name__, module)
  classes = runpy.run_path(str(ROOT / 'src/tasks/supermimic/playback/playback_viewer.py'))
  native = classes['NativePlayViewer']()
  native.setup()
  assert tuple(native.viewer.opt.geomgroup) == mask
  assert tuple(native.vopt.geomgroup) == mask
  viser = classes['ViserPlayViewer']()
  viser.setup()
  assert tuple(viser._scene.geom_groups_visible) == mask
  assert tuple(viser._scene.checkboxes) == mask
  assert all(viser._scene.mesh_visible)
  # User toggles still work after setup; these defaults aren't forced every tick.
  viser._scene.geom_groups_visible[4] = False
  viser._scene._sync_visibilities()
  assert viser._scene.mesh_visible == [True, False, True]
  assert model_groups.tolist() == [1, 4, 5] and depth_groups == (0, 2, 5)
