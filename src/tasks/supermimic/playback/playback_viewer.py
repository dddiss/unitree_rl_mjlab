"""Apply play's geom visibility to mjlab 1.6's interactive viewers.

The offscreen renderer already reads ViewerConfig.geom_group. The interactive
viewers do not, so apply it to their display options without changing model
geom groups (which would change depth-camera ray filtering).
"""
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer as _ViserPlayViewer


class NativePlayViewer(NativeMujocoViewer):
  def setup(self):
    super().setup()
    with self.viewer.lock():
      self.viewer.opt.geomgroup[:] = self.cfg.geom_group
    self.vopt.geomgroup[:] = self.cfg.geom_group


class ViserPlayViewer(_ViserPlayViewer):
  @property
  def _scene(self):
    return self._playback_scene

  @_scene.setter
  def _scene(self, scene):
    # mjlab 1.6 assigns _scene before creating the Groups checkboxes. Apply the
    # defaults here so both meshes and checkbox values start in the same state.
    # A post-setup change alone would leave the GUI showing G4/G5 as unchecked.
    self._playback_scene = scene
    scene.geom_groups_visible[:] = [bool(value) for value in self.cfg.geom_group]
    scene._sync_visibilities()
