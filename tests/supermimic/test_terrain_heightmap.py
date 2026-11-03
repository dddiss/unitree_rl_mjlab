"""Startup rasterization, native miss repair, and ray-free runtime lookups."""
import ast
from contextlib import nullcontext
import sys
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import pytest
import torch

from supermimic.terrain import heightmap as module
from supermimic.terrain.heightmap import TerrainHeightMap, mark_terrain_padding_ray_group
from test_env_lifecycle import ROOT


def step_model():
  return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body name="terrain">
      <geom type="box" group="5" pos="0 0 -.05" size="1 1 .05"/>
      <geom type="box" group="5" pos=".5 0 .15" size=".5 1 .15"/>
    </body>
    <body name="robot" pos="0 0 2"><freejoint/>
      <geom type="box" group="4" size=".3 .3 .3"/>
    </body>
  </worldbody></mujoco>''')


def test_startup_map_has_progress_and_preserves_state_rng_and_step_discontinuity(monkeypatch, capsys):
  model = step_model()
  original = model.geom_size.copy()
  rng = torch.random.get_rng_state()
  nrng = np.random.get_state()
  ground = TerrainHeightMap.from_model(model, 'cpu', resolution=.025, chunk_size=1000)
  output = capsys.readouterr()
  assert 'Terrain height map' in output.err and '100%' in output.err
  assert 'CPU rechecked 0 points' in output.out
  assert ground.heights.dtype == torch.float32 and ground.heights.shape == (80, 80)
  assert torch.equal(rng, torch.random.get_rng_state())
  np.testing.assert_array_equal(nrng[1], np.random.get_state()[1])
  np.testing.assert_array_equal(original, model.geom_size)
  def forbidden(*args, **kw):
    raise AssertionError('Runtime lookup must not cast rays')
  monkeypatch.setattr(mujoco, 'mj_ray', forbidden)
  # No interpolated slope at the step. The robot is excluded during rasterization.
  xy = torch.tensor([[[-.0001, 0.], [.0001, 0.]], [[-.3, .9], [.3, -.9]]], requires_grad=True)
  heights = ground.height(xy)
  torch.testing.assert_close(heights, torch.tensor([[0., .3], [0., .3]]), atol=1e-6, rtol=0)
  assert not heights.requires_grad
  assert ground.height(torch.empty(0, 2)).shape == (0,)


def test_only_missed_cells_are_cpu_rechecked_at_original_double_coordinates(monkeypatch, capsys):
  model = step_model()
  model.body_pos[model.body('terrain').id, :2] = [-20., 5.6]
  native = module._native_heights
  calls = []
  def miss_once(model, data, xy, ray_z):
    calls.append(xy.copy())
    assert xy.dtype == np.float64
    heights = native(model, data, xy, ray_z)
    if len(calls) == 1:
      heights[[0, 50, 99]] = np.nan
    return heights
  monkeypatch.setattr(module, '_native_heights', miss_once)
  ground = TerrainHeightMap.from_model(model, 'cpu', resolution=.2)
  assert [len(c) for c in calls] == [100, 3]
  np.testing.assert_array_equal(calls[1], calls[0][[0, 50, 99]])
  assert ground.cpu_rechecks == 3 and torch.isfinite(ground.heights).all()
  assert 'CPU MuJoCo recheck' in capsys.readouterr().err
  expected = native(model, mujoco_data(model), calls[0], .3 + 1.5)
  np.testing.assert_allclose(ground.heights.numpy().reshape(-1), expected, atol=1e-6)


def mujoco_data(model):
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return data


def test_cpu_unresolved_holes_fail_before_training_instead_of_filling_zero():
  model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <geom type="box" group="5" pos="-1 0 -.1" size=".3 1 .1"/>
    <geom type="box" group="5" pos="1 0 -.1" size=".3 1 .1"/>
  </worldbody></mujoco>''')
  with pytest.raises(RuntimeError, match='invalid cells after CPU MuJoCo recheck.*xy_w='):
    TerrainHeightMap.from_model(model, 'cpu', resolution=.2)


def test_warp_raster_adapter_uses_isolated_context_and_ground_filter(monkeypatch):
  """Run the production adapter with native CPU rays in place of the CUDA kernel."""
  model = step_model()
  data = mujoco_data(model)
  calls = []
  context = object()
  def put_data(m, d, **kwargs):
    assert m is model and d is data
    assert kwargs['nworld'] == 1 and kwargs['nconmax'] >= data.ncon
    assert kwargs['njmax'] >= data.nefc
    return data
  def create_context(m, **kwargs):
    assert m is model and kwargs['enabled_geom_groups'] == [5]
    assert kwargs['nworld'] == 1 and not kwargs['use_textures']
    return context
  def rays(m, d, origins, directions, groups, static, exclude, distances, geom_ids, normals, **kwargs):
    assert m is model and d is data and kwargs['rc'] is context
    assert groups == (0, 0, 0, 0, 0, -1) and static
    assert exclude.eq(-1).all() and directions[..., 2].eq(-1).all()
    assert origins.shape[0] == 1
    calls.append(origins.shape[1])
    xy = origins[0, :, :2].double().numpy()
    heights = module._native_heights(model, data, xy, float(origins[0, 0, 2]))
    distances[0] = origins[0, :, 2] - torch.from_numpy(heights)
    distances[~torch.isfinite(distances)] = -1
  monkeypatch.setitem(sys.modules, 'mujoco_warp', NS(put_model=lambda m: m, put_data=put_data,
                                                   create_render_context=create_context, rays=rays))
  monkeypatch.setitem(sys.modules, 'warp', NS(
    vec3=object(), from_torch=lambda t, **kw: t, stream_from_torch=lambda stream: stream,
    ScopedDevice=lambda device: nullcontext(), ScopedStream=lambda stream: nullcontext(),
    types=NS(vector=lambda **kw: lambda *values: values)))
  monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: None)
  raster = module._WarpRasterizer(model, data, torch.device('cpu'))
  values = raster(np.array([[-.5, 0.], [.5, 0.], [10., 0.]]), 1.8)
  np.testing.assert_allclose(values[:2], [0., .3], atol=1e-6)
  assert np.isnan(values[2]) and calls == [3]


def test_runtime_bounds_invalid_inputs_and_negative_height():
  ground = TerrainHeightMap(torch.full((3, 4), -1.7), (-2., -3.), (2., 3.))
  heights, valid = ground.sample(torch.tensor([[-2., -3.], [2., 3.], [2.001, 0.],
                                             [float('nan'), 0.], [0., float('inf')]]))
  assert valid.tolist() == [True, True, False, False, False]
  torch.testing.assert_close(heights[:2], torch.tensor([-1.7, -1.7]))
  assert heights[2:].isnan().all()
  with pytest.raises(RuntimeError, match=r'Reference XY is outside.*flat_indices=\[1\]'):
    ground.height(torch.tensor([[0., 0.], [9., 0.]]))


def test_reported_box_seams_are_valid_lookups():
  # Nonoverlapping boxes with the user's float32 seam geometry; runtime contains
  # no intersection arithmetic, including at exactly representable AABB faces.
  model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <geom type="box" group="5" pos="-22.65 5.45 -.15" size=".15 .15 .15"/>
    <geom type="box" group="5" pos="-22.65 5.75 -.15" size=".15 .15 .15"/>
  </worldbody></mujoco>''')
  ground = TerrainHeightMap.from_model(model, 'cpu', resolution=.025)
  xy = torch.tensor([[-22.799999237060547, 5.6], [-22.65, 5.6], [-22.65, 5.6000001]])
  torch.testing.assert_close(ground.height(xy), torch.zeros(3), atol=1e-6, rtol=0)


def test_scene_hook_keeps_padding_visible_without_modifying_robot():
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name='terrain')
  for x, y in ((-3, 0), (3, 0), (0, -3), (0, 3)):
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(1, 1, .1), pos=(x, y, -.1), group=0)
  robot = spec.worldbody.add_body(name='robot')
  robot.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(.1, 0, 0), group=0)
  mark_terrain_padding_ray_group(spec)
  assert all(g.group == 5 for g in body.geoms) and robot.geoms[0].group == 0


@pytest.mark.parametrize('terrain', ['plane', 'generator'])
def test_map_built_before_manager_probes(terrain):
  tree = ast.parse((ROOT/'src/tasks/supermimic/env.py').read_text())
  cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SuperMimicEnv')
  cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'load_managers']
  events = []
  class Base:
    def load_managers(self):
      events.append('managers')
      assert (self.terrain_height_map is not None) == (terrain == 'generator')
  class Map:
    @classmethod
    def from_model(cls, model, device, resolution):
      events.append(('map', resolution))
      return cls()
  ns = dict(ManagerBasedRlEnv=Base, TerrainHeightMap=Map)
  exec(compile(ast.Module(body=[cls], type_ignores=[]), '<environment>', 'exec'), ns)
  env = ns['SuperMimicEnv']()
  env.scene = NS(terrain=NS(cfg=NS(terrain_type=terrain)))
  env.cfg = NS(commands={'motion': NS(terrain_reset=NS(heightmap_resolution=.025))})
  env.sim, env.device = NS(mj_model=object()), 'cpu'
  env.load_managers()
  assert events == ([('map', .025), 'managers'] if terrain == 'generator' else ['managers'])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires NVIDIA GPU and mujoco-warp')
def test_cuda_raster_and_cpu_repair_match_native_and_leave_model_untouched(monkeypatch):
  model = step_model()
  original = model.geom_size.copy()
  raster = module._WarpRasterizer.__call__
  def force_miss(self, xy, ray_z):
    values = raster(self, xy, ray_z)
    values[0] = np.nan
    return values
  monkeypatch.setattr(module._WarpRasterizer, '__call__', force_miss)
  ground = TerrainHeightMap.from_model(model, 'cuda:0', resolution=.05)
  cpu = TerrainHeightMap.from_model(model, 'cpu', resolution=.05)
  torch.testing.assert_close(ground.heights.cpu(), cpu.heights, atol=2e-6, rtol=1e-4)
  assert ground.cpu_rechecks >= 1
  np.testing.assert_array_equal(model.geom_size, original)
  xy = torch.tensor([[[-.001, 0.], [.001, 0.]], [[-.95, .95], [.95, -.95]]])
  torch.testing.assert_close(ground.height(xy.cuda()).cpu(), cpu.height(xy), atol=2e-6, rtol=1e-4)


@pytest.mark.parametrize('max_cells', [3, 100000])
def test_rect_max_covers_interior_and_matches_brute_force(max_cells):
  gen = torch.Generator().manual_seed(13)
  grid = torch.rand((19, 17), generator=gen) - 1.
  ground = TerrainHeightMap(grid, (-4., -3.), (3., 8.))
  lower = torch.rand((2, 3, 2), generator=gen) * torch.tensor([3., 5.]) + ground.xy_min
  upper = lower + torch.rand((2, 3, 2), generator=gen) * torch.tensor([4., 6.])
  upper = torch.minimum(upper, ground.xy_max)
  before = grid.clone()
  rng = torch.random.get_rng_state()
  actual, valid = ground.max_in_rect(lower, upper, max_cells=max_cells)
  assert valid.all()
  for n in range(2):
    for k in range(3):
      lo = ((lower[n, k] - ground.xy_min) / ground.cell_size).floor().long()
      hi = ((upper[n, k] - ground.xy_min) / ground.cell_size).floor().long()
      hi = torch.minimum(hi, torch.tensor(grid.shape) - 1)
      expected = grid[lo[0]:hi[0]+1, lo[1]:hi[1]+1].max()
      assert actual[n, k] == expected
  torch.testing.assert_close(grid, before)
  assert torch.equal(rng, torch.random.get_rng_state())


def test_rect_max_interior_peak_and_closed_boundary():
  ground = TerrainHeightMap(torch.full((10, 10), -2.), (0., 0.), (10., 10.))
  ground.heights[3, 4] = 1.7  # Not a corner or the rectangle centre.
  ground.heights[8, 8] = 2.7
  actual, valid = ground.max_in_rect(torch.tensor([[1.1, 1.1], [1.1, 1.1]]),
                                     torch.tensor([[7.9, 7.9], [8., 8.]]))
  assert valid.all()
  torch.testing.assert_close(actual, torch.tensor([1.7, 2.7]))
  assert ground.max_in_rect(torch.tensor([.1, .1]), torch.tensor([.1, .1]))[0] == -2.
  assert ground.max_in_rect(torch.tensor([10., 10.]), torch.tensor([10., 10.]))[1]


def test_rect_max_rejects_outside_bounds_bad_cells_and_reversed_rectangles():
  ground = TerrainHeightMap(torch.zeros(4, 4), (0., 0.), (4., 4.))
  ground.heights[1, 1] = float('nan')
  lo = torch.tensor([[0., 0.], [-.1, 0.], [3., 3.], [float('nan'), 0.], [3., 3.], [2., 2.]])
  hi = torch.tensor([[2., 2.], [0., 0.], [4.1, 4.], [1., 1.], [2., 2.], [4., 4.]])
  heights, valid = ground.max_in_rect(lo, hi, max_cells=3)
  assert valid.tolist() == [False, False, False, False, False, True]
  assert heights[:5].isnan().all() and heights[-1] == 0.
  empty, valid = ground.max_in_rect(torch.empty(0, 3, 2), torch.empty(0, 3, 2))
  assert empty.shape == valid.shape == (0, 3)
