import numpy as np
import torch
import mujoco
from pathlib import Path
from supermimic.perception.camera import DepthCameraCfg, DepthPatternCfg, exclude_camera_housing
from supermimic.perception.preprocessing import preprocess_depth, depth_float


def test_pbfm_maxcrop_blur_quantization():
  # Independent numpy reference for the pinned G1 profile (sigma=1, reflect pad).
  raw = np.linspace(-.2, 3., 36*64, dtype=np.float32).reshape(36, 64)
  raw[20, 20] = np.nan
  raw[25, 25] = np.inf
  clean = np.where(np.isfinite(raw) & (raw >= 0), raw, 2.5).clip(0, 2.5)[18:, 16:-16]
  padded = np.pad(clean, 1, mode='reflect')
  g = np.exp(-np.arange(-1, 2, dtype=np.float32)**2/2); g /= g.sum()
  expected = sum(g[y]*g[x]*padded[y:y+18, x:x+32] for y in range(3) for x in range(3))
  expected = np.round(expected/2.5*255).astype(np.uint8)
  actual = preprocess_depth(torch.from_numpy(raw)[None])
  assert actual.shape == (1, 18, 32) and actual.dtype == torch.uint8
  np.testing.assert_array_equal(actual[0].numpy(), expected)
  assert preprocess_depth(torch.full((1,36,64),float('nan'))).eq(255).all()


def test_depth_float_preserves_predicted_image_gradients():
  image = torch.rand(2,18,32,requires_grad=True)
  depth_float(image).sum().backward()
  torch.testing.assert_close(image.grad, torch.ones_like(image))
  torch.testing.assert_close(depth_float(torch.tensor([0,255],dtype=torch.uint8)), torch.tensor([0.,1.]))


def test_camera_optical_z_and_mount():
  cfg = DepthCameraCfg()
  rays = cfg.optical_rays()
  torch.testing.assert_close(rays.norm(dim=-1), torch.ones(36,64))
  # A fronto-parallel plane is constant optical Z, not constant ray distance.
  distance = 1.2 / rays[..., 2]
  torch.testing.assert_close(distance * rays[..., 2], torch.full((36,64),1.2))
  assert distance.max() > 1.2
  origins, directions = DepthPatternCfg(cfg).generate_rays(None,'cpu')
  torch.testing.assert_close(origins, torch.tensor(cfg.mount_pos).expand(2304,3))
  torch.testing.assert_close(directions.norm(dim=-1), torch.ones(2304))
  assert directions[:,0].mean() > 0 and directions[:,2].mean() < 0


def test_g1_camera_sees_scene_without_changing_collisions():
  root=Path(__file__).resolve().parents[2]
  spec=mujoco.MjSpec.from_file(str(root/'src/assets/robots/unitree_g1/xmls/g1.xml'))
  spec.worldbody.add_geom(name='depth_test_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,
                          size=[10,10,.1],group=5)
  original=spec.compile()
  cfg=DepthCameraCfg()
  exclude_camera_housing(spec,cfg.excluded_housing_meshes)
  model=spec.compile()
  for field in ('geom_contype','geom_conaffinity','geom_condim','geom_friction',
                'body_mass','body_inertia'):
    np.testing.assert_array_equal(getattr(model,field),getattr(original,field))
  changed=np.flatnonzero(model.geom_group!=original.geom_group)
  assert len(changed)>0
  assert all(model.mesh(model.geom_dataid[g]).name=='head_link' for g in changed)
  d=mujoco.MjData(model);d.qpos[2]=.8;mujoco.mj_forward(model,d)
  b=model.body('torso_link').id
  rotation=d.xmat[b].reshape(3,3)
  offsets,rays=DepthPatternCfg(cfg).generate_rays(None,'cpu')
  origins=offsets.numpy()@rotation.T+d.xpos[b]
  rays=rays.numpy()@rotation.T
  groups=np.array([int(i in cfg.include_geom_groups) for i in range(6)],dtype=np.uint8)
  hits=[]
  for origin,ray in zip(origins,rays):
    geom=np.array([-1],dtype=np.int32)
    mujoco.mj_ray(model,d,origin,ray,groups,1,-1,geom)
    hits.append(geom[0])
  hits=np.array(hits)
  assert not np.isin(hits,changed).any()
  assert (hits==model.geom('depth_test_floor').id).sum()>len(hits)//2
