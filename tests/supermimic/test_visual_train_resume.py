import csv
import pytest
import torch
from supermimic.config.g1.rl_cfg import SuperMimicConfig, TerrainSuperMimicConfig
from supermimic.rl.runner import SuperMimicRunner, PolicyController
from supermimic.playback.evaluation import load_policy, evaluate, read_checkpoint_config
from supermimic.perception.history import select_history
from test_train_resume_eval import TensorEnv
from test_visual_algorithm import visual_config
from test_algorithm import small_config


class VisualTensorEnv(TensorEnv):
  """Real learner/controller; deterministic depth stands in for GPU raycasting."""
  depth_enabled=True
  def __init__(self,mdp):
    self.depth_image=torch.zeros(2,18,32,dtype=torch.uint8)
    super().__init__(mdp)

  def frames(self,frame):
    return (frame*3).clamp(0,255).to(torch.uint8)[:,None,None].expand(-1,18,32).clone()

  def reset(self,env_ids=None,**kwargs):
    result=super().reset(env_ids)
    ids=slice(None) if env_ids is None else env_ids
    self.depth_image[ids]=self.frames(self.command.time_steps-1)[ids]
    return result

  def step(self,action):
    previous=self.depth_image.clone()
    result=super().step(action)
    self.depth_image=self.frames(self.last_transition['f'])
    self.last_transition.update(depth=previous,depth_next=self.depth_image.clone())
    return result


def test_visual_train_resume_and_evaluate(mdp,tmp_path):
  torch.set_num_threads(2)
  cfg=visual_config(iterations=2,warmup_steps=4,wm_only_iters=1,ckpt_every=1,log_every=1,
                    wm_perception_lr=3e-4,latent_loss_weight=.3)
  runner=SuperMimicRunner(VisualTensorEnv(mdp),cfg,tmp_path/'run')
  runner.learn()
  with (tmp_path/'run/train_log.csv').open() as f:rows=list(csv.DictReader(f))
  assert all(float(row['wm_depth_loss'])>0 and float(row['wm_latent_loss'])>0 for row in rows)
  checkpoint=tmp_path/'run/last.pt'
  saved=read_checkpoint_config(checkpoint)
  assert type(saved) is TerrainSuperMimicConfig
  assert saved.depth_enabled and saved.wm_perception_lr==3e-4 and saved.latent_loss_weight==.3
  weights={k:v.clone() for k,v in runner.algorithm.policy.state_dict().items()}
  cfg.iterations=3
  resumed=SuperMimicRunner(VisualTensorEnv(mdp),cfg,tmp_path/'resume')
  resumed.load(checkpoint)
  assert resumed.buffer.size==0 and resumed.buffer.depth.latest==-1
  for k,v in weights.items():torch.testing.assert_close(v,resumed.algorithm.policy.state_dict()[k])
  assert resumed.algorithm.wm.depth_encoder is resumed.algorithm.policy.depth_encoder is resumed.algorithm.critic.depth_encoder
  resumed.learn()
  assert resumed.iteration==3 and resumed.buffer.size>=cfg.warmup_steps
  policy,loaded,_=load_policy(checkpoint,VisualTensorEnv(mdp))
  assert type(loaded) is TerrainSuperMimicConfig
  result=evaluate(VisualTensorEnv(mdp),policy,loaded)
  assert [r['steps'] for r in result['per_clip']]==[50,6]


def test_controller_repeated_calls_and_partial_reset_keep_visual_history(mdp):
  env=VisualTensorEnv(mdp);cfg=visual_config(depth_frames=8,depth_skip=5)
  windows=[]
  def policy(history,goal,depth):
    windows.append(depth.clone());return torch.zeros(2,29)
  controller=PolicyController(env,policy,cfg)
  for _ in range(8):env.step(controller())
  controller();before=controller.depth_history.data.clone()
  controller()
  torch.testing.assert_close(controller.depth_history.data,before)
  env.reset(env_ids=torch.tensor([0]));controller()
  assert controller.depth_history.data[0].eq(0).all()
  torch.testing.assert_close(controller.depth_history.data[1],before[1])
  torch.testing.assert_close(windows[-1],select_history(controller.depth_history.data))


def test_nonvisual_checkpoint_restores_explicit_depth_setting(mdp,tmp_path):
  cfg=small_config()
  runner=SuperMimicRunner(TensorEnv(mdp),cfg,tmp_path/'run')
  runner.save(tmp_path/'nonvisual.pt')
  loaded = read_checkpoint_config(tmp_path/'nonvisual.pt')
  assert type(loaded) is SuperMimicConfig and loaded.depth_enabled is False
  assert not hasattr(loaded, 'perception_class')
  load_policy(tmp_path/'nonvisual.pt',TensorEnv(mdp))
  with pytest.raises(ValueError, match="task's depth sensor"):
    load_policy(tmp_path/'nonvisual.pt',VisualTensorEnv(mdp))
  restored=SuperMimicRunner(TensorEnv(mdp),cfg,tmp_path/'restored')
  restored.load(tmp_path/'nonvisual.pt')
  assert restored.algorithm.set_lr(0)==cfg.lr
  # Keep the task identical to isolate the depth-structure mismatch on resume.
  visual=SuperMimicRunner(VisualTensorEnv(mdp),visual_config(task_id=cfg.task_id,wm_batch=cfg.wm_batch,pi_batch=cfg.pi_batch),tmp_path/'visual')
  with pytest.raises(ValueError,match='depth_enabled'):
    visual.load(tmp_path/'nonvisual.pt')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires mjlab 1.6 on Linux NVIDIA GPU')
def test_real_visual_environment_training_and_partial_reset(mdp,tmp_path):
  import sys
  from pathlib import Path
  sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
  from src.tasks.supermimic.config import make_env
  from src.tasks.supermimic.rl.runner import SuperMimicRunner as Runner
  cfg=visual_config(num_envs=4,wm_window=8,warmup_steps=16,iterations=2,wm_only_iters=1,
                    ckpt_every=1,log_every=1)
  env=make_env(mdp.motion.motion_files,num_envs=4)
  try:
    env.reset()
    assert env.depth_image.shape==(4,18,32) and env.depth_image.dtype==torch.uint8
    before=env.depth_image.clone()
    env.step(torch.zeros(4,29,device=env.device))
    torch.testing.assert_close(env.last_transition['depth'],before)
    terminal=env.last_transition['depth_next'].clone()
    env.reset(env_ids=torch.tensor([0],device=env.device))
    torch.testing.assert_close(env.depth_image[1:],terminal[1:])
    torch.testing.assert_close(env.last_transition['depth_next'],terminal)
    runner=Runner(env,cfg,tmp_path/'gpu');runner.learn()
    assert runner.iteration==2
  finally:env.close()


def test_old_visual_checkpoint_configuration_is_not_migrated(mdp,tmp_path):
  cfg=visual_config()
  runner=SuperMimicRunner(VisualTensorEnv(mdp),cfg,tmp_path/'run')
  path=tmp_path/'old_visual.pt'
  runner.save(path)
  ck=torch.load(path,weights_only=False)
  ck['format_version']=2
  ck['config']['actor_perception_lr']=1e-4
  ck['config']['critic_perception_lr']=1e-4
  ck['config'].pop('latent_loss_weight')
  torch.save(ck,path)
  with pytest.raises(ValueError,match='Legacy visual checkpoint'):
    runner.load(path)
  with pytest.raises(TypeError,match='actor_perception_lr'):
    read_checkpoint_config(path)
  with pytest.raises(TypeError,match='actor_perception_lr'):
    load_policy(path,VisualTensorEnv(mdp))
