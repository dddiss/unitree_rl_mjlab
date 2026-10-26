from pathlib import Path

import pytest
import torch
from supermimic.mdp.motion_lib import MotionLibrary, _resolve_motion_files
from supermimic.mdp.rotations import mat_to_6d,mat_to_quat,so3_exp


@pytest.mark.parametrize('as_string', [False, True])
def test_directory_loads_root_and_nested_motions_in_path_order(mdp, tmp_path, as_string):
  original = mdp.motion
  nested = tmp_path / 'nested'
  deep = nested / 'deeper'
  deep.mkdir(parents=True)
  root_clip = tmp_path / 'z_root.npz'
  nested_clip = nested / 'a.npz'
  deep_clip = deep / 'motion1.npz'
  Path(original.motion_files[0]).rename(root_clip)
  Path(original.motion_files[1]).rename(deep_clip)
  nested_clip.write_bytes(deep_clip.read_bytes())
  (deep / 'ignored.csv').write_text('not a motion archive')

  motion = MotionLibrary(str(tmp_path) if as_string else tmp_path)
  assert motion.motion_files == tuple(str(p.resolve()) for p in
                                      (nested_clip, deep_clip, root_clip))
  assert motion.motion_names == ('a', 'motion1', 'z_root')
  assert motion.clip_len.tolist() == [7, 7, 51]
  torch.testing.assert_close(motion.body_pos_w, torch.cat((
    original.body_pos_w[51:], original.body_pos_w[51:], original.body_pos_w[:51],
  )))


def test_file_and_explicit_list_keep_the_requested_selection(mdp):
  first, second = mdp.motion.motion_files
  assert _resolve_motion_files(first) == (first,)
  assert _resolve_motion_files(Path(second)) == (second,)
  assert _resolve_motion_files([second, first]) == (second, first)


def test_bins_and_goal_timing(mdp):
  m = mdp.motion
  assert m.num_bins == 2
  assert m.bin_stop.tolist() == [50,57]
  assert m.frame_bin[m.clip_end].tolist() == [-1,-1]
  # Distinct angles make a one-frame offset visible in the goal values.
  m.joint_pos[:]=.01*torch.arange(m.num_frames)[:,None]
  f = torch.tensor([1,50,52,57])
  k=f-1
  state=(*mdp.reference_root(k),m.joint_pos[k],m.joint_vel[k])
  assert mdp.goal(state,f).shape == (4,640)
  # Reset state is k, first target is k+1=f; target encoding adds no extra step.
  torch.testing.assert_close(mdp.goal(state,f).reshape(4,10,64)[:,0,:29],m.joint_pos[k+1])
  assert m.future_frames(torch.tensor([49]),3).tolist() == [[50,50,50]]
  frames,_ = m.sample_start_frames(200)
  assert not torch.isin(frames,m.clip_end).any()


def test_goal_clamps_each_clip_and_keeps_robot_orientation_at_end(mdp):
  m=mdp.motion
  frames=torch.arange(m.num_frames,dtype=torch.float32)
  rotations=so3_exp(torch.stack((.1*frames.sin(),.1*frames.cos(),.03*frames),-1))
  m.body_quat_w[:,mdp.root_body_index]=mat_to_quat(rotations)
  m.joint_pos[:]=.001*frames[:,None]+.0001*torch.arange(29)
  m.joint_vel[:]=.01*frames[:,None]
  m.body_ang_vel_w[:,mdp.root_body_index]=torch.tensor([.3,.2,.1])
  # Mixed clips and phases: ordinary, just before the last target, and after it.
  executed=torch.tensor([5,49,50,56,57])
  next_f=torch.tensor([6,50,50,57,57])
  future=torch.tensor([[6,7,8],[50,50,50],[50,50,50],[57,57,57],[57,57,57]])
  p,_,v,w=mdp.reference_root(executed)
  robot_r=so3_exp(torch.tensor([[.2,-.3,.5]])).expand(5,-1,-1)
  state=(p,robot_r,v,w,m.joint_pos[executed],m.joint_vel[executed])
  actual=mdp.goal(state,next_f,3).reshape(5,3,64)
  _,reference_r,_,_=mdp.reference_root(future)
  expected=torch.cat((m.joint_pos[future],m.joint_vel[future],
                      mat_to_6d(robot_r[:,None].transpose(-1,-2)@reference_r)),-1)
  torch.testing.assert_close(actual,expected)
  terminal=torch.tensor([1,2,3,4])
  torch.testing.assert_close(actual[terminal],actual[terminal,:1].expand(-1,3,-1))
  # Repeating a reference does not imply that the robot has reached its pose.
  identity=mat_to_6d(torch.eye(3)).expand(4,3,6)
  assert not torch.allclose(actual[terminal,:,58:],identity)
