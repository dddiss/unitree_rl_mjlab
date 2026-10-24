"""Differentiable FK from the actual compiled robot, including fixed bodies.

Adapted from supermimic_vibed; see THIRD_PARTY_NOTICES.md.
Joint inputs follow the supplied model joint IDs, not a hardcoded SDK order.
"""
import numpy as np
import torch
import mujoco

from .rotations import axis_angle_to_mat, quat_to_mat


class G1Kinematics:
  def __init__(self, model, joint_ids, tracked_body_ids, root_body_id, device="cpu"):
    joint_ids = [int(j) for j in joint_ids]
    tracked_body_ids = [int(b) for b in tracked_body_ids]
    joint_map = {j: i for i, j in enumerate(joint_ids)}
    bodies = [int(root_body_id)]
    for b in range(int(root_body_id) + 1, model.nbody):
      if int(model.body_parentid[b]) in bodies:
        bodies.append(b)
    local = {b: i for i, b in enumerate(bodies)}
    self.parent, self.joint_idx = [], []
    axes, anchors = [], []
    for b in bodies:
      self.parent.append(-1 if b == root_body_id else local[int(model.body_parentid[b])])
      if b == root_body_id or model.body_jntnum[b] == 0:
        self.joint_idx.append(-1)
        axes.append(np.zeros(3)); anchors.append(np.zeros(3))
      else:
        j = int(model.body_jntadr[b])
        if model.body_jntnum[b] != 1 or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
          raise ValueError("G1 FK expects one hinge per moving non-root body")
        self.joint_idx.append(joint_map[j])
        axes.append(model.jnt_axis[j]); anchors.append(model.jnt_pos[j])
    tensor = lambda x: torch.as_tensor(np.asarray(x).copy(), dtype=torch.float32, device=device)
    self.axes, self.anchors = tensor(axes), tensor(anchors)
    self.off_p = tensor(model.body_pos[bodies])
    self.off_r = quat_to_mat(tensor(model.body_quat[bodies]))
    self.tracked_idx = [local[b] for b in tracked_body_ids]
    self.q_lo, self.q_hi = tensor(model.jnt_range[joint_ids, 0]), tensor(model.jnt_range[joint_ids, 1])

  def forward(self, p, r, v, w, q, qd):
    ps, rs, vs, ws = [], [], [], []
    for i, parent in enumerate(self.parent):
      if parent < 0:
        pi, ri, vi, wi = p, r, v, w
      else:
        rp = rs[parent]
        origin = ps[parent] + (rp @ self.off_p[i, :, None]).squeeze(-1)
        base_r = rp @ self.off_r[i]
        j = self.joint_idx[i]
        if j < 0:
          pi, ri, wi = origin, base_r, ws[parent]
          vi = vs[parent] + torch.cross(ws[parent], pi - ps[parent], dim=-1)
        else:
          ri = base_r @ axis_angle_to_mat(self.axes[i], q[..., j])
          pivot = origin + (base_r @ self.anchors[i, :, None]).squeeze(-1)
          pi = pivot - (ri @ self.anchors[i, :, None]).squeeze(-1)
          joint_w = (base_r @ self.axes[i, :, None]).squeeze(-1) * qd[..., j:j+1]
          wi = ws[parent] + joint_w
          vi = (vs[parent] + torch.cross(ws[parent], pi - ps[parent], dim=-1)
                + torch.cross(joint_w, pi - pivot, dim=-1))
      ps.append(pi); rs.append(ri); vs.append(vi); ws.append(wi)
    ids = self.tracked_idx
    return (torch.stack([ps[i] for i in ids], -2), torch.stack([rs[i] for i in ids], -3),
            torch.stack([vs[i] for i in ids], -2), torch.stack([ws[i] for i in ids], -2))
