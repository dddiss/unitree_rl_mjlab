"""One uint8 current frame per transition, with sparse pre-reset next frames.

Continuing transitions share their successor's current frame. Terminal frames
use a recyclable pool; the latest row uses a pending successor until the next
row arrives. No generated image is ever written to this replay store.
"""
import torch


class DepthReplay:
  def __init__(self, capacity, num_envs, device):
    self.capacity, self.num_envs, self.device = capacity, num_envs, device
    self.frames = torch.empty(capacity, num_envs, 18, 32, dtype=torch.uint8, device=device)
    self.terminal_index = torch.zeros(capacity, num_envs, dtype=torch.long, device=device)
    self.pool = torch.zeros(1, 18, 32, dtype=torch.uint8, device=device)  # index 0 is a dummy
    self.free = []
    self.row_slots = [[] for _ in range(capacity)]
    self.latest = -1
    self.pending = torch.empty(num_envs, 18, 32, dtype=torch.uint8, device=device)

  @torch.no_grad()
  def add(self, row, depth, next_depth, done):
    if depth.dtype != torch.uint8 or next_depth.dtype != torch.uint8:
      raise TypeError("Real depth replay expects preprocessed uint8 frames")
    self.free.extend(self.row_slots[row])
    self.row_slots[row] = []
    self.terminal_index[row].zero_()
    self.frames[row].copy_(depth)
    self.pending.copy_(next_depth)
    self.latest = row
    ids = done.nonzero(as_tuple=False).flatten()
    count = ids.numel()
    if len(self.free) < count:
      old_size = len(self.pool)
      growth = max(self.num_envs, count - len(self.free), old_size)
      self.pool = torch.cat((self.pool, torch.empty(growth, 18, 32, dtype=torch.uint8,
                                                   device=self.device)), 0)
      self.free.extend(range(old_size, old_size + growth))
    if count:
      slots = self.free[-count:]
      del self.free[-count:]
      index = torch.tensor(slots, device=self.device)
      self.pool[index] = next_depth[ids]
      self.terminal_index[row, ids] = index
      self.row_slots[row] = slots

  def next_frames(self, rows, envs):
    index = self.terminal_index[rows, envs]
    frames = torch.where((index > 0)[..., None, None], self.pool[index],
                         self.frames[(rows + 1) % self.capacity, envs])
    return torch.where((rows == self.latest)[..., None, None], self.pending[envs], frames)

  def history(self, replay, ages, envs, length):
    """Dense history, padded with the earliest same-episode retained frame."""
    rows = replay._row(ages)
    steps = replay.data["ep_step"][rows, envs]
    back = torch.arange(length - 1, -1, -1, device=self.device)
    back = torch.minimum(back, torch.minimum(steps, ages)[..., None])
    return self.frames[replay._row(ages[..., None] - back), envs[..., None]]
