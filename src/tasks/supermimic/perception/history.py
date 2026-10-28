"""Dense depth history, oldest first; sample P-BFM's strided eight-frame window."""
import torch


def history_length(frames=8, skip=5):
  return (frames - 1) * skip + 1


def select_history(history, frames=8, skip=5):
  length = history_length(frames, skip)
  if history.shape[-3] < length:
    raise ValueError("Depth history does not cover all requested frame ages")
  return history[..., -length::skip, :, :]


def append_depth(history, frame):
  """Out-of-place append: preserve the graph through every predicted frame."""
  return torch.cat((history[:, 1:], frame[:, None]), dim=1)


def window_from_frames(history, frames=8, skip=5):
  """Stack only selected frames; a tuple queue avoids 32 copies of dense history."""
  return torch.stack(history[-history_length(frames, skip)::skip], dim=1)


class DepthHistory:
  def __init__(self, num_envs, length, device):
    self.data = torch.zeros(num_envs, length, 18, 32, dtype=torch.uint8, device=device)

  @torch.no_grad()
  def update(self, frame, fresh, advanced):
    rolled = append_depth(self.data, frame)
    self.data = torch.where(fresh[:, None, None, None], frame[:, None],
                            torch.where(advanced[:, None, None, None], rolled, self.data))
    return self.data
