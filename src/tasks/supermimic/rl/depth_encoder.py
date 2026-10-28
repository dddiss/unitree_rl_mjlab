# Adapted from RoboParty/P-BFM, CC BY-NC 4.0; see THIRD_PARTY_NOTICES.md.
"""G1 CNN s1 -> frame256 -> GRU256. Each owner constructs its own encoder."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..perception.preprocessing import depth_float


class DepthEncoder(nn.Module):
  def __init__(self, latent_dim=256, frame_dim=256, checkpoint_activations=True, batch_size=256):
    super().__init__()
    self.checkpoint_activations = checkpoint_activations
    self.batch_size = batch_size
    self.frame_cnn = nn.Sequential(
      nn.Conv2d(1, 16, 3, stride=1, padding=1), nn.ReLU(),
      nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
      nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
      nn.AdaptiveAvgPool2d((2, 4)),
    )
    self.frame_projection = nn.Sequential(nn.Flatten(), nn.Linear(512, frame_dim), nn.ReLU())
    self.gru = nn.GRU(frame_dim, latent_dim, batch_first=True)

  def _encode(self, depth):
    # cuDNN RNN inference cannot backpropagate to inputs when the critic is
    # frozen/eval. Native GRU also permits gradients through frozen parameters.
    shape = depth.shape
    x = depth_float(depth).reshape(-1, 1, 18, 32)
    x = self.frame_projection(self.frame_cnn(x))
    with torch.backends.cudnn.flags(enabled=False):
      _, hidden = self.gru(x.reshape(-1, shape[-3], x.shape[-1]))
      return hidden[-1].reshape(*shape[:-3], -1)

  def forward(self, depth):
    if depth.shape[-2:] != (18, 32):
      raise ValueError("DepthEncoder expects [..., frames, 18, 32]")
    leading = depth.shape[:-3]
    depth = depth.reshape(-1, *depth.shape[-3:])
    results = []
    for chunk in depth.split(self.batch_size):
      if self.checkpoint_activations and torch.is_grad_enabled():
        results.append(checkpoint(self._encode, chunk, use_reentrant=False))
      else:
        results.append(self._encode(chunk))
    return torch.cat(results, 0).reshape(*leading, -1)
