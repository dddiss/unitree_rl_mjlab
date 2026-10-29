# Adapted from RoboParty/P-BFM, CC BY-NC 4.0; see THIRD_PARTY_NOTICES.md.
"""G1 CNN s1 -> frame256 -> GRU256, with independently reusable frame features."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..perception.preprocessing import depth_float


class WindowGRU(nn.Module):
  """Single-layer, batch-first GRU expressed in compilable tensor operations.

  Parameter names, registration order, initialization and gate equations match
  nn.GRU. Keep the reset gate on the hidden projection (including its bias), as
  PyTorch does. Every history window still starts with a zero hidden state.
  """
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.input_size, self.hidden_size = input_size, hidden_size
    self.num_layers, self.batch_first = 1, True
    self.weight_ih_l0 = nn.Parameter(torch.empty(3 * hidden_size, input_size))
    self.weight_hh_l0 = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
    self.bias_ih_l0 = nn.Parameter(torch.empty(3 * hidden_size))
    self.bias_hh_l0 = nn.Parameter(torch.empty(3 * hidden_size))
    bound = 1 / math.sqrt(hidden_size)
    for parameter in self.parameters():
      nn.init.uniform_(parameter, -bound, bound)

  def forward(self, features):
    inputs = F.linear(features, self.weight_ih_l0, self.bias_ih_l0)
    hidden = features.new_zeros(features.shape[0], self.hidden_size)
    outputs = []
    for gates in inputs.unbind(1):
      ir, iz, inn = gates.chunk(3, -1)
      hr, hz, hn = F.linear(hidden, self.weight_hh_l0, self.bias_hh_l0).chunk(3, -1)
      reset, update = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
      new = torch.tanh(inn + reset * hn)
      hidden = new + update * (hidden - new)
      outputs.append(hidden)
    return torch.stack(outputs, 1), hidden.unsqueeze(0)


class DepthEncoder(nn.Module):
  def __init__(self, latent_dim=256, frame_dim=256, checkpoint_activations=True,
               batch_size=256, frames_per_window=8):
    super().__init__()
    self.checkpoint_activations = checkpoint_activations
    self.batch_size = batch_size
    self.frames_per_window = frames_per_window
    self.frame_cnn = nn.Sequential(
      nn.Conv2d(1, 16, 3, stride=1, padding=1), nn.ReLU(),
      nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
      nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
      nn.AdaptiveAvgPool2d((2, 4)),
    )
    self.frame_projection = nn.Sequential(nn.Flatten(), nn.Linear(512, frame_dim), nn.ReLU())
    self.gru = WindowGRU(frame_dim, latent_dim)

  def _encode_frames(self, depth):
    return self.frame_projection(self.frame_cnn(depth[:, None]))

  def encode_frames(self, depth):
    """Encode [..., 18, 32] once per frame; preserve gradients to predicted images.

    Callers may reuse these features only while this encoder's parameters are
    unchanged. No cache or recurrent hidden state is stored on the module.
    """
    if depth.shape[-2:] != (18, 32):
      raise ValueError("DepthEncoder expects [..., frames, 18, 32]")
    leading = depth.shape[:-2]
    depth = depth_float(depth).reshape(-1, 18, 32)
    # Keep the previous CNN workspace bound: batch_size history windows.
    frame_batch_size = self.batch_size * self.frames_per_window
    results = []
    for chunk in depth.split(frame_batch_size):
      if self.checkpoint_activations and torch.is_grad_enabled():
        results.append(checkpoint(self._encode_frames, chunk, use_reentrant=False))
      else:
        results.append(self._encode_frames(chunk))
    return torch.cat(results, 0).reshape(*leading, -1)

  def _aggregate(self, features):
    return self.gru(features)[1][0]

  def aggregate(self, features):
    """Aggregate the same strided history window as before, always from h=0."""
    leading = features.shape[:-2]
    features = features.reshape(-1, *features.shape[-2:])
    results = []
    for chunk in features.split(self.batch_size):
      if self.checkpoint_activations and torch.is_grad_enabled():
        results.append(checkpoint(self._aggregate, chunk, use_reentrant=False))
      else:
        results.append(self._aggregate(chunk))
    return torch.cat(results, 0).reshape(*leading, -1)

  def forward(self, depth):
    return self.aggregate(self.encode_frames(depth))
