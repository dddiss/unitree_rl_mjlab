"""Decode one normalized next depth image, not the entire observation history."""
import torch.nn.functional as F
from torch import nn


class DepthDecoder(nn.Module):
  def __init__(self, latent_dim=256):
    super().__init__()
    self.projection = nn.Sequential(nn.Linear(latent_dim, 64 * 3 * 4), nn.SiLU())
    self.coarse = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.SiLU())
    self.fine = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.SiLU(),
                              nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid())

  def forward(self, latent):
    x = self.projection(latent).reshape(-1, 64, 3, 4)
    x = self.coarse(F.interpolate(x, size=(9, 16), mode="bilinear", align_corners=False))
    x = self.fine(F.interpolate(x, size=(18, 32), mode="bilinear", align_corners=False))
    return x[:, 0].reshape(*latent.shape[:-1], 18, 32)
