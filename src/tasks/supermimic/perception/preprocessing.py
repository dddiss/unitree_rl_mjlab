# Adapted from RoboParty/P-BFM, CC BY-NC 4.0; see THIRD_PARTY_NOTICES.md.
"""G1 optical_z_maxcrop18_reset_v2 preprocessing, without sensor randomization."""
import torch
import torch.nn.functional as F


def preprocess_depth(depth: torch.Tensor, max_distance: float = 2.5) -> torch.Tensor:
  """Optical-Z metres [B,36,64] -> cropped, blurred uint8 [B,18,32]."""
  if depth.shape[-2:] != (36, 64):
    raise ValueError("Expected a 36x64 optical-Z image")
  depth = torch.where(torch.isfinite(depth) & (depth >= 0), depth, max_distance)
  depth = depth.clamp(0, max_distance)[:, 18:, 16:-16]
  gaussian = torch.exp(-torch.arange(-1, 2, device=depth.device, dtype=depth.dtype).square() / 2)
  gaussian = gaussian / gaussian.sum()
  kernel = (gaussian[:, None] * gaussian[None, :])[None, None]
  depth = F.conv2d(F.pad(depth[:, None], (1, 1, 1, 1), mode="reflect"), kernel)[:, 0]
  return (depth.clamp(0, max_distance) / max_distance * 255).round().to(torch.uint8)


def depth_float(depth: torch.Tensor) -> torch.Tensor:
  """Real replay is quantized; predicted frames remain differentiable floats."""
  return depth.float() / 255 if depth.dtype == torch.uint8 else depth.float()
