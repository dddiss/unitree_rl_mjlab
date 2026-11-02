"""Optional signed outputs for additional raw reward prediction targets."""
import torch
from torch import nn


class RewardOutputDomains(nn.Module):
  def __init__(self, nonnegative):
    super().__init__()
    self.register_buffer("nonnegative", torch.tensor(nonnegative, dtype=torch.bool), persistent=False)

  def forward(self, values):
    return torch.where(self.nonnegative, torch.nn.functional.softplus(values), values)


def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


# MLP matmuls only; state integration, FK and rewards remain FP32.
AMP_DTYPE: torch.dtype | None = None


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None or x.device.type != "cuda":
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()
