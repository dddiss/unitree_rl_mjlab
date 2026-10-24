"""Time × environment replay, adapted from vibed (THIRD_PARTY_NOTICES.md)."""
import torch

from ..mdp.state import STATE_DIM

FIELDS = {"s": (STATE_DIM, torch.float32), "s1": (STATE_DIM, torch.float32),
          "a": (29, torch.float32), "a_prev": (29, torch.float32),
          "r": (None, torch.float32), "f": (None, torch.long),
          "ep_step": (None, torch.long), "terminated": (None, torch.bool),
          "truncated": (None, torch.bool)}


class ReplayBuffer:
  def __init__(self, capacity, num_envs, xi, clip_end, clip_id, horizon):
    self.capacity, self.num_envs = capacity, num_envs
    self.xi = xi.detach().clone()  # startup parameters, immutable until buffer recreation
    self.device = xi.device
    self.clip_end, self.clip_id, self.horizon = clip_end, clip_id, horizon
    self.data = {k: torch.empty((capacity, num_envs) + (() if n is None else (n,)),
                               dtype=dtype, device=self.device) for k, (n, dtype) in FIELDS.items()}
    self.eligible = torch.zeros(capacity, num_envs, dtype=torch.bool, device=self.device)
    self.ptr = self.size = 0

  @torch.no_grad()
  def add(self, transition):
    for key in FIELDS:
      self.data[key][self.ptr].copy_(transition[key])
    f = transition["f"]
    self.eligible[self.ptr] = self.clip_end[self.clip_id[f]] - f + 1 >= self.horizon
    self.ptr = (self.ptr + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)

  def _row(self, age):
    return (self.ptr - self.size + age) % self.capacity

  def sample_windows(self, batch, length):
    if self.size < length:
      raise ValueError("Not enough collected steps for a world-model window")
    ages = torch.randint(self.size - length + 1, (batch,), device=self.device)[:, None]
    rows = self._row(ages + torch.arange(length, device=self.device))
    envs = torch.randint(self.num_envs, (batch,), device=self.device)[:, None]
    out = {k: self.data[k][rows, envs] for k in ("s", "s1", "a", "terminated", "truncated")}
    done = out["terminated"] | out["truncated"]
    out["mask"] = torch.cat((torch.ones(batch, 1, device=self.device),
                              (~done[:, :-1]).float()), -1).cumprod(-1)
    out["xi"] = self.xi[envs].expand(-1, length, -1)
    return out

  def sample_starts(self, batch, history):
    """Finite sampling from valid physical slots; None means no legal horizon."""
    choices = self.eligible.nonzero(as_tuple=False)
    if choices.numel() == 0:
      return None
    selected = choices[torch.randint(len(choices), (batch,), device=self.device)]
    row, envs = selected.unbind(-1)
    age = (row - (self.ptr - self.size)) % self.capacity
    step = self.data["ep_step"][row, envs]
    back = torch.arange(history - 1, -1, -1, device=self.device)[None]
    back = torch.minimum(torch.minimum(back, step[:, None]), age[:, None])
    rows = self._row(age[:, None] - back)
    keys = ("s", "a_prev", "f")
    start = {k: self.data[k][row, envs] for k in keys}
    hist = {k: self.data[k][rows, envs[:, None]] for k in keys}
    start["xi"] = self.xi[envs]
    hist["xi"] = self.xi[envs, None].expand(-1, history, -1)
    return start, hist
