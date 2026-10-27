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
    self._window_choices = {}

  @torch.no_grad()
  def add(self, transition):
    for key in FIELDS:
      self.data[key][self.ptr].copy_(transition[key])
    f = transition["f"]
    # f is the next target, so even the last frame has one valid transition.
    self.eligible[self.ptr] = self.clip_end[self.clip_id[f]] - f + 1 >= 1
    self.ptr = (self.ptr + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)
    self._window_choices.clear()

  def _row(self, age):
    return (self.ptr - self.size + age) % self.capacity

  def _strict_window_choices(self, length):
    """Return chronological (start age, environment) pairs within one episode."""
    if length not in self._window_choices:
      rows = self._row(torch.arange(self.size, device=self.device))
      steps = self.data["ep_step"][rows]
      done = self.data["terminated"][rows] | self.data["truncated"][rows]
      # A boundary sits between two transitions. The last transition in a
      # window may end the episode, but none of its earlier transitions may.
      boundaries = done[:-1] | (steps[1:] != steps[:-1] + 1)
      prefix = torch.cat((torch.zeros(1, self.num_envs, dtype=torch.long,
                                     device=self.device),
                          boundaries.cumsum(0)), dim=0)
      count = self.size - length + 1
      valid = prefix[length - 1:] == prefix[:count]
      self._window_choices[length] = valid.nonzero(as_tuple=False)
    return self._window_choices[length]

  def sample_windows(self, batch, length, *, strict=False):
    """Sample WM windows, optionally requiring all steps in one episode.

    Warmup uses the original masked windows. Strict sampling returns None if
    no complete window is available; it never loops over rejected samples.
    """
    if self.size < length:
      if strict:
        return None
      raise ValueError("Not enough collected steps for a world-model window")
    if strict:
      choices = self._strict_window_choices(length)
      if choices.numel() == 0:
        return None
      selected = choices[torch.randint(len(choices), (batch,), device=self.device)]
      ages, envs = selected[:, :1], selected[:, 1:]
    else:
      ages = torch.randint(self.size - length + 1, (batch,), device=self.device)[:, None]
      envs = torch.randint(self.num_envs, (batch,), device=self.device)[:, None]
    rows = self._row(ages + torch.arange(length, device=self.device))
    out = {k: self.data[k][rows, envs] for k in ("s", "s1", "a", "terminated", "truncated")}
    done = out["terminated"] | out["truncated"]
    out["mask"] = torch.cat((torch.ones(batch, 1, device=self.device),
                              (~done[:, :-1]).float()), -1).cumprod(-1)
    out["xi"] = self.xi[envs].expand(-1, length, -1)
    return out

  def sample_starts(self, batch, history):
    """Sample collected starts, including clips with fewer than horizon steps."""
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
