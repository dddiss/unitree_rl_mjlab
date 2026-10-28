import torch
from supermimic.perception.history import DepthHistory, select_history, window_from_frames


def test_strided_history_and_partial_reset():
  history = DepthHistory(2,36,'cpu')
  yes, no = torch.ones(2,dtype=torch.bool), torch.zeros(2,dtype=torch.bool)
  frame = lambda t: torch.full((2,18,32),t,dtype=torch.uint8)
  history.update(frame(0),yes,yes)
  for t in range(1,41): history.update(frame(t),no,yes)
  assert select_history(history.data)[0,:,0,0].tolist() == [5,10,15,20,25,30,35,40]
  saved = history.data.clone()
  history.update(frame(99),no,no)  # Viewer asking for another action doesn't advance time.
  torch.testing.assert_close(history.data,saved)
  history.update(frame(100),torch.tensor([True,False]),no)
  assert history.data[0].eq(100).all()
  torch.testing.assert_close(history.data[1],saved[1])


def test_predicted_frames_enter_the_correct_strided_slot():
  queue = tuple(torch.full((1,18,32),float(t)) for t in range(36))
  first = torch.full((1,18,32),36.,requires_grad=True)
  for t in range(36,42):
    image = first if t == 36 else torch.full_like(first,float(t))
    queue = (*queue[1:],image)
  window = window_from_frames(queue)
  assert window[0,:,0,0].tolist() == [6,11,16,21,26,31,36,41]
  window.sum().backward()
  torch.testing.assert_close(first.grad,torch.ones_like(first))
