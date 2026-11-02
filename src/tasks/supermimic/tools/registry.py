"""Independent SuperMimic task registry; never registers PPO runners with mjlab."""
from dataclasses import dataclass, replace
from importlib import import_module

# Relative factory paths stay rooted at supermimic, not its tools subpackage.
_TASK_PACKAGE = __package__.rpartition(".")[0]

@dataclass(frozen=True)
class TaskSpec:
  env_cfg_factory: str
  runner_cfg_factory: str


_TASKS: dict[str, TaskSpec] = {}


def register_task(task_id: str, *, env_cfg_factory: str, runner_cfg_factory: str):
  """Register lazy factories so listing tasks needs neither a simulator nor a GPU."""
  if not task_id or task_id in _TASKS:
    raise ValueError(f"Task name is empty or already registered: {task_id!r}")
  _TASKS[task_id] = TaskSpec(env_cfg_factory, runner_cfg_factory)


def list_tasks() -> tuple[str, ...]:
  import_module(".config.g1", package=_TASK_PACKAGE)
  return tuple(sorted(_TASKS))


def get_task(task_id: str) -> TaskSpec:
  available = list_tasks()
  if task_id not in _TASKS:
    raise ValueError(f"Unknown SuperMimic task {task_id!r}. Available tasks: {', '.join(available)}")
  return _TASKS[task_id]


def _factory(path):
  module, name = path.rsplit(":", 1)
  return getattr(import_module(module, package=_TASK_PACKAGE), name)


def load_env_cfg(task_id: str, **kwargs):
  """Construct a fresh environment configuration on every call."""
  return _factory(get_task(task_id).env_cfg_factory)(**kwargs)


def load_runner_cfg(task_id: str):
  return replace(_factory(get_task(task_id).runner_cfg_factory)(), task_id=task_id)


def runner_cfg_from_dict(config: dict):
  """Restore the registered task's concrete dataclass, including subclass fields."""
  return replace(load_runner_cfg(config["task_id"]), **config)
