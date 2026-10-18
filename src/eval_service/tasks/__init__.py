from .base import Example, Prediction, Task
from . import agnews, squad  # noqa: F401  (importing registers the tasks)

__all__ = ["Example", "Prediction", "Task"]
