"""Generic registries — the heart of the pluggable design.

Every swappable component kind (models, tasks, evaluators) gets one registry.
Adding a new component = writing a class and decorating it; no other file changes.
"""
from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, type] = {}

    def register(self, name: str | None = None) -> Callable[[T], T]:
        def decorator(cls: T) -> T:
            key = name or cls.__name__
            if key in self._items:
                raise ValueError(f"{self.kind} '{key}' is already registered")
            self._items[key] = cls
            return cls

        return decorator

    def get(self, name: str) -> type:
        if name not in self._items:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {self.list()}"
            )
        return self._items[name]

    def create(self, name: str, **kwargs):
        return self.get(name)(**kwargs)

    def list(self) -> list[str]:
        return sorted(self._items)


MODEL_REGISTRY = Registry("model")
TASK_REGISTRY = Registry("task")
EVALUATOR_REGISTRY = Registry("evaluator")
