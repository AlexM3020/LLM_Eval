"""Evaluator interface.

Task-specific deterministic metrics live in Task.score_example(). Evaluators are
*cross-task* scoring strategies: aggregation of deterministic metrics,
LLM-as-a-judge, and anything else you add later.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..tasks.base import Example, Prediction, Task


@dataclass
class EvalReport:
    evaluator: str
    per_example: list[dict] = field(default_factory=list)  # [{"example_id": ..., metric: value}]
    aggregate: dict[str, float] = field(default_factory=dict)


class Evaluator(ABC):
    name: str

    @abstractmethod
    def evaluate(
        self, task: Task, examples: list[Example], predictions: list[Prediction]
    ) -> EvalReport:
        ...
