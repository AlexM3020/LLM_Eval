"""Task interface — the unit of extensibility.

A task knows four things: how to load its dataset, how to turn an example into
a prompt, how to parse raw model output into a prediction, and how to score it
deterministically. NOTHING task-specific may live outside a Task implementation —
that one discipline is what makes the whole service pluggable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Example:
    id: str
    inputs: dict          # e.g. {"question": ..., "context": ...} or {"text": ...}
    references: list[str]  # gold answers / gold label
    meta: dict = field(default_factory=dict)


@dataclass
class Prediction:
    example_id: str
    raw_output: str
    parsed: str
    latency_ms: float = 0.0
    cost_usd: float = 0.0


class Task(ABC):
    name: str
    description: str = ""

    @abstractmethod
    def load_dataset(
        self,
        split: str = "validation",
        limit: int | None = None,
        path: str | None = None,
    ) -> list[Example]:
        """Load examples from a local file (path=) or a hosted dataset."""

    @abstractmethod
    def build_prompt(self, example: Example) -> str:
        """Prompt template lives HERE, versioned via the job's prompt_version field."""

    @abstractmethod
    def parse_prediction(self, raw_output: str) -> str:
        """Raw model text -> comparable prediction. Never raise; bad parses score as wrong."""

    @abstractmethod
    def score_example(self, prediction: Prediction, example: Example) -> dict[str, float]:
        """Deterministic per-example metrics, e.g. {"exact_match": 1.0, "f1": 0.67}."""
