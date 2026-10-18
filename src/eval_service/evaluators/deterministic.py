"""Aggregates the task's own deterministic metrics (EM/F1, accuracy, ...)."""
from __future__ import annotations

from ..registry import EVALUATOR_REGISTRY
from .base import EvalReport, Evaluator


@EVALUATOR_REGISTRY.register("deterministic")
class DeterministicEvaluator(Evaluator):
    name = "deterministic"

    def evaluate(self, task, examples, predictions):
        by_id = {e.id: e for e in examples}
        rows = []
        for p in predictions:
            rows.append({"example_id": p.example_id, **task.score_example(p, by_id[p.example_id])})
        aggregate = {
            key: sum(r[key] for r in rows) / len(rows)
            for key in rows[0]
            if key != "example_id"
        } if rows else {}
        return EvalReport(evaluator=self.name, per_example=rows, aggregate=aggregate)
