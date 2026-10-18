"""LLM-as-a-judge, task-agnostic: the task supplies context, any ModelAdapter can judge.

Reliability features — these are your strongest interview talking points:
  - pointwise rubric scoring at temperature 0 with parse-failure handling
  - pairwise comparison with POSITION-BIAS CORRECTION (judge both orders, keep
    only verdicts that are stable across orderings; log the flip rate)
  - calibration against human labels via cohen_kappa() below
"""
from __future__ import annotations

from collections import Counter

from ..models.base import GenerationConfig, ModelAdapter
from ..registry import EVALUATOR_REGISTRY
from .base import EvalReport, Evaluator

POINTWISE_RUBRIC = """You are grading a model's answer for this task: {task_description}

Input given to the model:
{inputs}

Reference answer(s): {references}

Model's answer: {prediction}

Score the model's answer from 1 (wrong or unhelpful) to 5 (fully correct).
Reply with only the integer."""

PAIRWISE_RUBRIC = """You are comparing two answers for this task: {task_description}

Input given to the models:
{inputs}

Reference answer(s): {references}

Answer A: {answer_a}
Answer B: {answer_b}

Which answer is better? Reply with only "A", "B", or "tie"."""


@EVALUATOR_REGISTRY.register("llm_judge")
class LLMJudgeEvaluator(Evaluator):
    name = "llm_judge"

    def __init__(self, judge: ModelAdapter, mode: str = "pointwise"):
        self.judge = judge
        self.mode = mode

    def evaluate(self, task, examples, predictions) -> EvalReport:
        if self.mode == "pairwise":
            raise NotImplementedError(
                "TODO(Day 4): pairwise mode — present (A,B) and (B,A) to the judge, "
                "accept a winner only if both orderings agree, log the flip rate "
                "as a position-bias metric."
            )
        by_id = {e.id: e for e in examples}
        rows = []
        for p in predictions:
            ex = by_id[p.example_id]
            prompt = POINTWISE_RUBRIC.format(
                task_description=task.description,
                inputs=ex.inputs,
                references=ex.references,
                prediction=p.parsed,
            )
            result = self.judge.generate(prompt, GenerationConfig(temperature=0.0, max_tokens=4))
            rows.append({"example_id": p.example_id, "judge_score": self._parse_score(result.text)})
        valid = [r["judge_score"] for r in rows if r["judge_score"] is not None]
        aggregate = {"judge_score": sum(valid) / len(valid)} if valid else {}
        return EvalReport(evaluator=self.name, per_example=rows, aggregate=aggregate)

    @staticmethod
    def _parse_score(text: str) -> int | None:
        # TODO(Day 4): retry-on-parse-failure or constrained decoding instead of dropping.
        try:
            return max(1, min(5, int(text.strip()[0])))
        except (ValueError, IndexError):
            return None


def cohen_kappa(labels_a: list, labels_b: list) -> float:
    """Inter-annotator agreement between judge scores and your human labels.

    Use this on Day 4: hand-label 50-100 examples, then compare judge
    configurations (model, rubric style, CoT vs direct) by kappa, not vibes.
    """
    assert len(labels_a) == len(labels_b) and labels_a, "need equal, non-empty label lists"
    n = len(labels_a)
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    ca, cb = Counter(labels_a), Counter(labels_b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0
