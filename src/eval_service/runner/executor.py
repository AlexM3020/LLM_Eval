"""Synchronous executor — the Day 2-3 baseline.

Day 5 replaces the inline loop with a job queue + worker pool; keep this as the
local/debug path. Discipline: nothing in this file may know about SQuAD, OpenAI,
or F1 — only registries and interfaces.
"""
from __future__ import annotations

import uuid

from ..models.base import ModelAdapter
from ..registry import EVALUATOR_REGISTRY, MODEL_REGISTRY, TASK_REGISTRY
from ..tasks.base import Prediction
from .jobs import EvalJob


def run_job(job: EvalJob, store=None, judge: ModelAdapter | None = None) -> dict:
    job.id = job.id or uuid.uuid4().hex[:12]

    model = MODEL_REGISTRY.create(job.model, **job.model_kwargs)
    task = TASK_REGISTRY.create(job.task)
    examples = task.load_dataset(
        split=job.dataset_split, limit=job.limit, path=job.dataset_path
    )

    predictions: list[Prediction] = []
    for ex in examples:
        # TODO(Day 5): bounded-concurrency async fan-out with rate limiting.
        result = model.generate(task.build_prompt(ex))
        predictions.append(
            Prediction(
                example_id=ex.id,
                raw_output=result.text,
                parsed=task.parse_prediction(result.text),
                latency_ms=result.latency_ms,
                cost_usd=result.cost_usd,
            )
        )

    reports = {}
    for name in job.evaluators:
        evaluator = (
            EVALUATOR_REGISTRY.create(name, judge=judge)
            if name == "llm_judge"
            else EVALUATOR_REGISTRY.create(name)
        )
        reports[name] = evaluator.evaluate(task, examples, predictions)

    run_record = {
        "id": job.id,
        "job": job.model_dump(mode="json"),
        "metrics": {k: v.aggregate for k, v in reports.items()},
        "total_cost_usd": sum(p.cost_usd for p in predictions),
        "total_latency_ms": sum(p.latency_ms for p in predictions),
    }
    if store is not None:
        store.save_run(run_record, predictions, reports)
    return run_record
