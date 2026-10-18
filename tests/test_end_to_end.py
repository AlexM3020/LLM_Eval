"""End-to-end smoke test: mock model + tiny SQuAD fixture, no API keys, no DB."""
import os

from eval_service.runner.executor import run_job
from eval_service.runner.jobs import EvalJob

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "squad_tiny.jsonl")


def test_run_job_end_to_end_with_mock_model():
    job = EvalJob(model="mock", task="squad", dataset_path=FIXTURE)
    record = run_job(job, store=None)

    assert record["id"]
    assert record["job"]["task"] == "squad"
    metrics = record["metrics"]["deterministic"]
    assert set(metrics) == {"exact_match", "f1"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    # mock model answers "mock answer" to everything -> nothing matches
    assert metrics["exact_match"] == 0.0
