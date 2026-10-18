"""CLI entrypoint.

Examples:
  python scripts/run_eval.py --model mock --task squad \
      --dataset-path tests/fixtures/squad_tiny.jsonl

  python scripts/run_eval.py --model openai --task squad \
      --dataset-path data/squad_val.jsonl --limit 200 \
      --evaluators deterministic llm_judge --prompt-version v2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval_service.runner.executor import run_job  # noqa: E402
from eval_service.runner.jobs import EvalJob  # noqa: E402
from eval_service.storage.db import Store  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Run an LLM evaluation job.")
    p.add_argument("--model", default="mock", help="registry name of the model adapter")
    p.add_argument("--model-kwargs", default="{}", help='JSON, e.g. \'{"model": "gpt-4o"}\'')
    p.add_argument("--task", default="squad")
    p.add_argument("--dataset-path", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--evaluators", nargs="+", default=["deterministic"])
    p.add_argument("--prompt-version", default="v1")
    p.add_argument("--no-store", action="store_true")
    args = p.parse_args()

    job = EvalJob(
        model=args.model,
        model_kwargs=json.loads(args.model_kwargs),
        task=args.task,
        dataset_path=args.dataset_path,
        limit=args.limit,
        evaluators=args.evaluators,
        prompt_version=args.prompt_version,
    )
    record = run_job(job, store=None if args.no_store else Store())
    print(json.dumps(record, indent=2, default=str))


if __name__ == "__main__":
    main()
