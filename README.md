# LLM Eval Service

A small **evaluation-as-a-service** platform for LLM applications.
Models, tasks/datasets, and evaluators are all pluggable via registries — the
service itself knows nothing about SQuAD, OpenAI, or F1.

Built as interview preparation for **LLM Evaluation** + **ML System Design**
roles. See [docs/DESIGN.md](docs/DESIGN.md) for the full design document,
methodology, and 7-day build plan.

## Quickstart

```bash
pip install -e ".[dev]"
pytest                              # 12 tests, no API key needed

# End-to-end smoke run with the deterministic mock model
python scripts/run_eval.py --model mock --task squad \
    --dataset-path tests/fixtures/squad_tiny.jsonl

# Real run (OpenAI-compatible provider)
export OPENAI_API_KEY=...
python scripts/run_eval.py --model openai --model-kwargs '{"model": "gpt-4o-mini"}' \
    --task squad --dataset-path data/squad_val.jsonl --limit 200 \
    --evaluators deterministic llm_judge --prompt-version v1

# API
uvicorn eval_service.api.main:app --app-dir src
```

## Architecture

```
        POST /evals  (EvalJob: model x task x dataset x evaluators)
                     |
 FastAPI --------> JobQueue  --------> Workers
  |                                            |
  |                       +--------------------v--------------------+
  |                       | Executor (registry-driven, sync baseline)|
  |                       |  ModelAdapter.generate  (cache, retries) |
  |                       |  Task: prompt -> parse -> score_example  |
  |                       |  Evaluators: deterministic / LLM-judge   |
  |                       +--------------------+--------------------+
  |                                            |
  +-----------------> Store (runs, predictions, scores) <------------+
                     GET /evals, GET /compare (regression diff)
```

## Repo map

```
src/eval_service/
  registry.py        generic registries — the pluggability mechanism
  models/            ModelAdapter interface + mock / OpenAI-compatible adapters
  tasks/             Task interface + squad (PoC) + agnews (extensibility proof)
  evaluators/        deterministic aggregation + LLM-as-a-judge + Cohen's kappa
  runner/            EvalJob schema, sync executor, job queue (Day 5 stub)
  storage/           SQLite store: runs, predictions, scores, compare_runs()
  caching.py         exact-match response cache
  api/               FastAPI surface
scripts/run_eval.py  CLI entrypoint
tests/               metric known-values, registry, end-to-end mock run
```

## Adding a component (the whole point)

- **New task**: subclass `Task`, decorate with `@TASK_REGISTRY.register("name")`. Nothing else changes.
- **New model/provider**: subclass `ModelAdapter`, implement `_call()`. Caching/retries/cost tracking come free.
- **New evaluator**: subclass `Evaluator`, decorate with `@EVALUATOR_REGISTRY.register("name")`.

