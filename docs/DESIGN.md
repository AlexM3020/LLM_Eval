# LLM Eval Service — Design Document

An evaluation-as-a-service platform where models, tasks/datasets,
and evaluators are all pluggable — with SQuAD-style extractive QA as the working
proof of concept and AG News classification as the proof that the abstraction is real.

This document is written to double as interview preparation. Every section maps to
questions you can expect in **LLM Evaluation** and **ML System Design** interviews.

---

## 1. Goals and non-goals

**Goals**
- Evaluate any registered model on any registered task with one API call:
  an *eval job* = (model × task × dataset slice × evaluator set × prompt version).
- Produce trustworthy numbers: deterministic metrics where they exist,
  LLM-as-a-judge where they don't — with the judge itself validated against human labels.
- Support *regression testing*: "did prompt v2 get better than v1, on which examples,
  at what cost?".
- Production-grade system design: async execution, rate limiting, caching,
  cost/latency tracking, durable and immutable run storage.

**Non-goals**
- Fixed set of tasks (2 implemented), one dataset per task. No user-uploaded datasets
  or arbitrary task definitions — that's v2.
- Models pluggable only across OpenAI-compatible API providers (no local GPU inference).
- Single-machine deployment; the scaling path is *designed* (§6.6), not built.

---

## 2. What this project demonstrates


|---|---|
| Metric selection & limitations | §5.1 — EM/F1 for QA, accuracy for classification, and *why they are insufficient* |
| LLM-as-a-judge done responsibly | §5.2 — rubric design, temperature 0, pairwise mode with position-bias correction |
| Judge validation / meta-evaluation | §5.3 — human calibration set, Cohen's kappa, judge config selection |
| Eval-driven development | §5.4 — prompt versioning, regression diffs, "CI for prompts" |
| Failure analysis | §5.5 — slice analysis, failure taxonomy from stored per-example scores |
| API & data modeling | §3 — EvalJob as single source of truth; immutable runs schema |
| Async systems & queues | §6.1 — job queue + worker pool, SKIP LOCKED claim semantics |
| Rate limiting & retries | §6.2 — token bucket in the worker, exponential backoff + jitter in the adapter |
| Caching | §6.3 — exact-match response cache; semantic cache as a designed stretch goal |
| Cost/latency engineering | §6.5 — per-call token accounting, cost rollups per run |
| Scaling & failure modes | §6.6, §6.7 — throughput math, crash recovery, partial results |

---

## 3. Architecture

```
        POST /evals  (EvalJob: model × task × dataset × evaluators × prompt_version)
                          │
   FastAPI ──────────► JobQueue (SQLite → Postgres) ──────────► Worker pool
      │                                                          │
      │                ┌─────────────────────────────────────────▼─────────┐
      │                │ Executor (registry-driven; sync baseline kept)    │
      │                │   ModelAdapter.generate ─ cache → retry → _call   │
      │                │   Task: load_dataset / build_prompt / parse / score│
      │                │   Evaluators: deterministic, llm_judge            │
      │                └─────────────────────────────────────────┬─────────┘
      │                                                          │
      └────────────────► Store: runs / predictions / scores ◄────┘
                        GET /evals · GET /evals/{id} · GET /compare
```

**Key structural decision:** all variability lives behind three registries
(`models`, `tasks`, `evaluators`). The executor and API depend only on the
interfaces. Adding a component is one new file + one decorator — which is why
AG News could be added on Day 6 in a couple of hours without touching the runner.

**Data model (storage/db.py):**
- `runs` — one row per eval job execution; stores the full job JSON (model, task,
  dataset path/split, evaluator list, prompt version). **Runs are immutable and
  append-only** — that is what makes experiment comparison trustworthy.
- `predictions` — per (run, example): raw model output, parsed prediction, latency, cost.
- `scores` — per (run, evaluator, example, metric): value. Per-example granularity is
  deliberate: aggregates alone can't answer "which examples regressed?"

**Request lifecycle:** `POST /evals` validates an `EvalJob` (a Pydantic model shared
by API and runner, so they can't drift), enqueues it, returns a run id immediately.
A worker claims the job, runs the executor, persists predictions and scores, marks
the run DONE. `GET /compare?a=&b=` returns metric deltas between two runs.

---

## 4. The extensibility model

Three interfaces define everything:

- **`ModelAdapter`** (`models/base.py`): subclasses implement one method, `_call()`.
  The template method `generate()` wraps it with caching, retries, and latency/cost
  instrumentation — cross-cutting concerns live in exactly one place.
  *Swapping providers = config change* (the OpenAI adapter speaks a protocol, not a vendor).
- **`Task`** (`tasks/base.py`): `load_dataset / build_prompt / parse_prediction /
  score_example`. Prompt templates live *inside* the task and are versioned through
  the job's `prompt_version` field.
- **`Evaluator`** (`evaluators/base.py`): cross-task scoring strategies.
  Task-specific deterministic metrics stay in `Task.score_example()`; evaluators
  aggregate or judge.

**The one discipline that makes this work:** nothing task-specific outside a `Task`
implementation. The executor file mentions no dataset name, no metric name, no provider.
In an interview, frame this as: *"the generalization was nearly free because I defined
the interfaces before writing any implementation — the cost of pluggability is upfront
design, not ongoing complexity."*

---

## 5. LLM evaluation methodology

### 5.1 Deterministic metrics (implemented)

- **SQuAD (PoC task):** Exact Match and token-level F1 with the standard SQuAD v1
  normalization (lowercase, strip articles/punctuation). Implemented from scratch in
  `tasks/squad.py` with known-value unit tests.
- **AG News (second task):** accuracy over 4 classes.

**Be ready to discuss their limits:** EM/F1 reward span overlap, not correctness —
"Paris" and "the capital of France, Paris" get partial credit while a correct
paraphrase gets zero. This motivates the judge. Also know *why* F1 maxes over multiple
gold references (`best_over_references`) and what that does to score distributions.

### 5.2 LLM-as-a-judge 

`evaluators/llm_judge.py`, two modes:

- **Pointwise:** rubric-based 1–5 scoring with references in the prompt, temperature 0.
  Parse failures are logged and dropped, never silently counted (stretch: constrained
  decoding / retry-on-parse-failure).
- **Pairwise with position-bias correction:** present (A, B) *and* (B, A); accept a
  winner only if both orderings agree; **log the flip rate as a bias metric**.
  Known judge biases to name in the interview: position bias, verbosity bias,
  self-preference bias, and egocentricity (judging answers similar to what the judge
  itself would produce).

### 5.3 Human calibration 

A judge you haven't validated is a random number generator with confidence.
The methodology:

1. Hand-label 50–100 examples yourself (binary correct/incorrect, or 1–5).
2. Run each candidate judge configuration (judge model × rubric style × CoT vs direct)
   on the same examples.
3. Compare with **Cohen's kappa** (`cohen_kappa()` implemented in `llm_judge.py`),
   which corrects for chance agreement — raw agreement is misleading on imbalanced data.
4. Pick the cheapest config whose kappa clears your bar (≈0.6+ "substantial"), and
   **report the judge's agreement alongside every metric it produces.**

This is what separates "used LLM-as-a-judge" from "understands LLM evaluation."

### 5.4 Regression testing and prompt versioning 

Every run stores `prompt_version`. `GET /compare?a=&b=` returns per-metric deltas;
the Day-6 TODO extends it to per-example diffs joined on `example_id`. This is
"CI for prompts": change a template → rerun the suite → see exactly which examples
improved or regressed, at what cost delta. Talking point: why aggregates hide
regressions ( Simpson's-paradox-style masking across slices).

### 5.5 Slice analysis & failure taxonomy

Because per-example scores and raw outputs are stored, you can bucket failures
(wrong-span vs hallucinated vs unparseable vs retrieval-miss if you add RAG) and
slice by example metadata (context length, question type). One afternoon of this
analysis gives you more interview material than any extra feature.

---

## 6. ML system design

### 6.1 Sync baseline → queue + workers 

LLM evals are slow (200 examples × ~1 s/call) and rate-limited — a synchronous
request handler is the wrong shape. Design: jobs table as the queue (SQLite now;
Postgres `SELECT ... FOR UPDATE SKIP LOCKED` later — no Redis needed at this scale).
Workers claim PENDING jobs, heartbeat, mark DONE/FAILED. The sync executor stays as
the local/debug path — same code, different driver.

### 6.2 Rate limiting & retries

- **Retries** live in `ModelAdapter.generate()`: exponential backoff, max 4 attempts
  (TODO: jitter, respect `Retry-After`, don't retry 4xx).
- **Rate limiting** belongs in the *worker* (token bucket around `generate`), not
  scattered across adapters — one place to enforce per-provider RPM/TPM budgets.
- **Idempotency:** re-running a job produces a new run row; runs never mutate. Safe
  retries by construction.

### 6.3 Caching

`caching.py`: exact-match cache keyed by sha256(model | prompt | generation config).
Makes crashed-run resumes and prompt-version comparisons free. Stretch: semantic cache
(embedding-similarity threshold) — be ready to discuss the false-hit risk: two
semantically-near prompts with different correct answers get poisoned, which in an
*eval harness* silently corrupts metrics. That tradeoff discussion is worth more than
building it.

### 6.4 Storage & versioning

Schema in §3. Principles: runs immutable; the full job spec stored with the run
(reproducibility); per-example granularity (enables slice analysis and pairwise diffs).
SQLite → Postgres is a connection-string change by design.

### 6.5 Cost & latency tracking

Every `GenerationResult` carries tokens, latency, and USD cost (price table per model);
runs roll up totals. This turns "prompt v2 is 1.3 F1 points better" into "1.3 points
better at 2.1× the cost" — the kind of statement that signals production maturity.

### 6.6 Scaling path (designed, not built)

1 worker → N workers on one box (jobs table handles claiming) → workers on separate
machines (Postgres) → per-provider queues with independent rate budgets.
Throughput math to have ready: 200-example eval × 2 s/call ≈ 7 min serial; with 10
concurrent workers against a 500 RPM budget, wall-clock is rate-limit-bound, not
worker-bound — so scale the *budget*, not just the pool.

### 6.7 Failure modes & mitigations

| Failure | Mitigation |
|---|---|
| Worker crashes mid-run | job stays claimed-but-stale; heartbeat timeout re-queues it |
| Provider rate limit / 5xx | backoff in adapter; budget in worker |
| Judge returns unparseable text | score recorded as null, never silently dropped from the audit trail |
| Model output breaks `parse_prediction` | contract: parse never raises; bad parses score as wrong and are visible in stored raw output |
| Poisoned cache (stale price table, config drift) | cache key includes full generation config |

### 6.8 Observability

Per run: aggregate metrics, total cost, p50/p95 latency, cache hit rate, judge parse
failure rate, position-flip rate. Dashboard (Streamlit or a minimal web UI) on Day 7 —
but the *data model* already supports everything the dashboard needs.

---


## 9. References 

- Rajpurkar et al., *SQuAD* (2016) — the EM/F1 protocol you implemented
- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (2023) — judge biases, position swap
- Liang et al., *HELM* (2022) — holistic, multi-metric evaluation framing
- Es et al., *RAGAS* (2023) — reference-free metrics (faithfulness, answer relevance) if you extend to RAG
