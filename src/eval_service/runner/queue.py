"""Day 5 target: persistent job queue + worker pool.

The interface is fixed NOW so the API and executor don't change when this lands.

Design notes (see docs/DESIGN.md, System Design section):
  - A SQLite/Postgres-backed jobs table is enough (SELECT ... FOR UPDATE SKIP
    LOCKED once you're on Postgres); no need for Redis on a one-week build
  - Workers claim PENDING jobs, heartbeat while running, mark DONE/FAILED
  - Rate limiting belongs in the WORKER (a token bucket around model.generate),
    not scattered across adapters
"""
from __future__ import annotations

from .jobs import EvalJob, JobStatus


class JobQueue:
    """TODO(Day 5): implement against SQLite first, Postgres via DATABASE_URL."""

    def submit(self, job: EvalJob) -> str:
        raise NotImplementedError

    def claim(self, worker_id: str) -> EvalJob | None:
        raise NotImplementedError

    def complete(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        raise NotImplementedError
