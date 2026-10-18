"""An eval job fully describes one experiment: model x task x data slice x evaluators.

Pydantic model so it doubles as the FastAPI request schema — the API and the
runner can never drift apart.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class EvalJob(BaseModel):
    model: str                                    # registry name, e.g. "openai"
    model_kwargs: dict = Field(default_factory=dict)
    task: str = "squad"
    dataset_split: str = "validation"
    dataset_path: str | None = None               # local file beats hosted dataset
    limit: int | None = None
    evaluators: list[str] = Field(default_factory=lambda: ["deterministic"])
    prompt_version: str = "v1"                    # bump when a template changes; enables regression diffs
    id: str | None = None
    status: JobStatus = JobStatus.PENDING
