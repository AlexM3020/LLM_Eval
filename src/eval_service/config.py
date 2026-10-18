"""Runtime configuration, sourced from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///./eval_service.db"
    cache_path: str = "./.cache/responses.sqlite3"
    openai_api_key: str | None = None
    openai_base_url: str | None = None  # any OpenAI-compatible endpoint
    judge_model: str = "gpt-4o-mini"    # model used by the LLM-as-a-judge evaluator

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=os.getenv("DATABASE_URL", cls.database_url),
            cache_path=os.getenv("CACHE_PATH", cls.cache_path),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            judge_model=os.getenv("JUDGE_MODEL", cls.judge_model),
        )
