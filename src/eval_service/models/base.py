"""Model adapter interface.

Every LLM we evaluate (or use as a judge) sits behind this interface, so the
runner, caching, retries, and cost tracking never care which provider is used.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.0
    max_tokens: int = 512
    extra: dict = field(default_factory=dict)


@dataclass
class GenerationResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False


class ModelAdapter(ABC):
    """Subclass and implement _call(); generate() gives you caching + retries for free."""

    name: str

    def generate(
        self, prompt: str, config: GenerationConfig | None = None
    ) -> GenerationResult:
        """Template method: cache lookup -> _call with retries -> instrumented result."""
        config = config or GenerationConfig()
        # TODO(Day 5): consult ResponseCache (eval_service.caching) before _call.
        return self._with_retries(lambda: self._timed_call(prompt, config))

    def _timed_call(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        start = time.perf_counter()
        result = self._call(prompt, config)
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    @staticmethod
    def _with_retries(fn, max_attempts: int = 4):
        delay = 1.0
        for attempt in range(max_attempts):
            try:
                return fn()
            except Exception:
                if attempt == max_attempts - 1:
                    raise
                time.sleep(delay)  # TODO(Day 5): add jitter; respect Retry-After headers
                delay *= 2

    @abstractmethod
    def _call(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """One raw call to the provider. Raise on transient errors; generate() retries."""
        ...
