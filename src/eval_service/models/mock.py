"""Deterministic fake model for development and tests — no API key needed."""
from __future__ import annotations

from ..registry import MODEL_REGISTRY
from .base import GenerationConfig, GenerationResult, ModelAdapter


@MODEL_REGISTRY.register("mock")
class MockAdapter(ModelAdapter):
    name = "mock"

    def __init__(self, response: str = "mock answer"):
        self.response = response

    def _call(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        return GenerationResult(
            text=self.response,
            model=self.name,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(self.response.split()),
        )
