"""Adapter for any OpenAI-compatible chat API (OpenAI, Azure, vLLM, Together, ...).

Because it speaks the OpenAI protocol, swapping providers is a config change
(base_url + api_key + model name), not a code change.
"""
from __future__ import annotations

import os

from ..registry import MODEL_REGISTRY
from .base import GenerationConfig, GenerationResult, ModelAdapter

# Rough USD per 1M tokens as (input, output). Extend as needed; used for cost tracking.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


@MODEL_REGISTRY.register("openai")
class OpenAIAdapter(ModelAdapter):
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def _call(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("pip install openai to use OpenAIAdapter") from e

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        usage = resp.usage
        price = PRICES.get(self.model, (0.0, 0.0))
        cost = (usage.prompt_tokens * price[0] + usage.completion_tokens * price[1]) / 1e6
        return GenerationResult(
            text=(resp.choices[0].message.content or "").strip(),
            model=self.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=cost,
        )
