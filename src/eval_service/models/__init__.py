from .base import GenerationConfig, GenerationResult, ModelAdapter
from . import mock, openai  # noqa: F401  (importing registers the adapters)

__all__ = ["GenerationConfig", "GenerationResult", "ModelAdapter"]
