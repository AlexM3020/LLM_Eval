"""LLM Eval Service — pluggable evaluation for LLM applications."""

# Importing the subpackages registers all built-in models, tasks, and evaluators.
from . import evaluators, models, tasks  # noqa: F401

__version__ = "0.1.0"
