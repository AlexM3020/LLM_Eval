"""AG News topic classification — the 'second task' that proves the abstraction.

Intentionally thin: if adding this file required touching the runner, models,
or API, the abstraction would have leaked. It doesn't.
"""
from __future__ import annotations

import csv

from ..registry import TASK_REGISTRY
from .base import Example, Prediction, Task

LABELS = ["World", "Sports", "Business", "Sci/Tech"]

PROMPT_TEMPLATE = """Classify the news article into exactly one of: {labels}.
Reply with only the label.

Article: {text}

Label:"""


@TASK_REGISTRY.register("agnews")
class AGNewsTask(Task):
    name = "agnews"
    description = "4-class news topic classification (AG News)."

    def load_dataset(self, split="test", limit=None, path=None):
        if path:  # CSV with header: label,text
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            examples = [
                Example(id=str(i), inputs={"text": r["text"]}, references=[r["label"]])
                for i, r in enumerate(rows)
            ]
        else:
            # TODO(Day 6): datasets.load_dataset("ag_news", split=split)
            #   map integer labels to LABELS[i]
            raise NotImplementedError(
                "Pass path= to a CSV, or implement the HuggingFace loader (see TODO)."
            )
        return examples[:limit] if limit else examples

    def build_prompt(self, example: Example) -> str:
        return PROMPT_TEMPLATE.format(labels=", ".join(LABELS), text=example.inputs["text"])

    def parse_prediction(self, raw_output: str) -> str:
        out = raw_output.strip().lower()
        for label in LABELS:
            if label.lower() in out:
                return label
        return raw_output.strip()  # unmatched outputs score as wrong; never crash

    def score_example(self, prediction: Prediction, example: Example) -> dict[str, float]:
        return {"accuracy": float(prediction.parsed == example.references[0])}
