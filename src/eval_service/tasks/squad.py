"""SQuAD-style extractive QA — the proof-of-concept task.

load_dataset supports a local JSONL file with one object per line:
    {"id": ..., "question": ..., "context": ..., "answers": [...]}
so the project runs fully offline; pass path=None to pull from HuggingFace instead.

Metric helpers below are the standard SQuAD v1 normalization + EM/F1.
"""
from __future__ import annotations

import json
import re
import string
from collections import Counter

from ..registry import TASK_REGISTRY
from .base import Example, Prediction, Task

PROMPT_TEMPLATE = """Answer the question using only the context below. Give the shortest exact span from the context that answers it.

Context:
{context}

Question: {question}

Answer:"""


# --- SQuAD v1 metric helpers (standard normalization) ---

def normalize_answer(s: str) -> str:
    def remove_articles(t: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def remove_punc(t: str) -> str:
        return "".join(ch for ch in t if ch not in set(string.punctuation))

    return " ".join(remove_articles(remove_punc(s.lower())).split())


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1(prediction: str, ground_truth: str) -> float:
    pred_toks = normalize_answer(prediction).split()
    gt_toks = normalize_answer(ground_truth).split()
    num_same = sum((Counter(pred_toks) & Counter(gt_toks)).values())
    if not pred_toks or not gt_toks:
        return float(pred_toks == gt_toks)
    if num_same == 0:
        return 0.0
    precision, recall = num_same / len(pred_toks), num_same / len(gt_toks)
    return 2 * precision * recall / (precision + recall)


def best_over_references(metric_fn, prediction: str, references: list[str]) -> float:
    return max((metric_fn(prediction, ref) for ref in references), default=0.0)


@TASK_REGISTRY.register("squad")
class SquadTask(Task):
    name = "squad"
    description = "Extractive question answering over a given context (SQuAD v1-style)."

    def load_dataset(self, split="validation", limit=None, path=None):
        if path:
            with open(path) as f:
                examples = [self._from_json(json.loads(line)) for line in f if line.strip()]
        else:
            # TODO(Day 2): from datasets import load_dataset
            #   ds = load_dataset("squad", split=split)
            #   examples = [self._from_json({**r, "answers": r["answers"]["text"]}) for r in ds]
            raise NotImplementedError(
                "Pass path= to a JSONL file, or implement the HuggingFace loader (see TODO)."
            )
        return examples[:limit] if limit else examples

    @staticmethod
    def _from_json(row: dict) -> Example:
        return Example(
            id=str(row["id"]),
            inputs={"question": row["question"], "context": row["context"]},
            references=list(row["answers"]),
        )

    def build_prompt(self, example: Example) -> str:
        return PROMPT_TEMPLATE.format(**example.inputs)

    def parse_prediction(self, raw_output: str) -> str:
        return raw_output.strip().split("\n")[0].strip()

    def score_example(self, prediction: Prediction, example: Example) -> dict[str, float]:
        return {
            "exact_match": best_over_references(exact_match, prediction.parsed, example.references),
            "f1": best_over_references(f1, prediction.parsed, example.references),
        }
