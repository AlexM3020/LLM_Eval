"""Known-value tests for the SQuAD metric helpers and task plumbing."""
import os

from eval_service.tasks.base import Example, Prediction
from eval_service.tasks.squad import (
    SquadTask,
    best_over_references,
    exact_match,
    f1,
    normalize_answer,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "squad_tiny.jsonl")


def test_normalize_strips_articles_punct_case():
    assert normalize_answer("The Eiffel Tower!") == "eiffel tower"


def test_exact_match_ignores_articles():
    assert exact_match("The New York City", "New York City") == 1.0
    assert exact_match("Paris", "New York City") == 0.0


def test_f1_partial_overlap():
    # pred tokens {broadcast, antenna}, gt tokens {a->removed, broadcast, antenna}
    assert f1("broadcast antenna", "a broadcast antenna") == 1.0
    assert 0.0 < f1("the tall broadcast antenna", "a broadcast antenna") < 1.0
    assert f1("nothing alike", "a broadcast antenna") == 0.0


def test_best_over_references():
    assert best_over_references(exact_match, "x", ["y", "x"]) == 1.0
    assert best_over_references(exact_match, "x", []) == 0.0


def test_parse_prediction_takes_first_line():
    task = SquadTask()
    assert task.parse_prediction("  1457\nSome explanation") == "1457"


def test_score_example_perfect_and_wrong():
    task = SquadTask()
    ex = Example(id="1", inputs={"question": "q", "context": "c"}, references=["1457"])
    good = task.score_example(Prediction(example_id="1", raw_output="", parsed="1457"), ex)
    bad = task.score_example(Prediction(example_id="1", raw_output="", parsed="1900"), ex)
    assert good == {"exact_match": 1.0, "f1": 1.0}
    assert bad == {"exact_match": 0.0, "f1": 0.0}


def test_load_dataset_from_jsonl():
    task = SquadTask()
    examples = task.load_dataset(path=FIXTURE, limit=2)
    assert len(examples) == 2
    assert examples[0].references == ["New York City"]
