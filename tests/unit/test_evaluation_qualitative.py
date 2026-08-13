"""Unit tests of the qualitative selection."""

from __future__ import annotations

import json

import pytest

from src.data.example import Example
from src.evaluation.qualitative import (
    ScoredPrediction,
    score_predictions,
    select_qualitative,
)
from src.metrics.rouge import RougeConfig, score_all

NO_BOOTSTRAP = RougeConfig(bootstrap_samples=0)


def make_scored(count: int) -> list[ScoredPrediction]:
    """Build records whose ROUGE-L rises with the index, so the ranking is known."""
    return [
        ScoredPrediction(
            example_id=f"id-{index:02d}",
            source=f"document {index}",
            reference=f"reference {index}",
            prediction=f"prediction {index}",
            scores={"rougeL": index / 100.0},
        )
        for index in range(count)
    ]


# ---------------------------------------------------------------------------
# Building the records
# ---------------------------------------------------------------------------


def test_each_prediction_is_attached_to_its_example_and_its_score() -> None:
    examples = [Example(example_id="a", source="doc a", target="the cat sat on the mat")]
    predictions = ["the cat sat"]
    per_example = score_all(predictions, ["the cat sat on the mat"], NO_BOOTSTRAP)

    scored = score_predictions(examples, predictions, per_example)

    assert scored[0].example_id == "a"
    assert scored[0].source == "doc a"
    assert scored[0].reference == "the cat sat on the mat"
    assert scored[0].prediction == "the cat sat"
    assert scored[0].scores["rouge1"] == pytest.approx(2 / 3)


def test_misaligned_inputs_are_refused() -> None:
    examples = [Example(example_id="a", source="doc", target="ref")]

    with pytest.raises(ValueError, match="aligned"):
        score_predictions(examples, ["one", "two"], [])


# ---------------------------------------------------------------------------
# Choosing what to read
# ---------------------------------------------------------------------------


def test_the_best_and_the_worst_come_out_at_their_own_ends() -> None:
    selection = select_qualitative(make_scored(20), best=3, worst=3, sample=0)

    assert [item.example_id for item in selection.best] == ["id-19", "id-18", "id-17"]
    assert [item.example_id for item in selection.worst] == ["id-00", "id-01", "id-02"]


def test_the_three_groups_never_share_an_example() -> None:
    selection = select_qualitative(make_scored(12), best=3, worst=3, sample=4)

    chosen = [
        item.example_id
        for group in (selection.best, selection.worst, selection.sample)
        for item in group
    ]
    assert len(chosen) == len(set(chosen)) == 10


def test_the_random_draw_reaches_the_middle_of_the_distribution() -> None:
    """Reading only the two ends would describe a model nobody uses."""
    selection = select_qualitative(make_scored(20), best=3, worst=3, sample=4)

    extremes = {"id-19", "id-18", "id-17", "id-00", "id-01", "id-02"}
    assert {item.example_id for item in selection.sample}.isdisjoint(extremes)
    assert len(selection.sample) == 4


def test_the_same_seed_selects_the_same_examples() -> None:
    first = select_qualitative(make_scored(20), sample=4, seed=7)
    second = select_qualitative(make_scored(20), sample=4, seed=7)
    other = select_qualitative(make_scored(20), sample=4, seed=8)

    assert [item.example_id for item in first.sample] == [item.example_id for item in second.sample]
    assert [item.example_id for item in first.sample] != [item.example_id for item in other.sample]


def test_ties_break_on_the_identifier() -> None:
    tied = [
        ScoredPrediction(
            example_id=identifier,
            source="doc",
            reference="ref",
            prediction="pred",
            scores={"rougeL": 0.5},
        )
        for identifier in ("c", "a", "b")
    ]

    selection = select_qualitative(tied, best=2, worst=0, sample=0)

    assert [item.example_id for item in selection.best] == ["a", "b"]


def test_asking_for_no_worst_examples_returns_none_of_them() -> None:
    """A slice of the last zero items is the whole list, not an empty one."""
    selection = select_qualitative(make_scored(10), best=2, worst=0, sample=0)

    assert selection.worst == ()


def test_a_split_smaller_than_the_request_shrinks_the_groups() -> None:
    selection = select_qualitative(make_scored(4), best=3, worst=3, sample=4)

    assert len(selection.best) == 3
    assert len(selection.worst) == 1
    assert selection.sample == ()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"best": -1}, "non negative"),
        ({"worst": -1}, "non negative"),
        ({"sample": -1}, "non negative"),
        ({"variant": "rouge9"}, "not scored"),
    ],
)
def test_an_invalid_selection_request_is_refused(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        select_qualitative(make_scored(5), **kwargs)  # type: ignore[arg-type]


def test_an_empty_split_cannot_be_sampled() -> None:
    with pytest.raises(ValueError, match="empty split"):
        select_qualitative([])


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_the_selection_carries_the_documents_it_selected() -> None:
    payload = select_qualitative(make_scored(10), best=2, worst=2, sample=2).to_dict()

    assert payload["variant"] == "rougeL"
    assert payload["best"][0]["source"] == "document 9"
    assert json.dumps(payload)


def test_a_per_example_record_can_leave_the_document_out() -> None:
    """Repeating a thousand documents would make the artefact larger than the split."""
    record = make_scored(1)[0]

    assert "source" not in record.to_dict(include_source=False)
    assert record.to_dict()["source"] == "document 0"
