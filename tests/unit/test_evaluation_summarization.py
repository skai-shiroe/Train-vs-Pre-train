"""Unit tests of the summarisation report."""

from __future__ import annotations

import json

import pytest

from src.evaluation.summarization import evaluate_summaries
from src.metrics.rouge import RougeConfig, RougeScore, score_all

NO_BOOTSTRAP = RougeConfig(bootstrap_samples=0)

REFERENCES = [
    "the cat sat on the mat",
    "the dog slept on the rug",
    "the bird sang in the tree",
]


def test_the_report_carries_the_corpus_score_and_the_split_size() -> None:
    report = evaluate_summaries(REFERENCES, REFERENCES, rouge_config=NO_BOOTSTRAP)

    assert report.size == 3
    assert report.reported == pytest.approx(1.0)
    assert report.rouge.size == 3


def test_a_failed_generation_is_counted_next_to_the_score() -> None:
    """A mean of zero over three documents reads differently from three failures."""
    report = evaluate_summaries(["", "  ", REFERENCES[2]], REFERENCES, rouge_config=NO_BOOTSTRAP)

    assert report.empty_predictions == 2
    assert report.reported == pytest.approx(1 / 3)


def test_the_generated_lengths_sit_next_to_the_reference_lengths() -> None:
    """A model that emits four words where references hold twenty is broken, not weak."""
    report = evaluate_summaries(["one two"] * 3, REFERENCES, rouge_config=NO_BOOTSTRAP)

    assert report.prediction_words.mean == pytest.approx(2.0)
    assert report.reference_words.mean == pytest.approx(6.0)
    assert report.prediction_words.minimum == 2
    assert report.reference_words.maximum == 6


def test_precomputed_scores_are_reused_rather_than_recomputed() -> None:
    """The evaluator scores once and hands the result over; the report must trust it."""
    fabricated = [{"rougeL": RougeScore(1.0, 1.0, 1.0)} for _ in REFERENCES]

    report = evaluate_summaries(
        ["quantum bicycles"] * 3,
        REFERENCES,
        per_example=fabricated,
        rouge_config=RougeConfig(variants=("rougeL",), bootstrap_samples=0),
    )

    assert report.reported == pytest.approx(1.0)


def test_scores_that_do_not_cover_every_example_are_refused() -> None:
    per_example = score_all(REFERENCES[:1], REFERENCES[:1], NO_BOOTSTRAP)

    with pytest.raises(ValueError, match="scored examples"):
        evaluate_summaries(REFERENCES, REFERENCES, per_example=per_example)


def test_misaligned_predictions_and_references_are_refused() -> None:
    with pytest.raises(ValueError, match="aligned"):
        evaluate_summaries(REFERENCES[:2], REFERENCES, rouge_config=NO_BOOTSTRAP)


def test_an_empty_split_cannot_be_reported_on() -> None:
    with pytest.raises(ValueError, match="empty split"):
        evaluate_summaries([], [], rouge_config=NO_BOOTSTRAP)


def test_the_report_renders_as_a_json_serialisable_mapping() -> None:
    report = evaluate_summaries(REFERENCES, REFERENCES, rouge_config=RougeConfig())
    payload = report.to_dict()

    assert payload["size"] == 3
    assert payload["empty_predictions"] == 0
    assert payload["prediction_words"]["mean"] == pytest.approx(6.0)
    assert set(payload["rouge"]["scores"]) == {"rouge1", "rouge2", "rougeL"}
    assert json.dumps(payload)
