"""Unit tests of the ROUGE layer.

The scoring itself belongs to ``rouge-score`` and is not retested here. What is
tested is everything this project owns around it: the order the reference and
the prediction are passed in, what happens to a failed generation, how the
per example scores become a corpus score, and whether the confidence interval
is reproducible.
"""

from __future__ import annotations

import json

import pytest

from src.metrics.rouge import (
    REPORTED_VARIANT,
    ROUGE_VARIANTS,
    ConfidenceInterval,
    CorpusRouge,
    RougeConfig,
    RougeScore,
    aggregate,
    score_all,
    score_corpus,
    score_example,
)

NO_BOOTSTRAP = RougeConfig(bootstrap_samples=0)


# ---------------------------------------------------------------------------
# Argument order
# ---------------------------------------------------------------------------


def test_a_short_prediction_scores_full_precision_and_partial_recall() -> None:
    """Pin the argument order of the underlying scorer.

    The F-measure is unchanged by swapping the reference and the prediction, so
    a swap survives any assertion written on F alone while precision and recall
    trade places. This is the assertion that catches it.
    """
    scores = score_example("the cat sat", "the cat sat on the mat", NO_BOOTSTRAP)

    assert scores["rouge1"].precision == pytest.approx(1.0)
    assert scores["rouge1"].recall == pytest.approx(0.5)
    assert scores["rouge1"].fmeasure == pytest.approx(2 / 3)


def test_the_unigram_and_bigram_scores_match_a_hand_computation() -> None:
    scores = score_example("the cat sat", "the cat sat on the mat", NO_BOOTSTRAP)

    # Two prediction bigrams, five reference bigrams, both prediction bigrams
    # present in the reference.
    assert scores["rouge2"].precision == pytest.approx(1.0)
    assert scores["rouge2"].recall == pytest.approx(0.4)
    # The longest common subsequence is the whole prediction.
    assert scores["rougeL"].recall == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Degenerate summaries
# ---------------------------------------------------------------------------


def test_an_identical_summary_scores_one_everywhere() -> None:
    scores = score_example("the cat sat on the mat", "the cat sat on the mat", NO_BOOTSTRAP)

    for variant in ROUGE_VARIANTS:
        assert scores[variant].fmeasure == pytest.approx(1.0)


def test_a_disjoint_summary_scores_zero_everywhere() -> None:
    scores = score_example("quantum bicycles", "the cat sat on the mat", NO_BOOTSTRAP)

    for variant in ROUGE_VARIANTS:
        assert scores[variant].fmeasure == pytest.approx(0.0)


def test_an_empty_prediction_scores_zero_without_raising() -> None:
    scores = score_example("", "the cat sat on the mat", NO_BOOTSTRAP)

    assert scores["rouge1"].fmeasure == pytest.approx(0.0)


def test_an_empty_reference_is_refused_as_a_corpus_defect() -> None:
    with pytest.raises(ValueError, match="corpus"):
        score_example("a summary", "   ", NO_BOOTSTRAP)


def test_a_failed_generation_stays_in_the_corpus_mean() -> None:
    """Dropping the documents a model failed on would reward it for failing."""
    corpus = score_corpus(
        ["the cat sat on the mat", ""],
        ["the cat sat on the mat", "the dog slept on the rug"],
        NO_BOOTSTRAP,
    )

    assert corpus.size == 2
    assert corpus.scores["rouge1"].fmeasure == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Stemming
# ---------------------------------------------------------------------------


def test_stemming_matches_a_plural_against_its_singular() -> None:
    stemmed = score_example(
        "the council reports", "the council report", RougeConfig(bootstrap_samples=0)
    )
    raw = score_example(
        "the council reports",
        "the council report",
        RougeConfig(use_stemmer=False, bootstrap_samples=0),
    )

    assert stemmed["rouge1"].fmeasure == pytest.approx(1.0)
    assert raw["rouge1"].fmeasure < 1.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_the_default_configuration_measures_the_three_required_variants() -> None:
    assert RougeConfig().variants == ROUGE_VARIANTS
    assert REPORTED_VARIANT == "rougeL"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"variants": ()}, "At least one"),
        ({"variants": ("rouge1", "rouge1")}, "Duplicate"),
        ({"variants": ("rougeLsum",)}, "Unsupported"),
        ({"bootstrap_samples": -1}, "non negative"),
        ({"confidence": 0.0}, "confidence"),
        ({"confidence": 1.0}, "confidence"),
        ({"seed": -1}, "seed"),
    ],
)
def test_an_invalid_measurement_setting_is_refused(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RougeConfig(**kwargs)  # type: ignore[arg-type]


def test_the_configuration_renders_as_a_json_serialisable_mapping() -> None:
    payload = RougeConfig().to_dict()

    assert payload["variants"] == list(ROUGE_VARIANTS)
    assert json.dumps(payload)


# ---------------------------------------------------------------------------
# Scoring a whole split
# ---------------------------------------------------------------------------


def test_misaligned_predictions_and_references_are_refused() -> None:
    with pytest.raises(ValueError, match="aligned"):
        score_all(["one"], ["one", "two"], NO_BOOTSTRAP)


def test_an_empty_split_cannot_be_scored() -> None:
    with pytest.raises(ValueError, match="empty corpus"):
        score_all([], [], NO_BOOTSTRAP)


def test_the_corpus_score_is_the_mean_of_the_per_example_scores() -> None:
    per_example = score_all(
        ["the cat sat on the mat", "the cat sat"],
        ["the cat sat on the mat", "the cat sat on the mat"],
        NO_BOOTSTRAP,
    )
    corpus = aggregate(per_example, NO_BOOTSTRAP)

    expected = (1.0 + 2 / 3) / 2
    assert corpus.scores["rouge1"].fmeasure == pytest.approx(expected)
    assert corpus.reported == pytest.approx(expected)


def test_an_empty_score_list_cannot_be_aggregated() -> None:
    with pytest.raises(ValueError, match="empty score list"):
        aggregate([], NO_BOOTSTRAP)


def test_aggregating_a_variant_that_was_never_scored_is_refused() -> None:
    per_example = score_all(["a summary"], ["a summary"], RougeConfig(variants=("rouge1",)))

    with pytest.raises(ValueError, match="never scored"):
        aggregate(per_example, NO_BOOTSTRAP)


def test_the_reported_metric_is_absent_when_rouge_l_was_not_measured() -> None:
    corpus = CorpusRouge(size=1, scores={"rouge1": RougeScore(1.0, 1.0, 1.0)}, intervals={})

    with pytest.raises(KeyError, match="section 2.1"):
        _ = corpus.reported


# ---------------------------------------------------------------------------
# Confidence interval
# ---------------------------------------------------------------------------


def test_the_interval_brackets_the_mean_and_is_reproducible() -> None:
    predictions = ["the cat sat on the mat", "the cat sat", "quantum bicycles"]
    references = ["the cat sat on the mat"] * 3
    config = RougeConfig(bootstrap_samples=200)

    first = score_corpus(predictions, references, config)
    second = score_corpus(predictions, references, config)

    interval = first.intervals[REPORTED_VARIANT]
    assert interval.low <= first.reported <= interval.high
    assert interval.confidence == pytest.approx(0.95)
    assert second.intervals[REPORTED_VARIANT] == interval


def test_a_different_seed_moves_the_interval() -> None:
    predictions = ["the cat sat on the mat", "the cat sat", "quantum bicycles"]
    references = ["the cat sat on the mat"] * 3

    first = score_corpus(predictions, references, RougeConfig(bootstrap_samples=200, seed=1))
    second = score_corpus(predictions, references, RougeConfig(bootstrap_samples=200, seed=2))

    assert first.reported == pytest.approx(second.reported)
    assert first.intervals[REPORTED_VARIANT] != second.intervals[REPORTED_VARIANT]


def test_the_interval_can_be_switched_off() -> None:
    corpus = score_corpus(["the cat"], ["the cat"], NO_BOOTSTRAP)

    assert corpus.intervals == {}


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_the_corpus_score_renders_as_a_json_serialisable_mapping() -> None:
    corpus = score_corpus(["the cat sat"], ["the cat sat on the mat"], RougeConfig())
    payload = corpus.to_dict()

    assert payload["size"] == 1
    assert set(payload["scores"]) == set(ROUGE_VARIANTS)
    assert set(payload["intervals"]) == set(ROUGE_VARIANTS)
    assert json.dumps(payload)


def test_a_score_and_an_interval_render_as_flat_mappings() -> None:
    assert RougeScore(1.0, 0.5, 0.75).to_dict() == {
        "precision": 1.0,
        "recall": 0.5,
        "fmeasure": 0.75,
    }
    assert ConfidenceInterval(0.1, 0.2, 0.95).to_dict() == {
        "low": 0.1,
        "high": 0.2,
        "confidence": 0.95,
    }
