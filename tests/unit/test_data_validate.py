"""Unit tests for corpus validation, including the leakage guard."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.data.example import Example
from src.data.validate import (
    ValidationError,
    require_clean,
    require_no_leakage,
    validate_split,
)


@pytest.mark.unit
def test_a_clean_split_reports_no_problem(make_examples: Callable[[int], list[Example]]) -> None:
    report = validate_split("train", make_examples(10))

    assert report.total == 10
    assert report.is_clean is True
    assert report.duplicate_ids == ()


@pytest.mark.unit
def test_empty_fields_are_counted() -> None:
    report = validate_split(
        "train",
        [
            Example("a", "", "resume"),
            Example("b", "document", "   "),
            Example("c", "document c", "resume c"),
        ],
    )

    assert report.empty_source == 1
    assert report.empty_target == 1
    assert report.is_clean is False


@pytest.mark.unit
def test_duplicate_identifiers_are_listed() -> None:
    report = validate_split(
        "train",
        [Example("dup", "a a a", "x"), Example("dup", "b b b", "y"), Example("ok", "c c", "z")],
    )

    assert report.duplicate_ids == ("dup",)
    assert report.is_clean is False


@pytest.mark.unit
def test_duplicate_documents_are_counted_without_blocking() -> None:
    report = validate_split(
        "train",
        [Example("a", "same document", "x"), Example("b", "same document", "y")],
    )

    assert report.duplicate_sources == 1
    assert report.is_clean is True


@pytest.mark.unit
def test_summaries_longer_than_their_document_are_reported_not_blocked() -> None:
    report = validate_split("train", [Example("a", "court", "un resume bien plus long")])

    assert report.target_longer_than_source == 1
    assert report.is_clean is True


@pytest.mark.unit
def test_describe_mentions_every_counter(make_examples: Callable[[int], list[Example]]) -> None:
    description = validate_split("test", make_examples(3)).describe()

    assert "test: 3 examples" in description
    assert "empty documents" in description
    assert "duplicate ids" in description


@pytest.mark.unit
def test_require_clean_accepts_a_clean_split(
    make_examples: Callable[[int], list[Example]],
) -> None:
    require_clean(validate_split("train", make_examples(5)))


@pytest.mark.unit
def test_require_clean_names_the_problems() -> None:
    report = validate_split("train", [Example("a", "", "")])

    with pytest.raises(ValidationError, match="empty documents"):
        require_clean(report)


@pytest.mark.unit
def test_no_leakage_passes_on_disjoint_splits(
    make_examples: Callable[[int], list[Example]],
) -> None:
    everything = make_examples(30)

    require_no_leakage(everything[:20], everything[20:25], everything[25:])


@pytest.mark.unit
def test_shared_identifier_between_train_and_test_is_fatal(
    make_examples: Callable[[int], list[Example]],
) -> None:
    everything = make_examples(10)

    with pytest.raises(ValidationError, match="shared between train and test"):
        require_no_leakage(everything[:5], everything[5:8], everything[4:6])


@pytest.mark.unit
def test_shared_document_under_a_different_identifier_is_fatal() -> None:
    document = "un document identique dans deux splits"
    train = [Example("train-1", document, "resume a")]
    test = [Example("test-1", document, "resume b")]

    with pytest.raises(ValidationError, match="document\\(s\\) shared between train and test"):
        require_no_leakage(train, [], test)


@pytest.mark.unit
def test_leakage_between_validation_and_test_is_fatal(
    make_examples: Callable[[int], list[Example]],
) -> None:
    everything = make_examples(10)

    with pytest.raises(ValidationError, match="shared between validation and test"):
        require_no_leakage(everything[:4], everything[4:7], everything[6:9])
