"""Unit tests for the descriptive statistics of a split."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.data.example import Example
from src.data.statistics import compute_statistics, percentile, summarise_lengths


@pytest.mark.unit
@pytest.mark.parametrize(
    ("values", "fraction", "expected"),
    [
        ([1, 2, 3, 4, 5], 0.0, 1.0),
        ([1, 2, 3, 4, 5], 1.0, 5.0),
        ([1, 2, 3, 4, 5], 0.5, 3.0),
        ([10], 0.95, 10.0),
        ([1, 2], 0.5, 1.5),
    ],
)
def test_percentile_interpolates(values: list[int], fraction: float, expected: float) -> None:
    assert percentile(values, fraction) == pytest.approx(expected)


@pytest.mark.unit
def test_percentile_ignores_the_input_order() -> None:
    assert percentile([5, 1, 3, 2, 4], 0.5) == percentile([1, 2, 3, 4, 5], 0.5)


@pytest.mark.unit
def test_percentile_of_an_empty_series_is_refused() -> None:
    with pytest.raises(ValueError, match="empty series"):
        percentile([], 0.5)


@pytest.mark.unit
@pytest.mark.parametrize("fraction", [-0.1, 1.1])
def test_an_out_of_range_fraction_is_refused(fraction: float) -> None:
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        percentile([1, 2, 3], fraction)


@pytest.mark.unit
def test_summarise_lengths_reports_the_expected_fields() -> None:
    summary = summarise_lengths([1, 2, 3, 4, 100])

    assert summary.minimum == 1
    assert summary.maximum == 100
    assert summary.median == 3
    assert summary.mean == pytest.approx(22.0)


@pytest.mark.unit
def test_summarise_lengths_refuses_an_empty_series() -> None:
    with pytest.raises(ValueError, match="empty length series"):
        summarise_lengths([])


@pytest.mark.unit
def test_statistics_count_characters_and_words() -> None:
    examples = [
        Example("a", "un deux trois", "un"),
        Example("b", "un deux trois quatre", "un deux"),
    ]

    statistics = compute_statistics("train", examples)

    assert statistics.split == "train"
    assert statistics.size == 2
    assert statistics.source_words.minimum == 3
    assert statistics.source_words.maximum == 4
    assert statistics.target_words.minimum == 1
    assert statistics.source_chars.minimum == len("un deux trois")


@pytest.mark.unit
def test_compression_ratio_is_the_mean_of_the_per_example_ratios() -> None:
    examples = [
        Example("a", "a b c d", "a"),
        Example("b", "a b", "a"),
    ]

    statistics = compute_statistics("train", examples)

    assert statistics.compression_ratio == pytest.approx((0.25 + 0.5) / 2)


@pytest.mark.unit
def test_statistics_of_an_empty_split_are_refused() -> None:
    with pytest.raises(ValueError, match="is empty"):
        compute_statistics("train", [])


@pytest.mark.unit
def test_statistics_serialise_to_a_nested_mapping(
    make_examples: Callable[[int], list[Example]],
) -> None:
    payload = compute_statistics("train", make_examples(5)).to_dict()

    assert payload["split"] == "train"
    assert payload["size"] == 5
    assert set(payload["source_chars"]) == {"mean", "median", "p95", "minimum", "maximum"}
