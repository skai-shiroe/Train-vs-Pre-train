"""Descriptive statistics of a corpus split.

The statistics are written next to the frozen corpus and reused in the
documentation. They justify the truncation lengths: choosing 512 source tokens
without knowing the length distribution would be arbitrary.
"""

from __future__ import annotations

import statistics as stdlib_statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from src.data.example import Example


@dataclass(frozen=True, slots=True)
class LengthStatistics:
    """Distribution summary of a length series.

    Attributes:
        mean: Arithmetic mean.
        median: Median value.
        p95: 95th percentile.
        minimum: Smallest observed value.
        maximum: Largest observed value.
    """

    mean: float
    median: float
    p95: float
    minimum: int
    maximum: int


@dataclass(frozen=True, slots=True)
class SplitStatistics:
    """Statistics of one split.

    Attributes:
        split: Name of the split.
        size: Number of examples.
        source_chars: Document length distribution, in characters.
        target_chars: Summary length distribution, in characters.
        source_words: Document length distribution, in whitespace separated words.
        target_words: Summary length distribution, in whitespace separated words.
        compression_ratio: Mean ratio of summary words over document words.
    """

    split: str
    size: int
    source_chars: LengthStatistics
    target_chars: LengthStatistics
    source_words: LengthStatistics
    target_words: LengthStatistics
    compression_ratio: float

    def to_dict(self) -> dict[str, Any]:
        """Render the statistics as a JSON serialisable mapping.

        Returns:
            A nested mapping mirroring the dataclass fields.
        """
        return asdict(self)


def percentile(values: Sequence[int], fraction: float) -> float:
    """Return a percentile using linear interpolation.

    The standard library offers ``quantiles`` but it requires at least two
    points, which makes it unusable on the degenerate splits used in tests.

    Args:
        values: The observed values, in any order.
        fraction: Requested percentile, between 0 and 1.

    Returns:
        The interpolated percentile.

    Raises:
        ValueError: If ``values`` is empty or ``fraction`` is out of range.
    """
    if not values:
        raise ValueError("Cannot compute a percentile of an empty series.")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"Fraction must lie in [0, 1], got {fraction}.")

    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])

    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarise_lengths(values: Sequence[int]) -> LengthStatistics:
    """Summarise a series of lengths.

    Args:
        values: The observed lengths.

    Returns:
        The distribution summary.

    Raises:
        ValueError: If the series is empty.
    """
    if not values:
        raise ValueError("Cannot summarise an empty length series.")

    return LengthStatistics(
        mean=round(stdlib_statistics.fmean(values), 2),
        median=round(stdlib_statistics.median(values), 2),
        p95=round(percentile(values, 0.95), 2),
        minimum=min(values),
        maximum=max(values),
    )


def compute_statistics(split: str, examples: Sequence[Example]) -> SplitStatistics:
    """Compute the descriptive statistics of a split.

    Args:
        split: Name of the split.
        examples: Examples to describe.

    Returns:
        The statistics of the split.

    Raises:
        ValueError: If the split is empty.
    """
    if not examples:
        raise ValueError(f"Split {split!r} is empty, nothing to describe.")

    source_chars = [len(example.source) for example in examples]
    target_chars = [len(example.target) for example in examples]
    source_words = [len(example.source.split()) for example in examples]
    target_words = [len(example.target.split()) for example in examples]

    ratios = [
        target / source for target, source in zip(target_words, source_words, strict=True) if source
    ]

    return SplitStatistics(
        split=split,
        size=len(examples),
        source_chars=summarise_lengths(source_chars),
        target_chars=summarise_lengths(target_chars),
        source_words=summarise_lengths(source_words),
        target_words=summarise_lengths(target_words),
        compression_ratio=round(stdlib_statistics.fmean(ratios), 4) if ratios else 0.0,
    )
