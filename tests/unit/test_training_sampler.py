"""Unit tests for the length grouped sampler.

The sampler exists to cut the padding waste left by dynamic padding. The test
that matters is therefore not that it runs, but that it actually reduces the
waste while keeping the schedule reproducible.

The lengths are drawn to match CNN/DailyMail rather than spread evenly: 85
percent of the articles reach the 512 token cap, which is what makes the waste
small before grouping and is measured in the module under test. A uniform draw
would hand the sampler an easy corpus and let a regression that only shows up
on a saturated one through.
"""

from __future__ import annotations

import random

import pytest

from src.training.sampler import LengthGroupedSampler, padding_waste

BATCH_SIZE = 8


def skewed_lengths(count: int, *, seed: int = 7) -> list[int]:
    """Return lengths shaped like the CNN/DailyMail training split."""
    generator = random.Random(seed)
    return [512 if generator.random() < 0.85 else generator.randint(69, 511) for _ in range(count)]


def random_batches(count: int, batch_size: int, *, seed: int = 0) -> list[list[int]]:
    """Return the batches a plain shuffling sampler would produce."""
    order = list(range(count))
    random.Random(seed).shuffle(order)
    return [order[start : start + batch_size] for start in range(0, count, batch_size)]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(("batch_size", "factor"), [(0, 50), (-1, 50), (8, 0)])
def test_non_positive_sizes_are_refused(batch_size: int, factor: int) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        LengthGroupedSampler([1, 2, 3], batch_size, seed=42, mega_batch_factor=factor)


@pytest.mark.unit
def test_an_empty_corpus_yields_no_batch() -> None:
    sampler = LengthGroupedSampler([], BATCH_SIZE, seed=42)

    assert list(sampler) == []


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_every_example_appears_exactly_once() -> None:
    lengths = skewed_lengths(200)
    sampler = LengthGroupedSampler(lengths, BATCH_SIZE, seed=42, mega_batch_factor=4)

    seen = [index for batch in sampler for index in batch]

    assert sorted(seen) == list(range(200))


@pytest.mark.unit
def test_the_reported_length_matches_the_batches_produced() -> None:
    # The DataLoader trusts __len__ to plan the epoch, so a mismatch would make
    # the step count of the scheduler wrong.
    lengths = skewed_lengths(203)
    sampler = LengthGroupedSampler(lengths, BATCH_SIZE, seed=42)

    assert len(sampler) == len(list(sampler))


@pytest.mark.unit
def test_dropping_the_last_batch_removes_only_the_incomplete_one() -> None:
    lengths = skewed_lengths(203)
    sampler = LengthGroupedSampler(lengths, BATCH_SIZE, seed=42, drop_last=True)

    batches = list(sampler)

    assert len(sampler) == len(batches) == 203 // BATCH_SIZE
    assert all(len(batch) == BATCH_SIZE for batch in batches)


@pytest.mark.unit
def test_a_corpus_smaller_than_a_batch_yields_one_batch() -> None:
    sampler = LengthGroupedSampler([10, 20, 30], BATCH_SIZE, seed=42)

    assert [len(batch) for batch in sampler] == [3]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_same_seed_and_epoch_replay_the_same_batches() -> None:
    lengths = skewed_lengths(300)

    first = LengthGroupedSampler(lengths, BATCH_SIZE, seed=42)
    second = LengthGroupedSampler(lengths, BATCH_SIZE, seed=42)
    first.set_epoch(3)
    second.set_epoch(3)

    assert list(first) == list(second)


@pytest.mark.unit
def test_the_batching_ignores_the_global_generator() -> None:
    # A sampler that drew from the global generator would produce different
    # batches depending on how much randomness the rest of the run consumed.
    lengths = skewed_lengths(300)
    sampler = LengthGroupedSampler(lengths, BATCH_SIZE, seed=42)

    before = list(sampler)
    random.random()
    after = list(sampler)

    assert before == after


@pytest.mark.unit
def test_a_new_epoch_reshuffles() -> None:
    lengths = skewed_lengths(300)
    sampler = LengthGroupedSampler(lengths, BATCH_SIZE, seed=42)

    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)

    assert first != second


@pytest.mark.unit
def test_a_different_seed_produces_a_different_schedule() -> None:
    lengths = skewed_lengths(300)

    assert list(LengthGroupedSampler(lengths, BATCH_SIZE, seed=1)) != list(
        LengthGroupedSampler(lengths, BATCH_SIZE, seed=2)
    )


@pytest.mark.unit
def test_shuffling_off_keeps_the_corpus_order_between_windows() -> None:
    lengths = list(range(40))
    sampler = LengthGroupedSampler(
        lengths, batch_size=4, seed=42, mega_batch_factor=1, shuffle=False
    )

    batches = list(sampler)

    # One window is one batch here, so each batch holds four consecutive
    # indices, sorted by decreasing length inside the batch.
    assert batches[0] == [3, 2, 1, 0]
    assert batches[-1] == [39, 38, 37, 36]


# ---------------------------------------------------------------------------
# The point of the sampler
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_grouping_cuts_the_padding_waste() -> None:
    lengths = skewed_lengths(2000)

    grouped = padding_waste(lengths, LengthGroupedSampler(lengths, BATCH_SIZE, seed=42))
    shuffled = padding_waste(lengths, random_batches(len(lengths), BATCH_SIZE))

    # Both bounds are loose on purpose. What the saturated corpus leaves to
    # recover is a few percent, not the quarter a spread out corpus would, so
    # the assertion that carries the meaning is the ratio: a sampler that
    # stopped grouping would land on the shuffled figure.
    assert shuffled > 0.05
    assert grouped < shuffled / 3


@pytest.mark.unit
def test_a_wider_window_groups_more_tightly() -> None:
    lengths = skewed_lengths(4000)

    narrow = padding_waste(
        lengths, LengthGroupedSampler(lengths, BATCH_SIZE, seed=42, mega_batch_factor=2)
    )
    wide = padding_waste(
        lengths, LengthGroupedSampler(lengths, BATCH_SIZE, seed=42, mega_batch_factor=100)
    )

    assert wide < narrow


@pytest.mark.unit
def test_the_widest_batch_comes_first() -> None:
    # The peak memory of the epoch is then paid on the first step, so a
    # configuration that does not fit fails immediately.
    lengths = skewed_lengths(500)
    sampler = LengthGroupedSampler(lengths, BATCH_SIZE, seed=42)

    batches = list(sampler)
    widths = [max(lengths[index] for index in batch) for batch in batches]

    assert widths[0] == max(widths)


# ---------------------------------------------------------------------------
# The measurement helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_uniform_batch_wastes_nothing() -> None:
    assert padding_waste([10, 10, 10, 10], [[0, 1], [2, 3]]) == 0.0


@pytest.mark.unit
def test_the_waste_is_the_share_of_padded_positions() -> None:
    # One batch of widths 10 and 30: 40 real positions out of 60 allocated.
    assert padding_waste([10, 30], [[0, 1]]) == pytest.approx(1 / 3)


@pytest.mark.unit
def test_measuring_nothing_reports_no_waste() -> None:
    assert padding_waste([], []) == 0.0
    assert padding_waste([5], [[]]) == 0.0
