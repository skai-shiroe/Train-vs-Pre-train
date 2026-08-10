"""Unit tests for the frozen working corpus and its nested subsets."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.data.example import Example
from src.data.split import (
    build_working_corpus,
    checksum,
    proportion_subset,
    sample_split,
)


@pytest.mark.unit
def test_sampling_is_deterministic(make_examples: Callable[[int], list[Example]]) -> None:
    population = make_examples(100)

    first = sample_split(population, 20, seed=42)
    second = sample_split(population, 20, seed=42)

    assert first == second


@pytest.mark.unit
def test_a_different_seed_gives_a_different_draw(
    make_examples: Callable[[int], list[Example]],
) -> None:
    population = make_examples(100)

    assert sample_split(population, 20, seed=1) != sample_split(population, 20, seed=2)


@pytest.mark.unit
def test_sampling_does_not_consume_the_global_generator(
    make_examples: Callable[[int], list[Example]],
) -> None:
    import random

    population = make_examples(50)

    random.seed(7)
    before = random.random()
    sample_split(population, 10, seed=42)
    random.seed(7)
    after = random.random()

    assert before == after


@pytest.mark.unit
def test_the_draw_is_shuffled_not_a_head_slice(
    make_examples: Callable[[int], list[Example]],
) -> None:
    population = make_examples(200)

    drawn = sample_split(population, 50, seed=42)

    assert [example.example_id for example in drawn] != [
        example.example_id for example in population[:50]
    ]


@pytest.mark.unit
def test_the_draw_holds_no_duplicate(make_examples: Callable[[int], list[Example]]) -> None:
    drawn = sample_split(make_examples(100), 40, seed=42)

    assert len({example.example_id for example in drawn}) == 40


@pytest.mark.unit
def test_drawing_more_than_available_is_refused(
    make_examples: Callable[[int], list[Example]],
) -> None:
    with pytest.raises(ValueError, match="Cannot draw 20 examples from a population of 10"):
        sample_split(make_examples(10), 20, seed=42)


@pytest.mark.unit
@pytest.mark.parametrize("size", [0, -5])
def test_a_non_positive_size_is_refused(
    make_examples: Callable[[int], list[Example]], size: int
) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        sample_split(make_examples(10), size, seed=42)


@pytest.mark.unit
def test_working_corpus_has_the_requested_sizes(
    make_examples: Callable[[int], list[Example]],
) -> None:
    everything = make_examples(300)

    corpus = build_working_corpus(
        everything[:200],
        everything[200:250],
        everything[250:],
        train_size=40,
        validation_size=10,
        test_size=10,
        seed=42,
    )

    assert (len(corpus.train), len(corpus.validation), len(corpus.test)) == (40, 10, 10)
    assert corpus.seed == 42


@pytest.mark.unit
def test_changing_the_training_size_leaves_the_test_split_untouched(
    make_examples: Callable[[int], list[Example]],
) -> None:
    everything = make_examples(300)
    kwargs = {
        "train": everything[:200],
        "validation": everything[200:250],
        "test": everything[250:],
        "validation_size": 10,
        "test_size": 10,
        "seed": 42,
    }

    small = build_working_corpus(train_size=20, **kwargs)  # type: ignore[arg-type]
    large = build_working_corpus(train_size=80, **kwargs)  # type: ignore[arg-type]

    assert small.test == large.test
    assert small.validation == large.validation


@pytest.mark.unit
def test_splits_exposes_the_three_splits(make_examples: Callable[[int], list[Example]]) -> None:
    everything = make_examples(60)
    corpus = build_working_corpus(
        everything[:40],
        everything[40:50],
        everything[50:],
        train_size=10,
        validation_size=5,
        test_size=5,
        seed=42,
    )

    assert set(corpus.splits()) == {"train", "validation", "test"}


@pytest.mark.unit
def test_subsets_are_nested(make_examples: Callable[[int], list[Example]]) -> None:
    train = tuple(make_examples(100))

    ten = proportion_subset(train, 10)
    fifty = proportion_subset(train, 50)
    hundred = proportion_subset(train, 100)

    assert len(ten) == 10
    assert len(fifty) == 50
    assert len(hundred) == 100
    assert fifty[: len(ten)] == ten
    assert hundred[: len(fifty)] == fifty


@pytest.mark.unit
def test_a_subset_never_collapses_to_zero(make_examples: Callable[[int], list[Example]]) -> None:
    assert len(proportion_subset(tuple(make_examples(5)), 10)) == 1


@pytest.mark.unit
@pytest.mark.parametrize("percentage", [0, -10, 101])
def test_an_out_of_range_percentage_is_refused(
    make_examples: Callable[[int], list[Example]], percentage: int
) -> None:
    with pytest.raises(ValueError, match=r"must lie in \(0, 100\]"):
        proportion_subset(tuple(make_examples(10)), percentage)


@pytest.mark.unit
def test_checksum_is_stable(make_examples: Callable[[int], list[Example]]) -> None:
    examples = make_examples(20)

    assert checksum(examples) == checksum(list(examples))


@pytest.mark.unit
def test_checksum_changes_when_the_order_changes(
    make_examples: Callable[[int], list[Example]],
) -> None:
    examples = make_examples(20)

    assert checksum(examples) != checksum(list(reversed(examples)))


@pytest.mark.unit
def test_checksum_changes_when_a_single_character_changes(
    make_examples: Callable[[int], list[Example]],
) -> None:
    examples = make_examples(5)
    edited = [*examples[:-1], Example(examples[-1].example_id, examples[-1].source, "autre")]

    assert checksum(examples) != checksum(edited)


@pytest.mark.unit
def test_checksum_separates_fields(make_examples: Callable[[int], list[Example]]) -> None:
    # Without a separator, moving text from the document to the summary would
    # leave the digest unchanged.
    left = [Example("id", "ab", "c")]
    right = [Example("id", "a", "bc")]

    assert checksum(left) != checksum(right)


@pytest.mark.unit
def test_checksum_of_an_empty_corpus_is_defined() -> None:
    assert len(checksum([])) == 64
