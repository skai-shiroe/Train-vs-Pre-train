"""Construction of the frozen working corpus.

The upstream corpus is too large for the hardware budget of the project, so a
subset is drawn once, with a fixed seed, and frozen. Section 2.2 of the
specification makes this explicit: every reported score refers to this working
corpus, never to the full dataset.

The training split is stored in shuffled order. Every ablation proportion is
therefore a prefix of it, which makes the subsets nested by construction: the
10 percent subset is contained in the 50 percent subset, itself contained in
the full corpus. Nesting isolates the effect of the corpus size from the effect
of the sample composition.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass

from src.data.example import Example


@dataclass(frozen=True, slots=True)
class WorkingCorpus:
    """The frozen splits used by every experiment.

    Attributes:
        train: Training examples, in shuffled order.
        validation: Validation examples.
        test: Test examples, identical for every comparison.
        seed: Seed of the draw that produced the splits.
    """

    train: tuple[Example, ...]
    validation: tuple[Example, ...]
    test: tuple[Example, ...]
    seed: int

    def splits(self) -> dict[str, tuple[Example, ...]]:
        """Return the three splits keyed by name.

        Returns:
            A mapping from split name to examples.
        """
        return {"train": self.train, "validation": self.validation, "test": self.test}


def sample_split(
    examples: Sequence[Example],
    size: int,
    seed: int,
) -> tuple[Example, ...]:
    """Draw a shuffled subset of a split, deterministically.

    A dedicated generator is used rather than the global one, so that drawing a
    split never depends on how much randomness was consumed before.

    Args:
        examples: Population to draw from.
        size: Number of examples to keep.
        seed: Seed of the dedicated generator.

    Returns:
        The drawn examples, in shuffled order.

    Raises:
        ValueError: If ``size`` is not strictly positive, or if the population
            is smaller than the requested size.
    """
    if size <= 0:
        raise ValueError(f"Requested size must be strictly positive, got {size}.")
    if len(examples) < size:
        raise ValueError(
            f"Cannot draw {size} examples from a population of {len(examples)}. "
            "Reduce the working corpus size or check the upstream download."
        )

    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    return tuple(examples[index] for index in indices[:size])


def build_working_corpus(
    train: Sequence[Example],
    validation: Sequence[Example],
    test: Sequence[Example],
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
) -> WorkingCorpus:
    """Draw the three frozen splits from the upstream corpus.

    Each split gets its own derived seed, so that changing the training size
    never changes the validation or the test split.

    Args:
        train: Upstream training population.
        validation: Upstream validation population.
        test: Upstream test population.
        train_size: Number of training examples to keep.
        validation_size: Number of validation examples to keep.
        test_size: Number of test examples to keep.
        seed: Base seed of the draw.

    Returns:
        The frozen working corpus.
    """
    return WorkingCorpus(
        train=sample_split(train, train_size, seed),
        validation=sample_split(validation, validation_size, seed + 1),
        test=sample_split(test, test_size, seed + 2),
        seed=seed,
    )


def proportion_subset(train: Sequence[Example], percentage: int) -> tuple[Example, ...]:
    """Return the ablation subset for a given percentage.

    Args:
        train: The frozen training split, in its stored order.
        percentage: Requested proportion, in percent.

    Returns:
        The first ``percentage`` percent of the training split, at least one
        example.

    Raises:
        ValueError: If the percentage is outside the ``(0, 100]`` range.
    """
    if not 0 < percentage <= 100:
        raise ValueError(f"Percentage must lie in (0, 100], got {percentage}.")

    size = max(1, round(len(train) * percentage / 100))
    return tuple(train[:size])


def checksum(examples: Sequence[Example]) -> str:
    """Compute a stable checksum over an ordered sequence of examples.

    The checksum covers the identifiers, the documents and the summaries, in
    order. Any reordering, addition or edit changes it. It is the value stored
    as ``dataset_version`` and traced in MLflow with every run.

    Args:
        examples: The examples to fingerprint.

    Returns:
        The hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    for example in examples:
        digest.update(example.example_id.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(example.source.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(example.target.encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()
