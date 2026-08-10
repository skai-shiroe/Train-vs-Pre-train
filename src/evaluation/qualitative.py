"""Selection of generated summaries for the qualitative analysis.

Section 2 requires a qualitative analysis next to the scores. A ROUGE gap says
that one model matches more reference n-grams; it does not say whether the
weaker model produces truncated sentences, repeats the first line of the
document, or hallucinates a plausible summary of something else. Only reading
the output answers that, and reading it means choosing what to read.

**Best and worst are not enough.** They are the two ends of the distribution and
they are the two least representative examples in it. A model can have a
brilliant best case, a catastrophic worst case, and a flat, generic middle that
is what it actually does. The selection therefore adds a random draw from the
rest, seeded, so that the analysis includes examples nobody picked.

**The selection is deterministic.** Ties break on the example identifier and the
draw is seeded, so the same run selects the same documents. A qualitative
analysis whose examples changed between two readings of the same results would
not be verifiable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data.example import Example
from src.metrics.rouge import REPORTED_VARIANT, RougeScore


@dataclass(frozen=True, slots=True)
class ScoredPrediction:
    """One generated summary with the score it obtained.

    Attributes:
        example_id: Identifier of the example in the frozen corpus.
        source: The document that was summarised.
        reference: The reference summary.
        prediction: The generated summary.
        scores: F-measure per ROUGE variant.
    """

    example_id: str
    source: str
    reference: str
    prediction: str
    scores: dict[str, float]

    def to_dict(self, *, include_source: bool = True) -> dict[str, Any]:
        """Render the prediction as a mapping.

        Args:
            include_source: Whether to carry the document. The per example
                records written for a whole split leave it out: the identifier
                already points at the frozen corpus, and repeating a thousand
                documents would make the artefact larger than the corpus split
                it describes.

        Returns:
            A JSON serialisable mapping.
        """
        payload: dict[str, Any] = {
            "id": self.example_id,
            "reference": self.reference,
            "prediction": self.prediction,
            "scores": dict(self.scores),
        }
        if include_source:
            payload["source"] = self.source
        return payload


@dataclass(frozen=True, slots=True)
class QualitativeSelection:
    """The examples chosen for the qualitative analysis.

    Attributes:
        variant: ROUGE variant the ranking is based on.
        best: Highest scoring examples, best first.
        worst: Lowest scoring examples, worst first.
        sample: Examples drawn at random from the rest.
    """

    variant: str
    best: tuple[ScoredPrediction, ...]
    worst: tuple[ScoredPrediction, ...]
    sample: tuple[ScoredPrediction, ...]

    def to_dict(self) -> dict[str, Any]:
        """Render the selection as a nested mapping.

        Returns:
            A JSON serialisable mapping, documents included.
        """
        return {
            "variant": self.variant,
            "best": [item.to_dict() for item in self.best],
            "worst": [item.to_dict() for item in self.worst],
            "sample": [item.to_dict() for item in self.sample],
        }


def score_predictions(
    examples: Sequence[Example],
    predictions: Sequence[str],
    per_example: Sequence[Mapping[str, RougeScore]],
) -> list[ScoredPrediction]:
    """Attach each prediction to its example and its score.

    Args:
        examples: The examples that were summarised, in order.
        predictions: The generated summaries, aligned with ``examples``.
        per_example: Scores produced by :func:`src.metrics.rouge.score_all`.

    Returns:
        One record per example, in the order they were given.

    Raises:
        ValueError: If the three sequences do not have the same length.
    """
    if not (len(examples) == len(predictions) == len(per_example)):
        raise ValueError(
            f"{len(examples)} examples, {len(predictions)} predictions and "
            f"{len(per_example)} score mappings. The three must be aligned."
        )

    return [
        ScoredPrediction(
            example_id=example.example_id,
            source=example.source,
            reference=example.target,
            prediction=prediction,
            scores={variant: score.fmeasure for variant, score in scores.items()},
        )
        for example, prediction, scores in zip(examples, predictions, per_example, strict=True)
    ]


def select_qualitative(
    scored: Sequence[ScoredPrediction],
    *,
    variant: str = REPORTED_VARIANT,
    best: int = 3,
    worst: int = 3,
    sample: int = 4,
    seed: int = 42,
) -> QualitativeSelection:
    """Choose the examples the qualitative analysis reads.

    The three groups never overlap: the best are taken first, the worst from
    what is left, and the random draw from what remains after both. On a split
    smaller than the requested counts, the groups shrink rather than repeat an
    example under two headings.

    Args:
        scored: Every scored prediction of the split.
        variant: ROUGE variant the ranking is based on.
        best: Number of highest scoring examples to keep.
        worst: Number of lowest scoring examples to keep.
        sample: Number of examples to draw at random from the rest.
        seed: Seed of the draw.

    Returns:
        The selection.

    Raises:
        ValueError: If the split is empty, if a count is negative, or if the
            variant was not scored.
    """
    if not scored:
        raise ValueError("Cannot select examples from an empty split.")
    if best < 0 or worst < 0 or sample < 0:
        raise ValueError(
            f"Counts must be non negative, got best={best}, worst={worst}, sample={sample}."
        )
    if variant not in scored[0].scores:
        available = ", ".join(sorted(scored[0].scores))
        raise ValueError(f"Variant {variant!r} was not scored. Available: {available}.")

    # Descending score, then identifier: two documents with the same score must
    # rank in the same order on every reading of the same results.
    ordered = sorted(scored, key=lambda item: (-item.scores[variant], item.example_id))

    best_items = tuple(ordered[:best])
    remaining = ordered[best:]

    # A slice of [-0:] is the whole list, not an empty one.
    worst_items = tuple(reversed(remaining[-worst:])) if worst > 0 else ()
    rest = remaining[: len(remaining) - len(worst_items)]

    drawn = min(sample, len(rest))
    positions: list[int] = []
    if drawn > 0:
        generator = np.random.default_rng(seed)
        positions = sorted(
            int(index) for index in generator.choice(len(rest), size=drawn, replace=False)
        )
    sample_items = tuple(rest[position] for position in positions)

    return QualitativeSelection(
        variant=variant, best=best_items, worst=worst_items, sample=sample_items
    )
