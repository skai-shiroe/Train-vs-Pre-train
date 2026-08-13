"""The single interface the evaluator knows about.

Section 2.1 compares a hand written Transformer with ``t5-small`` on the same
test set. If the evaluator held a branch on which model it was given, the two
sides would drift: a truncation applied on one path and not the other, a
decoding budget read from a different place. There is one path, and both models
reach it through this protocol.

It is structural on purpose. :class:`src.models.pretrained.base.PretrainedSummarizer`
and :class:`src.models.scratch.summarizer.ScratchSummarizer` satisfy it without
importing it, so neither model package depends on the evaluation package. The
conformance is checked in ``tests/unit/test_evaluation_evaluator.py`` rather
than by inheritance.

Two methods are required, and the second is not decoration. Section 19 wants a
reported score to carry what produced it, so a summariser that cannot describe
itself cannot be evaluated.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from src.models.generation import GenerationConfig


@runtime_checkable
class Summarizer(Protocol):
    """A model that turns documents into summaries."""

    def describe(self) -> dict[str, str]:
        """Return what produced the scores, for the run record.

        Returns:
            A flat mapping of strings, always JSON serialisable. It must state
            the mode the weights are in, since the same architecture measured
            untrained and trained is two different results.
        """

    def summarize(
        self,
        documents: Sequence[str],
        config: GenerationConfig | None = None,
        *,
        batch_size: int = 8,
    ) -> list[str]:
        """Summarise documents, in batches.

        Args:
            documents: The documents to summarise, raw. Applying the task
                prefix and the truncation is the summariser's job, because both
                depend on how the model was trained.
            config: Decoding configuration. The evaluator always passes one
                explicitly rather than relying on a per model default.
            batch_size: Number of documents encoded at once.

        Returns:
            One summary per document, in the order they were given.
        """
