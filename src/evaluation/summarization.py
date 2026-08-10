"""The summarisation report: ROUGE plus what makes a ROUGE readable.

Section 16 asks for ROUGE-1, ROUGE-2 and ROUGE-L. Three numbers are enough to
rank two models and not enough to understand them, so the report carries two
more things that cost nothing to compute and change how a score is read.

**The length of what was generated.** A model that emits four words on a corpus
whose references average twenty is not a weak model, it is a broken one, and its
ROUGE looks merely low. The length distributions sit next to the score so the
two are read together.

**The number of empty predictions.** They score zero and stay in the mean, as
:mod:`src.metrics.rouge` explains. Counting them separately is what lets a mean
of 0.05 over a thousand documents be told apart from a mean of 0.05 over a
hundred documents and nine hundred failures.

Translation is not covered here and has no module of its own. Section 2.1 puts
it out of scope, with no code and no metric, which makes the ``translation.py``
of the section 16 listing a file with nothing to hold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.data.statistics import LengthStatistics, summarise_lengths
from src.metrics.rouge import CorpusRouge, RougeConfig, RougeScore, aggregate, score_all


@dataclass(frozen=True, slots=True)
class SummarizationReport:
    """Everything measured on one model over one split.

    Attributes:
        size: Number of scored examples.
        rouge: Corpus level ROUGE, with its confidence intervals.
        prediction_words: Length distribution of the generated summaries.
        reference_words: Length distribution of the reference summaries.
        empty_predictions: Number of predictions that hold no word at all.
    """

    size: int
    rouge: CorpusRouge
    prediction_words: LengthStatistics
    reference_words: LengthStatistics
    empty_predictions: int

    @property
    def reported(self) -> float:
        """Return the metric named by section 2.1.

        Returns:
            The mean ROUGE-L F-measure.
        """
        return self.rouge.reported

    def to_dict(self) -> dict[str, Any]:
        """Render the report as a nested mapping.

        Returns:
            A JSON serialisable mapping.
        """
        return {
            "size": self.size,
            "rouge": self.rouge.to_dict(),
            "prediction_words": {
                "mean": self.prediction_words.mean,
                "median": self.prediction_words.median,
                "p95": self.prediction_words.p95,
                "minimum": self.prediction_words.minimum,
                "maximum": self.prediction_words.maximum,
            },
            "reference_words": {
                "mean": self.reference_words.mean,
                "median": self.reference_words.median,
                "p95": self.reference_words.p95,
                "minimum": self.reference_words.minimum,
                "maximum": self.reference_words.maximum,
            },
            "empty_predictions": self.empty_predictions,
        }


def evaluate_summaries(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    per_example: Sequence[Mapping[str, RougeScore]] | None = None,
    rouge_config: RougeConfig | None = None,
) -> SummarizationReport:
    """Build the report of one model over one split.

    Args:
        predictions: Generated summaries, aligned with ``references``.
        references: Reference summaries.
        per_example: Scores already computed by
            :func:`src.metrics.rouge.score_all`. Supplied by the evaluator,
            which needs them for the qualitative selection as well; scoring
            twice would double the cost of an evaluation for nothing.
        rouge_config: Measurement settings.

    Returns:
        The report.

    Raises:
        ValueError: If the sequences are empty or misaligned, or if the
            supplied scores do not cover every example.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"{len(predictions)} predictions for {len(references)} references. "
            "The two must be aligned."
        )
    if not predictions:
        raise ValueError("Cannot report on an empty split.")

    settings = rouge_config or RougeConfig()
    if per_example is None:
        scored: Sequence[Mapping[str, RougeScore]] = score_all(predictions, references, settings)
    elif len(per_example) != len(predictions):
        raise ValueError(f"{len(per_example)} scored examples for {len(predictions)} predictions.")
    else:
        scored = per_example

    return SummarizationReport(
        size=len(predictions),
        rouge=aggregate(scored, settings),
        prediction_words=summarise_lengths([len(text.split()) for text in predictions]),
        reference_words=summarise_lengths([len(text.split()) for text in references]),
        empty_predictions=sum(1 for text in predictions if not text.strip()),
    )
