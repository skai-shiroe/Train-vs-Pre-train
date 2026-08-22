"""ROUGE, computed with the reference implementation.

Section 2.1 locks the metrics to ROUGE-1, ROUGE-2 and ROUGE-L, and names
ROUGE-L in F-measure as the single number carried by the performance versus
corpus size curve.

**The scoring itself is not reimplemented.** ``rouge-score`` is the package
section 4 lists, and it is the implementation the summarisation literature
reports against. A hand written ROUGE that differed by a fraction of a point
would make every number in this project incomparable with any published one,
and the difference would be invisible: both would look like plausible scores.
What this module owns is everything around the call, which is where a
comparison actually breaks.

**Argument order.** ``RougeScorer.score`` takes the reference first and the
prediction second. Swapping them leaves the F-measure unchanged, so the mistake
survives any test written on F alone, while precision and recall silently trade
places. :func:`score_example` fixes the order once and
``tests/unit/test_metrics_rouge.py`` pins it with an asymmetric case.

**ROUGE-L, not ROUGE-Lsum.** ``rougeLsum`` splits on newlines and takes a union
of per sentence longest common subsequences. A CNN/DailyMail reference is three
to four sentences, and the pipeline stores it on a single line, so the two do
not agree here: ``rougeLsum`` would match each reference sentence against its
best counterpart anywhere in the generation, and ROUGE-L requires one
subsequence running through the whole pair. Section 2.1 says ROUGE-L, and that
is what is measured. What that costs has to be said rather than absorbed: the
published CNN/DailyMail figures are ROUGE-Lsum, they are the more permissive of
the two, and no number in this project is comparable to them. Every comparison
made here is internal, the same metric on the same test set for every model.

**An empty prediction scores zero and stays in the mean.** Dropping the
documents a model failed on would raise its average for having failed. The
count of empty predictions is reported next to the score by
:mod:`src.evaluation.summarization`, so a low mean can be told apart from a
broken model.

**The mean comes with a bootstrap interval.** A corpus mean alone invites
reading a gap of three tenths of a point as a result. Resampling the per example
scores gives the spread the mean was drawn from, seeded so that the interval is
reproducible. It measures sampling noise on this test set, nothing else: it
says nothing about a different corpus, and it does not turn an overlap into a
proof of equality.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from rouge_score.rouge_scorer import RougeScorer

#: Variants measured by the project, in the order they are reported.
ROUGE_VARIANTS: tuple[str, ...] = ("rouge1", "rouge2", "rougeL")

#: The single metric plotted against the corpus size, in F-measure.
REPORTED_VARIANT = "rougeL"


@dataclass(frozen=True, slots=True)
class RougeScore:
    """One ROUGE variant, in its three forms.

    Attributes:
        precision: Share of the generated units that appear in the reference.
        recall: Share of the reference units that appear in the generation.
        fmeasure: Harmonic mean of the two, the reported form.
    """

    precision: float
    recall: float
    fmeasure: float

    def to_dict(self) -> dict[str, float]:
        """Render the score as a flat mapping.

        Returns:
            A mapping with the three fields, JSON serialisable.
        """
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Percentile bootstrap interval around a corpus mean.

    Attributes:
        low: Lower bound.
        high: Upper bound.
        confidence: Nominal coverage, for example ``0.95``.
    """

    low: float
    high: float
    confidence: float

    def to_dict(self) -> dict[str, float]:
        """Render the interval as a flat mapping.

        Returns:
            A mapping with the three fields, JSON serialisable.
        """
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RougeConfig:
    """How ROUGE is measured, fixed once for every model compared.

    The settings belong to the measurement, not to a model. Scoring one model
    with stemming and another without would move the gap for a reason that has
    nothing to do with either of them, so the same configuration crosses the
    whole ablation.

    Attributes:
        variants: Variants to compute, a non empty subset of
            :data:`ROUGE_VARIANTS`.
        use_stemmer: Whether to reduce words to their Porter stem before
            matching. Enabled by default, as in the summarisation literature:
            without it ``report`` and ``reports`` count as a miss.
        bootstrap_samples: Number of resamples behind the confidence interval.
            Zero skips the interval entirely.
        confidence: Nominal coverage of the interval.
        seed: Seed of the resampling, so that the interval is reproducible.
    """

    variants: tuple[str, ...] = ROUGE_VARIANTS
    use_stemmer: bool = True
    bootstrap_samples: int = 1000
    confidence: float = 0.95
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate the measurement settings.

        Raises:
            ValueError: If the variants are empty, duplicated or unsupported,
                if the resample count is negative, if the confidence lies
                outside ``(0, 1)``, or if the seed is negative.
        """
        if not self.variants:
            raise ValueError("At least one ROUGE variant is required.")
        if len(set(self.variants)) != len(self.variants):
            raise ValueError(f"Duplicate ROUGE variants: {self.variants}.")

        unknown = [variant for variant in self.variants if variant not in ROUGE_VARIANTS]
        if unknown:
            raise ValueError(
                f"Unsupported ROUGE variants: {unknown}. "
                f"Section 2.1 measures {', '.join(ROUGE_VARIANTS)}."
            )

        if self.bootstrap_samples < 0:
            raise ValueError(
                f"bootstrap_samples must be non negative, got {self.bootstrap_samples}."
            )
        if not 0.0 < self.confidence < 1.0:
            raise ValueError(f"confidence must lie in (0, 1), got {self.confidence}.")
        if self.seed < 0:
            raise ValueError(f"seed must be non negative, got {self.seed}.")

    def to_dict(self) -> dict[str, Any]:
        """Render the configuration as a flat mapping.

        Returns:
            A mapping suitable for MLflow parameter logging.
        """
        payload = asdict(self)
        payload["variants"] = list(self.variants)
        return payload


@dataclass(frozen=True, slots=True)
class CorpusRouge:
    """ROUGE over a whole split.

    Attributes:
        size: Number of scored examples, including the failed ones.
        scores: Mean of the per example scores, per variant.
        intervals: Bootstrap interval of the mean F-measure, per variant. Empty
            when the interval was disabled.
    """

    size: int
    scores: dict[str, RougeScore]
    intervals: dict[str, ConfidenceInterval]

    @property
    def reported(self) -> float:
        """Return the metric named by section 2.1.

        Returns:
            The mean ROUGE-L F-measure.

        Raises:
            KeyError: If ROUGE-L was not among the measured variants.
        """
        if REPORTED_VARIANT not in self.scores:
            raise KeyError(
                f"{REPORTED_VARIANT} was not measured, so the reported metric of "
                "section 2.1 does not exist for this run."
            )
        return self.scores[REPORTED_VARIANT].fmeasure

    def to_dict(self) -> dict[str, Any]:
        """Render the corpus scores as a nested mapping.

        Returns:
            A JSON serialisable mapping.
        """
        return {
            "size": self.size,
            "scores": {variant: score.to_dict() for variant, score in self.scores.items()},
            "intervals": {
                variant: interval.to_dict() for variant, interval in self.intervals.items()
            },
        }


@lru_cache(maxsize=8)
def _build_scorer(variants: tuple[str, ...], use_stemmer: bool) -> RougeScorer:
    """Build a scorer, cached per configuration.

    Building the scorer loads the stemmer. Rebuilding it once per example would
    dominate the cost of scoring a thousand summaries.

    Args:
        variants: Variants the scorer computes.
        use_stemmer: Whether to stem before matching.

    Returns:
        The scorer.
    """
    return RougeScorer(list(variants), use_stemmer=use_stemmer)


def score_example(
    prediction: str, reference: str, config: RougeConfig | None = None
) -> dict[str, RougeScore]:
    """Score one generated summary against its reference.

    Args:
        prediction: The generated summary. May be empty, and then scores zero.
        reference: The reference summary. Must not be empty.
        config: Measurement settings. Defaults to the project settings.

    Returns:
        One score per configured variant.

    Raises:
        ValueError: If the reference is blank. An empty reference is a corpus
            defect, not a model failure, and scoring it as a zero would blame
            the model for it.
    """
    settings = config or RougeConfig()
    if not reference.strip():
        raise ValueError(
            "The reference summary is empty. A missing reference is a corpus "
            "defect: fix the corpus rather than scoring the prediction as zero."
        )

    # score() takes the reference first. See the module docstring: the mistake
    # is invisible on the F-measure and silently swaps precision and recall.
    raw = _build_scorer(settings.variants, settings.use_stemmer).score(reference, prediction)
    return {
        variant: RougeScore(
            precision=float(score.precision),
            recall=float(score.recall),
            fmeasure=float(score.fmeasure),
        )
        for variant, score in raw.items()
    }


def score_all(
    predictions: Sequence[str],
    references: Sequence[str],
    config: RougeConfig | None = None,
) -> list[dict[str, RougeScore]]:
    """Score every prediction of a split against its reference.

    Args:
        predictions: Generated summaries, aligned with ``references``.
        references: Reference summaries.
        config: Measurement settings.

    Returns:
        One score mapping per example, in the order they were given.

    Raises:
        ValueError: If the two sequences have different lengths, or if they are
            empty.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"{len(predictions)} predictions for {len(references)} references. "
            "The two must be aligned, or every score refers to the wrong document."
        )
    if not predictions:
        raise ValueError("Cannot score an empty corpus.")

    settings = config or RougeConfig()
    return [
        score_example(prediction, reference, settings)
        for prediction, reference in zip(predictions, references, strict=True)
    ]


def _bootstrap_interval(values: Sequence[float], config: RougeConfig) -> ConfidenceInterval:
    """Resample a series of per example scores and bound its mean.

    Args:
        values: The per example F-measures.
        config: Provides the resample count, the coverage and the seed.

    Returns:
        The percentile interval of the resampled means.
    """
    observations = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(config.seed)
    draws = generator.integers(
        0, observations.size, size=(config.bootstrap_samples, observations.size)
    )
    means = observations[draws].mean(axis=1)

    tail = (1.0 - config.confidence) / 2.0
    bounds = np.quantile(means, [tail, 1.0 - tail])
    return ConfidenceInterval(
        low=float(bounds[0]), high=float(bounds[1]), confidence=config.confidence
    )


def aggregate(
    per_example: Sequence[Mapping[str, RougeScore]],
    config: RougeConfig | None = None,
) -> CorpusRouge:
    """Average per example scores over a split.

    The corpus score is the mean of the per example scores, which is what the
    summarisation literature reports. Concatenating every prediction and
    scoring once would weight a long summary more than a short one.

    Args:
        per_example: Scores produced by :func:`score_all`.
        config: Measurement settings. Must list the same variants that were
            scored.

    Returns:
        The corpus level scores and their confidence intervals.

    Raises:
        ValueError: If the sequence is empty, or if a configured variant is
            missing from the scored examples.
    """
    if not per_example:
        raise ValueError("Cannot aggregate an empty score list.")

    settings = config or RougeConfig()
    missing = [variant for variant in settings.variants if variant not in per_example[0]]
    if missing:
        raise ValueError(
            f"Variants {missing} were never scored. Aggregate with the same "
            "configuration the examples were scored with."
        )

    scores: dict[str, RougeScore] = {}
    intervals: dict[str, ConfidenceInterval] = {}
    for variant in settings.variants:
        precisions = [float(example[variant].precision) for example in per_example]
        recalls = [float(example[variant].recall) for example in per_example]
        fmeasures = [float(example[variant].fmeasure) for example in per_example]

        scores[variant] = RougeScore(
            precision=float(np.mean(precisions)),
            recall=float(np.mean(recalls)),
            fmeasure=float(np.mean(fmeasures)),
        )
        if settings.bootstrap_samples > 0:
            intervals[variant] = _bootstrap_interval(fmeasures, settings)

    return CorpusRouge(size=len(per_example), scores=scores, intervals=intervals)


def score_corpus(
    predictions: Sequence[str],
    references: Sequence[str],
    config: RougeConfig | None = None,
) -> CorpusRouge:
    """Score a whole split in one call.

    Args:
        predictions: Generated summaries, aligned with ``references``.
        references: Reference summaries.
        config: Measurement settings.

    Returns:
        The corpus level scores.
    """
    settings = config or RougeConfig()
    return aggregate(score_all(predictions, references, settings), settings)
