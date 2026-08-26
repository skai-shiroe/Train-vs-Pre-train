"""Integration tests of the evaluation chain.

Section 35 asks for a ``model -> generation -> metric`` chain and a
``generation -> metric`` one. Both are covered here, first offline on a tiny
randomly initialised T5, then on the real ``t5-small`` weights under the ``slow``
marker.

Nothing here reports a score. The corpus is a handful of hand written
sentences, and any number measured on it would say nothing about CNN/DailyMail.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from transformers import T5Config, T5ForConditionalGeneration

from src.data.example import Example
from src.evaluation.evaluator import evaluate_summarizer
from src.metrics.rouge import REPORTED_VARIANT, RougeConfig
from src.models.generation import GenerationConfig
from src.models.pretrained.base import BaselineConfig
from src.models.pretrained.factory import build_summarizer
from src.models.pretrained.t5 import T5Summarizer
from src.utils.seed import set_seed
from tests.conftest import FAKE_VOCAB_SIZE, FakeTokenizer

pytestmark = pytest.mark.integration

REFERENCES = (
    "the council approved the bridge plan",
    "the school will open in september",
    "the river flooded the market square",
    "the museum reopened after two years",
)

DOCUMENTS = (
    "The city council has approved a plan to rebuild the old railway bridge over the river. "
    "Work is expected to start in the spring.",
    "A new primary school will open its doors in September after eighteen months of building "
    "work on the edge of the town.",
    "Heavy rain overnight caused the river to burst its banks and flood the market square, "
    "closing several shops.",
    "The county museum has reopened to the public after a two year refurbishment funded by a "
    "national grant.",
)


def offline_t5(fake_tokenizer: FakeTokenizer) -> T5Summarizer:
    """Build a tiny T5 without downloading weights or a tokenizer."""
    model = T5ForConditionalGeneration(
        T5Config(
            vocab_size=FAKE_VOCAB_SIZE,
            d_model=16,
            d_ff=32,
            d_kv=8,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            dropout_rate=0.0,
            pad_token_id=0,
            eos_token_id=1,
            decoder_start_token_id=0,
        )
    )
    return T5Summarizer(
        BaselineConfig(
            hf_id="t5-small",
            source_prefix="summarize: ",
            max_source_tokens=32,
            max_target_tokens=8,
        ),
        model,
        fake_tokenizer,  # type: ignore[arg-type]
        pretrained=False,
    )


def make_examples() -> list[Example]:
    return [
        Example(example_id=f"id-{index}", source=document, target=reference)
        for index, (document, reference) in enumerate(zip(DOCUMENTS, REFERENCES, strict=True))
    ]


class LookupSummarizer:
    """Return a summary chosen from a table, to test the alignment of the chain."""

    def __init__(self, summaries: Sequence[str]) -> None:
        self._table = dict(zip(DOCUMENTS, summaries, strict=True))

    def describe(self) -> dict[str, str]:
        return {"model": "lookup", "mode": "mock"}

    def summarize(
        self,
        documents: Sequence[str],
        config: GenerationConfig | None = None,
        *,
        batch_size: int = 8,
    ) -> list[str]:
        return [self._table[document] for document in documents]


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def test_each_prediction_is_scored_against_the_reference_of_its_own_document() -> None:
    """Rotating the answers by one must cost the run its score, not go unnoticed."""
    rotated = (*REFERENCES[1:], REFERENCES[0])

    aligned = evaluate_summarizer(
        LookupSummarizer(REFERENCES), make_examples(), rouge=RougeConfig(bootstrap_samples=0)
    )
    shifted = evaluate_summarizer(
        LookupSummarizer(rotated), make_examples(), rouge=RougeConfig(bootstrap_samples=0)
    )

    assert aligned.report.reported == pytest.approx(1.0)
    assert shifted.report.reported < 0.5


# ---------------------------------------------------------------------------
# Model to generation to metric, offline
# ---------------------------------------------------------------------------


def test_the_random_t5_runs_the_whole_chain(fake_tokenizer: FakeTokenizer) -> None:
    """From documents to a corpus score, with real T5 generation and ROUGE."""
    set_seed(0)
    summarizer = offline_t5(fake_tokenizer)

    result = evaluate_summarizer(
        summarizer,
        make_examples(),
        generation=GenerationConfig(max_new_tokens=6),
        rouge=RougeConfig(bootstrap_samples=50),
    )

    assert result.report.size == 4
    assert result.model["mode"] == "untrained"
    assert 0.0 <= result.report.reported <= 1.0
    interval = result.report.rouge.intervals[REPORTED_VARIANT]
    assert interval.low <= result.report.reported <= interval.high


def test_beam_search_and_greedy_search_both_reach_the_metric(
    fake_tokenizer: FakeTokenizer,
) -> None:
    set_seed(0)
    summarizer = offline_t5(fake_tokenizer)
    examples = make_examples()
    settings = RougeConfig(bootstrap_samples=0)

    greedy = evaluate_summarizer(
        summarizer, examples, generation=GenerationConfig(max_new_tokens=6), rouge=settings
    )
    beams = evaluate_summarizer(
        summarizer,
        examples,
        generation=GenerationConfig(max_new_tokens=6, num_beams=3),
        rouge=settings,
    )

    assert greedy.report.size == beams.report.size == 4
    assert greedy.generation.num_beams == 1
    assert beams.generation.num_beams == 3


# ---------------------------------------------------------------------------
# The real baseline
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_real_baseline_is_evaluated_deterministically() -> None:
    """Greedy decoding plus a fixed metric must give the same score twice.

    An evaluation that moved between two runs of the same model on the same
    examples could not support any comparison, and the gap between two models
    would be indistinguishable from the noise of the measurement itself.
    """
    summarizer = build_summarizer(
        "t5",
        BaselineConfig(
            hf_id="t5-small",
            source_prefix="summarize: ",
            max_source_tokens=128,
            max_target_tokens=24,
        ),
    )
    examples = make_examples()
    decoding = GenerationConfig(max_new_tokens=24)
    settings = RougeConfig(bootstrap_samples=100)

    first = evaluate_summarizer(summarizer, examples, generation=decoding, rouge=settings)
    second = evaluate_summarizer(summarizer, examples, generation=decoding, rouge=settings)

    assert [item.prediction for item in first.predictions] == [
        item.prediction for item in second.predictions
    ]
    assert first.report.reported == pytest.approx(second.report.reported)
    assert first.report.empty_predictions == 0
    assert 0.0 <= first.report.reported <= 1.0
    interval = first.report.rouge.intervals[REPORTED_VARIANT]
    assert interval.low <= first.report.reported <= interval.high
