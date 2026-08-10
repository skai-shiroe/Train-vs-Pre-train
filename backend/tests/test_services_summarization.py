"""Unit tests for the inference services of sections 23 and 24."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
import torch

from backend.app.core.errors import (
    ErrorCode,
    InferenceFailedError,
    InferenceTimeoutError,
    ModelNotFoundError,
    ModelNotReadyError,
)
from backend.app.inference.base import LoadedModel
from backend.app.registry.models import ModelVersion
from backend.app.services.store import ModelStore
from backend.app.services.summarization import compare, summarize
from backend.tests.conftest import build_version
from backend.tests.test_services_store import FakeRegistry
from src.models.generation import GenerationConfig

CPU = torch.device("cpu")
DECODING = GenerationConfig(max_new_tokens=16)


class ScriptedSummarizer:
    """A summariser whose behaviour a test decides."""

    def __init__(
        self, answer: str = "resume", *, sleep: float = 0.0, raises: BaseException | None = None
    ) -> None:
        self.answer = answer
        self.sleep = sleep
        self.raises = raises
        self.seen: list[str] = []

    def summarize(self, documents: Any, config: Any = None, *, batch_size: int = 8) -> list[str]:
        self.seen.extend(documents)
        if self.raises is not None:
            raise self.raises
        if self.sleep:
            time.sleep(self.sleep)
        return [self.answer for _ in documents]

    def describe(self) -> dict[str, str]:
        return {"model": self.answer}


def build_store(
    budgets: dict[str, int] | None = None, **summarizers: ScriptedSummarizer
) -> ModelStore:
    """Return a store whose names are already loaded with scripted models."""
    registry = FakeRegistry(
        {name: [build_version(name=name)] for name in summarizers},
        current=dict.fromkeys(summarizers, "v1"),
        aliases={"champion": (next(iter(summarizers)), "v1")},
    )
    budget = budgets or {}

    def loader(version: ModelVersion, weights: Path | None) -> LoadedModel:
        return LoadedModel(
            version=version,
            summarizer=summarizers[version.name],
            device=CPU,
            token_budget=budget.get(version.name, 64),
        )

    return ModelStore(registry, device=CPU, loader=loader)


# -- One model --------------------------------------------------------------


@pytest.mark.unit
async def test_a_prediction_names_the_version_that_produced_it() -> None:
    store = build_store(scratch=ScriptedSummarizer("un resume"))

    prediction = await summarize(store, "un document", config=DECODING, timeout_s=5.0)

    assert prediction.summary == "un resume"
    assert prediction.version.label == "scratch:v1"
    assert prediction.latency_ms >= 0.0
    assert prediction.rouge is None


@pytest.mark.unit
async def test_the_document_reaches_the_model_unchanged() -> None:
    model = ScriptedSummarizer()
    store = build_store(scratch=model)

    await summarize(store, "  espaces conserves  ", config=DECODING, timeout_s=5.0)

    assert model.seen == ["  espaces conserves  "]


@pytest.mark.unit
async def test_a_generation_over_the_deadline_is_a_timeout() -> None:
    store = build_store(scratch=ScriptedSummarizer(sleep=0.5))

    with pytest.raises(InferenceTimeoutError) as raised:
        await summarize(store, "un document", config=DECODING, timeout_s=0.01)

    assert raised.value.code is ErrorCode.TIMEOUT
    assert raised.value.status_code == 504
    assert raised.value.details == {"model": "scratch", "timeout_s": 0.01}


@pytest.mark.unit
async def test_a_model_that_raises_answers_inference_failed() -> None:
    store = build_store(scratch=ScriptedSummarizer(raises=RuntimeError("CUDA out of memory")))

    with pytest.raises(InferenceFailedError) as raised:
        await summarize(store, "un document", config=DECODING, timeout_s=5.0)

    assert raised.value.status_code == 500
    assert raised.value.details == {"model": "scratch"}
    # What torch said stays in the log: the client is told which model failed.
    assert "CUDA" not in raised.value.message


@pytest.mark.unit
async def test_an_application_error_is_not_rewritten() -> None:
    store = build_store(scratch=ScriptedSummarizer(raises=ModelNotReadyError("poids absents")))

    with pytest.raises(ModelNotReadyError, match="poids absents"):
        await summarize(store, "un document", config=DECODING, timeout_s=5.0)


@pytest.mark.unit
async def test_an_unknown_model_is_not_found() -> None:
    store = build_store(scratch=ScriptedSummarizer())

    with pytest.raises(ModelNotFoundError):
        await summarize(store, "un document", name="absent", config=DECODING, timeout_s=5.0)


# -- Several models ---------------------------------------------------------


@pytest.mark.unit
async def test_a_comparison_answers_in_the_order_asked_for() -> None:
    store = build_store(
        scratch=ScriptedSummarizer("cote from scratch"),
        pretrained=ScriptedSummarizer("cote pre-entraine"),
    )

    comparison = await compare(
        store, "un document", names=["pretrained", "scratch"], config=DECODING, timeout_s=5.0
    )

    predictions = comparison.predictions
    assert [prediction.version.name for prediction in predictions] == ["pretrained", "scratch"]
    assert [prediction.summary for prediction in predictions] == [
        "cote pre-entraine",
        "cote from scratch",
    ]


@pytest.mark.unit
async def test_without_a_reference_a_comparison_carries_no_score() -> None:
    store = build_store(scratch=ScriptedSummarizer())

    comparison = await compare(
        store, "un document", names=["scratch"], config=DECODING, timeout_s=5.0
    )

    assert comparison.predictions[0].rouge is None


@pytest.mark.unit
async def test_a_reference_is_scored_against_every_prediction() -> None:
    store = build_store(
        scratch=ScriptedSummarizer("the cat sat"),
        pretrained=ScriptedSummarizer("nothing alike"),
    )

    comparison = await compare(
        store,
        "un document",
        names=["scratch", "pretrained"],
        reference="the cat sat on the mat",
        config=DECODING,
        timeout_s=5.0,
    )

    scratch, pretrained = comparison.predictions
    assert scratch.rouge is not None
    # Three words of three, all present in the reference: full precision.
    assert scratch.rouge["rouge1"].precision == pytest.approx(1.0)
    assert scratch.rouge["rouge1"].recall == pytest.approx(0.5)
    assert pretrained.rouge is not None
    assert pretrained.rouge["rouge1"].fmeasure == pytest.approx(0.0)


@pytest.mark.unit
async def test_the_budget_is_capped_by_what_the_weights_can_produce() -> None:
    store = build_store({"scratch": 8}, scratch=ScriptedSummarizer())

    prediction = await summarize(
        store, "un document", config=GenerationConfig(max_new_tokens=64), timeout_s=5.0
    )

    assert prediction.decoding.max_new_tokens == 8


@pytest.mark.unit
async def test_a_budget_the_weights_can_honour_is_left_alone() -> None:
    store = build_store({"scratch": 64}, scratch=ScriptedSummarizer())

    prediction = await summarize(
        store, "un document", config=GenerationConfig(max_new_tokens=16), timeout_s=5.0
    )

    assert prediction.decoding.max_new_tokens == 16


@pytest.mark.unit
async def test_a_comparison_runs_under_the_smallest_budget_of_its_models() -> None:
    store = build_store(
        {"scratch": 8, "pretrained": 32},
        scratch=ScriptedSummarizer(),
        pretrained=ScriptedSummarizer(),
    )

    comparison = await compare(
        store,
        "un document",
        names=["pretrained", "scratch"],
        config=GenerationConfig(max_new_tokens=64),
        timeout_s=5.0,
    )

    assert comparison.decoding.max_new_tokens == 8
    assert {prediction.decoding.max_new_tokens for prediction in comparison.predictions} == {8}


@pytest.mark.unit
async def test_a_comparison_stops_on_an_unknown_model() -> None:
    store = build_store(scratch=ScriptedSummarizer())

    with pytest.raises(ModelNotFoundError):
        await compare(
            store, "un document", names=["scratch", "absent"], config=DECODING, timeout_s=5.0
        )
