"""Unit tests for the request and response schemas of sections 23 and 24."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings
from backend.app.core.errors import ErrorCode
from backend.app.schemas import inference as schemas
from backend.app.schemas.common import ErrorResponse, error_responses
from backend.app.schemas.inference import MAX_BEAMS, CompareRequest, PredictRequest


@pytest.fixture(autouse=True)
def bounded_instance(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Pin the limits the validators read, whatever the local .env holds."""
    settings = Settings(_env_file=None, max_input_chars=100, max_summary_tokens=64)
    monkeypatch.setattr(schemas, "get_settings", lambda: settings)
    return settings


# -- The document -----------------------------------------------------------


@pytest.mark.unit
def test_a_minimal_request_carries_only_a_document() -> None:
    request = PredictRequest(text="un document")

    assert request.model is None
    assert request.version is None
    assert request.max_new_tokens is None
    assert request.num_beams == 1
    # Sampled by default, and without a seed: the endpoint draws one so that the
    # answer stays replayable without the caller having to think about it.
    assert request.do_sample is True
    assert request.seed is None


@pytest.mark.unit
@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_a_blank_document_is_refused(text: str) -> None:
    with pytest.raises(ValidationError, match="ne peut pas etre vide"):
        PredictRequest(text=text)


@pytest.mark.unit
def test_a_document_over_the_configured_limit_is_refused() -> None:
    with pytest.raises(ValidationError, match="limite de 100 caracteres"):
        PredictRequest(text="a" * 101)


@pytest.mark.unit
def test_the_limit_comes_from_the_settings_not_from_a_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        schemas, "get_settings", lambda: Settings(_env_file=None, max_input_chars=5)
    )

    with pytest.raises(ValidationError, match="limite de 5 caracteres"):
        PredictRequest(text="a" * 6)


@pytest.mark.unit
def test_the_document_is_not_rewritten() -> None:
    assert PredictRequest(text="  garde  ").text == "  garde  "


# -- Decoding ---------------------------------------------------------------


@pytest.mark.unit
def test_a_budget_over_the_instance_budget_is_refused() -> None:
    with pytest.raises(ValidationError, match="depasser 64 tokens"):
        PredictRequest(text="un document", max_new_tokens=65)


@pytest.mark.unit
@pytest.mark.parametrize("beams", [0, MAX_BEAMS + 1])
def test_a_beam_width_outside_the_bounds_is_refused(beams: int) -> None:
    with pytest.raises(ValidationError):
        PredictRequest(text="un document", num_beams=beams)


@pytest.mark.unit
def test_the_evaluation_beam_width_is_reachable() -> None:
    request = PredictRequest(text="un document", do_sample=False, num_beams=4)

    assert request.num_beams == 4


@pytest.mark.unit
def test_an_unknown_field_is_refused() -> None:
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({"text": "un document", "penalite": 0.9})


# -- Sampling ---------------------------------------------------------------


@pytest.mark.unit
def test_the_sampling_defaults_lean_towards_the_likely_tokens() -> None:
    request = PredictRequest(text="un document")

    assert request.temperature == schemas.DEFAULT_TEMPERATURE
    assert request.top_k == schemas.DEFAULT_TOP_K
    assert request.top_p == schemas.DEFAULT_TOP_P


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"temperature": 0.0},
        {"temperature": schemas.MAX_TEMPERATURE + 0.1},
        {"top_k": -1},
        {"top_p": 0.0},
        {"top_p": 1.1},
        {"seed": -1},
        {"seed": schemas.MAX_SEED + 1},
    ],
)
def test_a_sampling_knob_outside_its_domain_is_refused(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        PredictRequest(text="un document", **overrides)


@pytest.mark.unit
def test_sampling_and_a_beam_search_cannot_be_asked_for_together() -> None:
    # Scoring beams on draws is a third strategy. Refusing is what keeps the
    # caller from believing one half of the request was honoured.
    with pytest.raises(ValidationError, match="exclusifs"):
        PredictRequest(text="un document", num_beams=4)


@pytest.mark.unit
def test_the_exclusion_also_guards_a_comparison() -> None:
    with pytest.raises(ValidationError, match="exclusifs"):
        CompareRequest(text="un document", num_beams=2)


# -- Comparison -------------------------------------------------------------


@pytest.mark.unit
def test_a_comparison_defaults_to_every_registered_model() -> None:
    assert CompareRequest(text="un document").models is None


@pytest.mark.unit
def test_an_explicit_null_is_the_same_as_an_absent_list() -> None:
    assert CompareRequest.model_validate({"text": "un document", "models": None}).models is None


@pytest.mark.unit
def test_an_empty_list_of_models_is_refused() -> None:
    with pytest.raises(ValidationError, match="liste vide"):
        CompareRequest(text="un document", models=[])


@pytest.mark.unit
def test_the_same_model_twice_is_refused() -> None:
    with pytest.raises(ValidationError, match="deux fois le meme"):
        CompareRequest(text="un document", models=["scratch", "scratch"])


@pytest.mark.unit
def test_too_many_models_are_refused() -> None:
    with pytest.raises(ValidationError):
        CompareRequest(text="un document", models=["a", "b", "c", "d", "e"])


@pytest.mark.unit
def test_a_blank_reference_is_refused() -> None:
    with pytest.raises(ValidationError, match="reference ne peut pas etre vide"):
        CompareRequest(text="un document", reference="   ")


@pytest.mark.unit
def test_an_absent_reference_is_accepted() -> None:
    assert CompareRequest(text="un document", reference=None).reference is None


# -- The error envelope -----------------------------------------------------


@pytest.mark.unit
def test_the_documented_status_of_a_code_is_the_one_the_handler_uses() -> None:
    responses = error_responses(ErrorCode.MODEL_NOT_FOUND, ErrorCode.TIMEOUT)

    assert set(responses) == {404, 504}
    assert responses[404]["model"] is ErrorResponse


@pytest.mark.unit
def test_codes_sharing_a_status_are_documented_together() -> None:
    responses = error_responses(ErrorCode.INVALID_INPUT, ErrorCode.INVALID_TASK)

    assert list(responses) == [422]
    assert responses[422]["description"] == "INVALID_INPUT, INVALID_TASK"


@pytest.mark.unit
def test_the_envelope_refuses_an_unknown_field() -> None:
    payload: dict[str, Any] = {"error": {"code": "TIMEOUT", "message": "trop long", "retry": True}}

    with pytest.raises(ValidationError):
        ErrorResponse.model_validate(payload)
