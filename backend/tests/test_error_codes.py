"""Unit tests for the error contract defined in section 26."""

from __future__ import annotations

import pytest

from backend.app.core.errors import (
    ERROR_MESSAGES,
    ERROR_STATUS,
    ErrorCode,
    InferenceTimeoutError,
    InvalidInputError,
    ModelNotFoundError,
    ModelNotReadyError,
    SyntraError,
)

# Every concrete error the services are allowed to raise.
CONCRETE_ERRORS = [
    subclass for subclass in SyntraError.__subclasses__() if subclass is not SyntraError
]


@pytest.mark.unit
def test_every_code_has_a_status() -> None:
    assert set(ERROR_STATUS) == set(ErrorCode)


@pytest.mark.unit
def test_every_code_has_a_default_message() -> None:
    assert set(ERROR_MESSAGES) == set(ErrorCode)


@pytest.mark.unit
def test_no_default_message_is_empty() -> None:
    assert all(message.strip() for message in ERROR_MESSAGES.values())


@pytest.mark.unit
@pytest.mark.parametrize("error_class", CONCRETE_ERRORS, ids=lambda cls: cls.__name__)
def test_each_error_class_maps_to_a_distinct_code(error_class: type[SyntraError]) -> None:
    error = error_class()
    assert error.code in ErrorCode
    assert error.status_code == ERROR_STATUS[error.code]


@pytest.mark.unit
def test_concrete_errors_cover_every_code() -> None:
    covered = {error_class.code for error_class in CONCRETE_ERRORS}
    assert covered == set(ErrorCode)


@pytest.mark.unit
def test_payload_matches_the_documented_shape() -> None:
    payload = ModelNotFoundError().to_payload()

    assert payload == {
        "error": {
            "code": "MODEL_NOT_FOUND",
            "message": ERROR_MESSAGES[ErrorCode.MODEL_NOT_FOUND],
        }
    }


@pytest.mark.unit
def test_custom_message_and_details_are_propagated() -> None:
    error = InvalidInputError("Texte source vide.", details={"field": "text"})
    payload = error.to_payload()

    assert payload["error"]["message"] == "Texte source vide."
    assert payload["error"]["details"] == {"field": "text"}


@pytest.mark.unit
def test_status_codes_follow_the_specification() -> None:
    assert ModelNotFoundError().status_code == 404
    assert ModelNotReadyError().status_code == 503
    assert InferenceTimeoutError().status_code == 504
    assert InvalidInputError().status_code == 422


@pytest.mark.unit
def test_errors_are_catchable_as_the_base_class() -> None:
    with pytest.raises(SyntraError):
        raise ModelNotReadyError


@pytest.mark.unit
def test_error_code_is_a_plain_string_for_json() -> None:
    assert ErrorCode.TIMEOUT == "TIMEOUT"
    assert f"{ErrorCode.TIMEOUT}" == "TIMEOUT"
