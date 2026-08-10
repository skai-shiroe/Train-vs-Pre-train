"""Unit tests for the application assembly of sections 21, 25 and 26."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.core.errors import ErrorCode, InferenceTimeoutError
from backend.app.main import (
    code_for_status,
    create_app,
    handle_http_exception,
    handle_syntra_error,
    handle_validation_error,
)
from backend.app.monitoring.middleware import REQUEST_ID_HEADER

PREFIX = "/api/v1"


def build_settings(tmp_path: Path, **overrides: object) -> Settings:
    """Return settings pointing at an empty registry outside the repository."""
    return Settings(_env_file=None, registry_root=tmp_path / "registry", **overrides)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """Return a client over a started application with an empty registry."""
    with TestClient(create_app(build_settings(tmp_path))) as started:
        yield started


# -- Probes -----------------------------------------------------------------


@pytest.mark.unit
def test_liveness_answers_without_touching_the_registry(client: TestClient) -> None:
    response = client.get(f"{PREFIX}/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": "Syntra",
        "environment": "local",
        "api_version": "1.0.0",
    }


@pytest.mark.unit
def test_readiness_refuses_while_no_model_is_loaded(client: TestClient) -> None:
    response = client.get(f"{PREFIX}/ready")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == ErrorCode.MODEL_NOT_READY.value


@pytest.mark.unit
def test_an_instance_that_never_started_is_not_ready(tmp_path: Path) -> None:
    # No context manager, so the lifespan never ran and the state is empty.
    response = TestClient(create_app(build_settings(tmp_path))).get(f"{PREFIX}/ready")

    assert response.status_code == 503
    assert response.json()["error"]["details"] == {"reason": "startup incomplete"}


# -- The error envelope -----------------------------------------------------


@pytest.mark.unit
def test_an_unknown_route_is_not_reported_as_a_missing_model(client: TestClient) -> None:
    response = client.get(f"{PREFIX}/absent")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == ErrorCode.INVALID_INPUT.value


@pytest.mark.unit
def test_a_method_that_is_not_allowed_stays_in_the_envelope(client: TestClient) -> None:
    response = client.get(f"{PREFIX}/predict")

    assert response.status_code == 405
    assert set(response.json()["error"]) == {"code", "message"}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (404, ErrorCode.INVALID_INPUT),
        (405, ErrorCode.INVALID_INPUT),
        (502, ErrorCode.INFERENCE_FAILED),
    ],
)
def test_a_framework_status_maps_to_a_code_of_section_26(status: int, expected: ErrorCode) -> None:
    assert code_for_status(status) is expected


@pytest.mark.unit
def test_a_rejected_payload_never_echoes_what_was_submitted(client: TestClient) -> None:
    response = client.post(f"{PREFIX}/predict", json={"text": "", "num_beams": 99})

    assert response.status_code == 422
    body = response.json()["error"]
    assert body["code"] == ErrorCode.INVALID_INPUT.value
    assert {entry["field"] for entry in body["details"]["fields"]} == {"text", "num_beams"}
    assert all(set(entry) == {"field", "message"} for entry in body["details"]["fields"])


@pytest.mark.unit
def test_a_rule_over_the_whole_payload_names_the_body(client: TestClient) -> None:
    # The exclusion between sampling and beam search belongs to no single field.
    # An empty name is something a client can neither display nor highlight.
    response = client.post(f"{PREFIX}/predict", json={"text": "un document", "num_beams": 4})

    assert response.status_code == 422
    fields = response.json()["error"]["details"]["fields"]
    assert [entry["field"] for entry in fields] == ["body"]
    assert "exclusifs" in fields[0]["message"]


@pytest.mark.unit
def test_an_unexpected_exception_is_answered_without_its_message(tmp_path: Path) -> None:
    app = create_app(build_settings(tmp_path))

    @app.get("/boom")
    async def boom() -> None:
        raise ZeroDivisionError("chemin interne /var/secrets")

    with TestClient(app, raise_server_exceptions=False) as started:
        response = started.get("/boom")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == ErrorCode.INFERENCE_FAILED.value
    assert "secrets" not in response.text


@pytest.mark.unit
def test_an_application_error_keeps_its_status_and_its_details(tmp_path: Path) -> None:
    app = create_app(build_settings(tmp_path))

    @app.get("/slow")
    async def slow() -> None:
        raise InferenceTimeoutError(details={"model": "scratch", "timeout_s": 30.0})

    with TestClient(app) as started:
        response = started.get("/slow")

    assert response.status_code == 504
    assert response.json()["error"]["details"] == {"model": "scratch", "timeout_s": 30.0}


# -- Surface ----------------------------------------------------------------


@pytest.mark.unit
def test_the_interactive_documentation_can_be_switched_off(tmp_path: Path) -> None:
    app = create_app(build_settings(tmp_path, docs_enabled=False))

    with TestClient(app) as started:
        assert started.get("/docs").status_code == 404
        assert started.get("/openapi.json").status_code == 404


@pytest.mark.unit
def test_the_documentation_is_exposed_by_default(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


@pytest.mark.unit
def test_the_routes_sit_behind_the_configured_prefix(tmp_path: Path) -> None:
    app = create_app(build_settings(tmp_path, api_v1_prefix="/v1"))

    with TestClient(app) as started:
        assert started.get("/v1/health").status_code == 200
        assert started.get("/api/v1/health").status_code == 404


@pytest.mark.unit
def test_a_declared_origin_is_allowed(tmp_path: Path) -> None:
    app = create_app(build_settings(tmp_path, cors_origins=["http://localhost:3000"]))

    with TestClient(app) as started:
        response = started.get(f"{PREFIX}/health", headers={"Origin": "http://localhost:3000"})

    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


@pytest.mark.unit
def test_no_cross_origin_header_is_sent_when_none_is_configured(client: TestClient) -> None:
    response = client.get(f"{PREFIX}/health", headers={"Origin": "http://localhost:3000"})

    assert "access-control-allow-origin" not in response.headers


@pytest.mark.unit
def test_every_response_carries_a_correlation_identifier(client: TestClient) -> None:
    generated = client.get(f"{PREFIX}/health")
    supplied = client.get(f"{PREFIX}/health", headers={REQUEST_ID_HEADER: "trace-42"})

    assert generated.headers[REQUEST_ID_HEADER]
    assert supplied.headers[REQUEST_ID_HEADER] == "trace-42"


# -- Startup ----------------------------------------------------------------


@pytest.mark.unit
def test_an_unwritten_registry_backend_stops_the_instance(tmp_path: Path) -> None:
    app = create_app(build_settings(tmp_path, registry_backend="mlflow"))

    with pytest.raises(NotImplementedError, match="SYNTRA_REGISTRY_BACKEND"), TestClient(app):
        pass  # pragma: no cover - the lifespan raises before the body runs


@pytest.mark.unit
def test_an_unknown_registry_backend_stops_the_instance(tmp_path: Path) -> None:
    app = create_app(build_settings(tmp_path, registry_backend="postgres"))

    with pytest.raises(ValueError, match="Unknown registry backend"), TestClient(app):
        pass  # pragma: no cover - the lifespan raises before the body runs


@pytest.mark.unit
def test_an_empty_registry_has_nothing_to_compare(client: TestClient) -> None:
    response = client.post(f"{PREFIX}/compare", json={"text": "un document"})

    assert response.status_code == 404
    assert response.json()["error"]["message"].startswith("Le registre ne contient aucun modele")


@pytest.mark.unit
@pytest.mark.parametrize(
    "handler",
    [handle_syntra_error, handle_validation_error, handle_http_exception],
)
async def test_a_handler_refuses_an_exception_it_does_not_render(handler: Any) -> None:
    # Registered for one type each. Reaching one with anything else would mean
    # the registration was wrong, and swallowing it would answer a 500 with a
    # payload describing an error that never happened.
    unexpected = ZeroDivisionError("pas une erreur applicative")

    with pytest.raises(ZeroDivisionError):
        await handler(None, unexpected)


@pytest.mark.unit
def test_lazy_loading_starts_without_touching_a_model(tmp_path: Path) -> None:
    app: FastAPI = create_app(build_settings(tmp_path, eager_load_models=False))

    with TestClient(app) as started:
        assert started.get(f"{PREFIX}/models").json() == {
            "models": [],
            "aliases": {},
            "default": "champion",
        }
