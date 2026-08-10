"""Unit tests for the logs and the request middleware of section 25."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.monitoring.logging import JsonFormatter, configure_logging

PREFIX = "/api/v1"


def record(message: str = "un evenement", **extra: object) -> logging.LogRecord:
    """Build a record the way ``logging`` builds one."""
    built = logging.LogRecord("syntra.test", logging.INFO, __file__, 10, message, None, None)
    for key, value in extra.items():
        setattr(built, key, value)
    return built


# -- The formatter ----------------------------------------------------------


@pytest.mark.unit
def test_a_record_is_rendered_as_one_json_object() -> None:
    payload = json.loads(JsonFormatter().format(record()))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "syntra.test"
    assert payload["message"] == "un evenement"
    assert payload["time"]


@pytest.mark.unit
def test_what_a_call_site_attached_reaches_the_line() -> None:
    payload = json.loads(JsonFormatter().format(record(model="scratch", input_chars=1200)))

    assert payload["model"] == "scratch"
    assert payload["input_chars"] == 1200


@pytest.mark.unit
def test_a_value_that_is_not_serialisable_is_rendered_rather_than_dropped() -> None:
    payload = json.loads(JsonFormatter().format(record(device=Path("/tmp/x"))))

    assert "x" in payload["device"]


@pytest.mark.unit
def test_an_exception_travels_with_its_record() -> None:
    try:
        raise ZeroDivisionError("division par zero")
    except ZeroDivisionError:
        import sys

        failed = record()
        failed.exc_info = sys.exc_info()

    payload = json.loads(JsonFormatter().format(failed))

    assert "ZeroDivisionError" in payload["exception"]


# -- Installing the handler -------------------------------------------------


@pytest.mark.unit
def test_configuring_twice_leaves_one_handler() -> None:
    settings = Settings(_env_file=None)
    root = logging.getLogger()

    first = configure_logging(settings)
    second = configure_logging(settings)

    assert first not in root.handlers
    assert second in root.handlers


@pytest.mark.unit
def test_a_handler_installed_by_someone_else_survives() -> None:
    root = logging.getLogger()
    foreign = logging.NullHandler()
    root.addHandler(foreign)

    try:
        configure_logging(Settings(_env_file=None))
        assert foreign in root.handlers
    finally:
        root.removeHandler(foreign)


@pytest.mark.unit
def test_plain_logs_are_not_json() -> None:
    handler = configure_logging(Settings(_env_file=None, log_json=False))

    assert not isinstance(handler.formatter, JsonFormatter)


# -- The middleware ---------------------------------------------------------


@pytest.mark.unit
def test_a_request_is_logged_once_with_its_duration(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    app = create_app(Settings(_env_file=None, registry_root=tmp_path / "registry"))

    with (
        caplog.at_level(logging.INFO, logger="backend.app.monitoring.middleware"),
        TestClient(app) as client,
    ):
        client.get(f"{PREFIX}/health")

    lines = [entry for entry in caplog.records if entry.message == "request"]
    assert len(lines) == 1
    assert lines[0].__dict__["status"] == 200
    assert lines[0].__dict__["duration_ms"] >= 0.0


@pytest.mark.unit
def test_the_query_string_never_reaches_the_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    app = create_app(Settings(_env_file=None, registry_root=tmp_path / "registry"))

    with (
        caplog.at_level(logging.INFO, logger="backend.app.monitoring.middleware"),
        TestClient(app) as client,
    ):
        client.get(f"{PREFIX}/health?document=texte-confidentiel")

    line = next(entry for entry in caplog.records if entry.message == "request")
    assert line.__dict__["path"] == f"{PREFIX}/health"
    # The whole rendered line, not only the field: an extra added later must
    # not smuggle the query back in.
    assert "confidentiel" not in JsonFormatter().format(line)


@pytest.mark.unit
def test_a_failing_request_is_logged_with_its_traceback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    app = create_app(Settings(_env_file=None, registry_root=tmp_path / "registry"))

    @app.get("/boom")
    async def boom() -> None:
        raise ZeroDivisionError("interne")

    with (
        caplog.at_level(logging.ERROR, logger="backend.app.monitoring.middleware"),
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        client.get("/boom")

    failed = next(entry for entry in caplog.records if entry.message == "request failed")
    assert failed.exc_info is not None
