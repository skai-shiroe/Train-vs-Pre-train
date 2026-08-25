"""Unit tests of the tracker selection and of the failure policy.

The property under test is that a tracking failure costs the mirror of a
measurement and never the measurement itself. A run that trained for six hours
must not end on a crash raised by a store.
"""

from __future__ import annotations

import pytest

from src.tracking import client
from src.tracking.client import (
    DEFAULT_EXPERIMENT,
    MlflowTracker,
    NullTracker,
    build_tracker,
    log_safely,
)
from src.tracking.payload import TrackedRun

pytestmark = pytest.mark.unit


class BrokenTracker:
    """Tracker standing in for an unreachable store."""

    def log(self, payload: TrackedRun) -> str | None:
        raise ConnectionError("the store is unreachable")


class RecordingTracker:
    """Tracker keeping what it was handed."""

    def __init__(self) -> None:
        self.seen: list[TrackedRun] = []

    def log(self, payload: TrackedRun) -> str | None:
        self.seen.append(payload)
        return "run-1"


def test_tracking_is_on_by_default() -> None:
    tracker = build_tracker()

    assert isinstance(tracker, MlflowTracker)
    assert tracker.experiment == DEFAULT_EXPERIMENT


def test_turning_tracking_off_selects_a_tracker_that_says_it_stored_nothing() -> None:
    # Returning None rather than a made up identifier is what keeps "off" from
    # looking like "logged".
    tracker = build_tracker(enabled=False)

    assert isinstance(tracker, NullTracker)
    assert tracker.log(TrackedRun(name="scratch_10")) is None


def test_an_unreachable_store_is_a_warning_not_a_crash(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = log_safely(BrokenTracker(), TrackedRun(name="scratch_10"))

    assert result is None
    assert "tracking failed for scratch_10" in capsys.readouterr().err


def test_a_working_store_returns_its_identifier() -> None:
    tracker = RecordingTracker()

    assert log_safely(tracker, TrackedRun(name="scratch_10")) == "run-1"
    assert [payload.name for payload in tracker.seen] == ["scratch_10"]


def test_an_explicit_uri_is_kept_as_given(monkeypatch: pytest.MonkeyPatch) -> None:
    # The command line wins over everything: --tracking-uri exists to send one
    # run somewhere else without touching the configuration of the machine.
    monkeypatch.setattr(client, "resolve_tracking_uri", lambda: "postgresql://configure@h/d")

    assert (
        MlflowTracker(tracking_uri="http://localhost:5000").tracking_uri == "http://localhost:5000"
    )


def test_the_configured_uri_is_resolved_when_none_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The store moved to PostgreSQL, so the URI carries a password and lives in
    # an ignored file rather than in the repository. What is pinned here is the
    # wiring; :mod:`tests.unit.test_tracking_store` pins the resolution itself.
    monkeypatch.setattr(client, "resolve_tracking_uri", lambda: "postgresql://configure@h/d")

    assert MlflowTracker().tracking_uri == "postgresql://configure@h/d"


def test_nothing_configured_leaves_mlflow_to_resolve_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Not an error: a fresh clone with no database still traces, into a local
    # SQLite file, and the runner prints which of the two it got.
    monkeypatch.setattr(client, "resolve_tracking_uri", lambda: None)

    assert MlflowTracker().tracking_uri is None


def test_a_local_artifact_root_becomes_an_absolute_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    # A relative root would be resolved against the working directory of
    # whoever reads the store next, which is not the one that wrote it.
    monkeypatch.setattr(client, "resolve_tracking_uri", lambda: None)
    tracker = MlflowTracker(artifact_root="mlartifacts")

    location = tracker._artifact_location()

    assert location.startswith("file://")
    assert location.endswith("/mlartifacts")


def test_an_artifact_root_that_is_already_a_uri_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client, "resolve_tracking_uri", lambda: None)
    tracker = MlflowTracker(artifact_root="s3://bucket/artefacts")

    assert tracker._artifact_location() == "s3://bucket/artefacts"
