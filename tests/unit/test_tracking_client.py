"""Unit tests of the tracker selection and of the failure policy.

The property under test is that a tracking failure costs the mirror of a
measurement and never the measurement itself. A run that trained for six hours
must not end on a crash raised by a store.
"""

from __future__ import annotations

import pytest

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


def test_the_tracking_uri_is_left_to_mlflow_when_absent() -> None:
    # Section 21.1 forbids reading the environment outside the settings module.
    # Passing nothing lets MLflow resolve its own, which is what makes
    # MLFLOW_TRACKING_URI work without this project ever reading it.
    assert MlflowTracker().tracking_uri is None
    assert (
        MlflowTracker(tracking_uri="http://localhost:5000").tracking_uri == "http://localhost:5000"
    )
