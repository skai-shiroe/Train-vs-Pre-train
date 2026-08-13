"""Unit tests of the live half of the tracking.

Two properties are under test. A run must be followable while it trains, which
is what the callback provides, and following it must never be able to end it:
a store that dies at step 50 of 6250 costs the live view and nothing else.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from src.tracking.client import NullTracker
from src.tracking.live import (
    LiveMetricsCallback,
    NullLiveRun,
    finish_safely,
    open_live_run,
)
from src.tracking.payload import TrackedRun
from src.training.state import EpochMetrics, TrainingState

pytestmark = pytest.mark.unit


class RecordingRun:
    """Open run keeping every point it was handed."""

    def __init__(self) -> None:
        self.points: list[tuple[int, dict[str, float]]] = []
        self.closed: TrackedRun | None = None

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        self.points.append((step, dict(metrics)))

    def finish(self, payload: TrackedRun) -> str | None:
        self.closed = payload
        return "run-1"


class BrokenRun:
    """Open run standing in for a store that became unreachable."""

    def __init__(self) -> None:
        self.attempts = 0

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        self.attempts += 1
        raise ConnectionError("the store is unreachable")

    def finish(self, payload: TrackedRun) -> str | None:
        raise ConnectionError("the store is unreachable")


class MirrorOnlyTracker:
    """Tracker that can mirror a finished run and nothing else."""

    def log(self, payload: TrackedRun) -> str | None:
        return "run-1"


class UnreachableTracker:
    """Tracker whose store refuses the connection at opening."""

    def log(self, payload: TrackedRun) -> str | None:
        return None

    def open(self, name: str) -> NullLiveRun:
        raise ConnectionError("the store is unreachable")


def state(*, global_step: int) -> TrainingState:
    """Build a loop state at a given step.

    Args:
        global_step: Step the state reports.

    Returns:
        The state.
    """
    return TrainingState(
        epoch=0,
        epochs=3,
        global_step=global_step,
        total_steps=100,
        learning_rate=0.0003,
        last_loss=2.5,
        running_loss=2.7,
        grad_norm=1.2,
    )


def epoch_metrics(*, epoch: int) -> EpochMetrics:
    """Build the summary of one finished epoch.

    Args:
        epoch: Zero based epoch index.

    Returns:
        The metrics.
    """
    return EpochMetrics(
        epoch=epoch,
        train_loss=4.0,
        validation_loss=3.5,
        learning_rate=0.0002,
        steps=50,
        duration_seconds=120.0,
        improved=True,
    )


def test_the_loop_is_streamed_at_the_declared_interval() -> None:
    run = RecordingRun()
    callback = LiveMetricsCallback(run, every_steps=2)

    for step in (1, 2, 3, 4):
        callback.on_step_end(state(global_step=step))

    assert [step for step, _ in run.points] == [2, 4]
    assert set(run.points[0][1]) == {
        "step_loss",
        "step_running_loss",
        "step_learning_rate",
        "step_grad_norm",
    }


def test_the_epoch_losses_are_streamed_against_the_epoch_index() -> None:
    # The early stopping signal is the one worth watching while a run goes, and
    # it is indexed by epoch, not by optimisation step.
    run = RecordingRun()
    callback = LiveMetricsCallback(run, every_steps=50)

    callback.on_epoch_end(state(global_step=50), epoch_metrics(epoch=1))

    assert run.points == [(1, {"epoch_train_loss": 4.0, "epoch_validation_loss": 3.5})]


def test_the_streamed_names_never_collide_with_the_record_curves() -> None:
    # build_payload writes train_loss and validation_loss against the epoch
    # index. A stream using those names would draw two abscissae as one curve.
    run = RecordingRun()
    callback = LiveMetricsCallback(run, every_steps=1)

    callback.on_step_end(state(global_step=1))
    callback.on_epoch_end(state(global_step=1), epoch_metrics(epoch=0))

    streamed = {name for _, metrics in run.points for name in metrics}

    assert streamed.isdisjoint({"train_loss", "validation_loss", "learning_rate"})


def test_a_store_that_dies_mid_training_is_asked_once(capsys: pytest.CaptureFixture[str]) -> None:
    run = BrokenRun()
    callback = LiveMetricsCallback(run, every_steps=1)

    for step in (1, 2, 3, 4, 5):
        callback.on_step_end(state(global_step=step))

    assert run.attempts == 1
    assert capsys.readouterr().err.count("live tracking stopped") == 1


def test_a_tracker_that_only_mirrors_opens_nothing() -> None:
    # Not a degraded mode: the run is traced at the end, which is what the
    # store did before a live run existed.
    assert open_live_run(MirrorOnlyTracker(), "scratch_100") is None


def test_no_tracker_opens_nothing() -> None:
    assert open_live_run(None, "scratch_100") is None


def test_turning_tracking_off_still_gives_a_run_to_stream_into() -> None:
    # Otherwise the training loop would carry a branch on whether tracking is
    # on, which is exactly what NullTracker exists to avoid.
    live = open_live_run(NullTracker(), "scratch_100")

    assert isinstance(live, NullLiveRun)
    assert live.finish(TrackedRun(name="scratch_100")) is None


def test_a_store_unreachable_at_opening_costs_the_live_view_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert open_live_run(UnreachableTracker(), "scratch_100") is None
    assert "live tracking unavailable for scratch_100" in capsys.readouterr().err


def test_a_failure_while_closing_is_a_warning_not_a_crash(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = finish_safely(BrokenRun(), TrackedRun(name="scratch_100"))

    assert result is None
    assert "tracking failed for scratch_100" in capsys.readouterr().err


def test_the_streaming_interval_must_be_strictly_positive() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        LiveMetricsCallback(NullLiveRun(), every_steps=0)
