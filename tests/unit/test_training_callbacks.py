"""Unit tests for the training callbacks."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.training.callbacks import (
    Callback,
    CallbackList,
    HistoryCallback,
    LoggingCallback,
    default_callbacks,
)
from src.training.state import EpochMetrics, TrainingResult, TrainingState


class RecordingCallback(Callback):
    """Record the order in which the hooks fire."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_train_begin(self, state: TrainingState) -> None:
        self.events.append("train_begin")

    def on_epoch_begin(self, state: TrainingState) -> None:
        self.events.append("epoch_begin")

    def on_step_end(self, state: TrainingState) -> None:
        self.events.append("step_end")

    def on_epoch_end(self, state: TrainingState, metrics: EpochMetrics) -> None:
        self.events.append("epoch_end")

    def on_train_end(self, state: TrainingState, result: TrainingResult) -> None:
        self.events.append("train_end")


def make_metrics(epoch: int = 0, *, validation_loss: float = 1.0) -> EpochMetrics:
    """Return metrics for one epoch."""
    return EpochMetrics(
        epoch=epoch,
        train_loss=2.0,
        validation_loss=validation_loss,
        learning_rate=1e-4,
        steps=10,
        duration_seconds=1.5,
        improved=True,
    )


# ---------------------------------------------------------------------------
# The base class and the fan out
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_base_callback_does_nothing() -> None:
    callback = Callback()
    state = TrainingState()

    callback.on_train_begin(state)
    callback.on_epoch_begin(state)
    callback.on_step_end(state)
    callback.on_epoch_end(state, make_metrics())
    callback.on_train_end(state, TrainingResult())


@pytest.mark.unit
def test_every_callback_receives_every_event() -> None:
    first, second = RecordingCallback(), RecordingCallback()
    callbacks = CallbackList([first, second])
    state = TrainingState()

    callbacks.on_train_begin(state)
    callbacks.on_epoch_begin(state)
    callbacks.on_step_end(state)
    callbacks.on_epoch_end(state, make_metrics())
    callbacks.on_train_end(state, TrainingResult())

    expected = ["train_begin", "epoch_begin", "step_end", "epoch_end", "train_end"]
    assert first.events == second.events == expected


@pytest.mark.unit
def test_an_empty_list_is_harmless() -> None:
    callbacks = CallbackList()

    callbacks.on_train_begin(TrainingState())

    assert len(callbacks) == 0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_non_positive_interval_is_refused() -> None:
    with pytest.raises(ValueError, match="every_steps must be strictly positive"):
        LoggingCallback(every_steps=0)


@pytest.mark.unit
def test_progress_is_logged_at_the_configured_interval(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("syntra.test.progress")
    callback = LoggingCallback(every_steps=5, logger=logger)
    state = TrainingState(total_steps=20)

    with caplog.at_level(logging.INFO, logger=logger.name):
        for step in range(1, 11):
            state.global_step = step
            callback.on_step_end(state)

    assert len(caplog.records) == 2


@pytest.mark.unit
def test_the_run_boundaries_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("syntra.test.boundaries")
    callback = LoggingCallback(logger=logger)
    state = TrainingState(epochs=3, total_steps=30)

    with caplog.at_level(logging.INFO, logger=logger.name):
        callback.on_train_begin(state)
        callback.on_epoch_end(state, make_metrics())
        callback.on_train_end(state, TrainingResult(stopped_early=True, best_epoch=1))

    messages = [record.getMessage() for record in caplog.records]
    assert "30 optimisation steps planned" in messages[0]
    assert "best so far" in messages[1]
    assert "early stopping" in messages[2]


@pytest.mark.unit
def test_a_completed_run_is_not_reported_as_stopped_early(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("syntra.test.completed")
    callback = LoggingCallback(logger=logger)

    with caplog.at_level(logging.INFO, logger=logger.name):
        callback.on_train_end(TrainingState(), TrainingResult(stopped_early=False))

    assert "epoch budget reached" in caplog.records[0].getMessage()


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_history_collects_the_curves() -> None:
    history = HistoryCallback()
    state = TrainingState()

    for step, loss in enumerate([3.0, 2.5, 2.0], start=1):
        state.global_step = step
        state.last_loss = loss
        state.learning_rate = 1e-4
        history.on_step_end(state)
    history.on_epoch_end(state, make_metrics(0, validation_loss=2.2))
    history.on_epoch_end(state, make_metrics(1, validation_loss=1.9))

    curves = history.to_dict()
    assert curves["step_loss"] == [3.0, 2.5, 2.0]
    assert curves["learning_rate"] == [1e-4, 1e-4, 1e-4]
    assert curves["validation_loss"] == [2.2, 1.9]
    assert curves["train_loss"] == [2.0, 2.0]
    assert len(history.epochs) == 2


@pytest.mark.unit
def test_the_history_is_written_when_a_path_is_given(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "history.json"
    history = HistoryCallback(path=path)
    state = TrainingState()
    state.last_loss = 1.0
    history.on_step_end(state)
    history.on_epoch_end(state, make_metrics())

    history.on_train_end(state, TrainingResult(best_epoch=0, best_validation_loss=1.0))

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["curves"]["step_loss"] == [1.0]
    assert payload["result"]["best_validation_loss"] == 1.0


@pytest.mark.unit
def test_the_history_stays_in_memory_without_a_path() -> None:
    history = HistoryCallback()

    history.on_train_end(TrainingState(), TrainingResult())

    assert history.to_dict()["step_loss"] == []


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_default_callbacks_log_and_record() -> None:
    callbacks = default_callbacks(log_every_steps=10)

    assert isinstance(callbacks[0], LoggingCallback)
    assert isinstance(callbacks[1], HistoryCallback)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_result_renders_as_a_serialisable_mapping() -> None:
    result = TrainingResult(
        epochs=(make_metrics(),),
        best_epoch=0,
        best_validation_loss=1.0,
        global_step=10,
        best_checkpoint=Path("runs") / "best.pt",
    )

    payload = result.to_dict()

    assert json.loads(json.dumps(payload))["epochs"][0]["validation_loss"] == 1.0
    assert payload["best_checkpoint"] is not None


@pytest.mark.unit
def test_a_run_without_checkpoint_reports_none() -> None:
    assert TrainingResult().to_dict()["best_checkpoint"] is None
