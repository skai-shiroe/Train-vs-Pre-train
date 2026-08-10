"""Observation points of the training loop.

The trainer owns the optimisation, nothing else. Logging, history collection
and any future MLflow tracing plug in here, so that adding an observer never
requires touching the loop itself.

Callbacks observe, they do not steer. A callback that needed to stop a run
would be hiding a control decision inside an observer, so early stopping is a
first class part of the trainer instead.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

from src.training.state import EpochMetrics, TrainingResult, TrainingState


class Callback:
    """No-op base class. Subclasses override the hooks they care about."""

    def on_train_begin(self, state: TrainingState) -> None:
        """Called once, before the first epoch.

        Args:
            state: The run state.
        """

    def on_epoch_begin(self, state: TrainingState) -> None:
        """Called at the start of every epoch.

        Args:
            state: The run state.
        """

    def on_step_end(self, state: TrainingState) -> None:
        """Called after every optimisation step.

        Args:
            state: The run state.
        """

    def on_epoch_end(self, state: TrainingState, metrics: EpochMetrics) -> None:
        """Called after the validation pass of every epoch.

        Args:
            state: The run state.
            metrics: What the epoch produced.
        """

    def on_train_end(self, state: TrainingState, result: TrainingResult) -> None:
        """Called once, after the last epoch.

        Args:
            state: The run state.
            result: What the whole run produced.
        """


class CallbackList(Callback):
    """Fan a single event out to several callbacks, in registration order."""

    def __init__(self, callbacks: Iterable[Callback] = ()) -> None:
        """Build the list.

        Args:
            callbacks: The callbacks to notify.
        """
        self._callbacks: tuple[Callback, ...] = tuple(callbacks)

    def __len__(self) -> int:
        """Return the number of registered callbacks.

        Returns:
            The callback count.
        """
        return len(self._callbacks)

    def on_train_begin(self, state: TrainingState) -> None:
        """Notify every callback.

        Args:
            state: The run state.
        """
        for callback in self._callbacks:
            callback.on_train_begin(state)

    def on_epoch_begin(self, state: TrainingState) -> None:
        """Notify every callback.

        Args:
            state: The run state.
        """
        for callback in self._callbacks:
            callback.on_epoch_begin(state)

    def on_step_end(self, state: TrainingState) -> None:
        """Notify every callback.

        Args:
            state: The run state.
        """
        for callback in self._callbacks:
            callback.on_step_end(state)

    def on_epoch_end(self, state: TrainingState, metrics: EpochMetrics) -> None:
        """Notify every callback.

        Args:
            state: The run state.
            metrics: What the epoch produced.
        """
        for callback in self._callbacks:
            callback.on_epoch_end(state, metrics)

    def on_train_end(self, state: TrainingState, result: TrainingResult) -> None:
        """Notify every callback.

        Args:
            state: The run state.
            result: What the whole run produced.
        """
        for callback in self._callbacks:
            callback.on_train_end(state, result)


class LoggingCallback(Callback):
    """Report progress on the standard logger."""

    def __init__(self, *, every_steps: int = 50, logger: logging.Logger | None = None) -> None:
        """Build the callback.

        Args:
            every_steps: Interval, in optimisation steps, between two records.
            logger: Destination logger. Defaults to ``syntra.training``.

        Raises:
            ValueError: If the interval is not strictly positive.
        """
        if every_steps <= 0:
            raise ValueError(f"every_steps must be strictly positive, got {every_steps}.")
        self._every_steps = every_steps
        self._logger = logger or logging.getLogger("syntra.training")

    def on_train_begin(self, state: TrainingState) -> None:
        """Announce the size of the run.

        Args:
            state: The run state.
        """
        self._logger.info(
            "Training starts: %d epochs, %d optimisation steps planned",
            state.epochs,
            state.total_steps,
        )

    def on_step_end(self, state: TrainingState) -> None:
        """Log a progress record at the configured interval.

        Args:
            state: The run state.
        """
        if state.global_step % self._every_steps != 0:
            return
        self._logger.info(
            "step %d/%d  loss %.4f  lr %.2e  grad_norm %.2f",
            state.global_step,
            state.total_steps,
            state.running_loss,
            state.learning_rate,
            state.grad_norm,
        )

    def on_epoch_end(self, state: TrainingState, metrics: EpochMetrics) -> None:
        """Log the summary of an epoch.

        Args:
            state: The run state.
            metrics: What the epoch produced.
        """
        self._logger.info(
            "epoch %d/%d  train_loss %.4f  val_loss %.4f  %.1fs%s",
            metrics.epoch + 1,
            state.epochs,
            metrics.train_loss,
            metrics.validation_loss,
            metrics.duration_seconds,
            "  (best so far)" if metrics.improved else "",
        )

    def on_train_end(self, state: TrainingState, result: TrainingResult) -> None:
        """Log the outcome of the run.

        Args:
            state: The run state.
            result: What the whole run produced.
        """
        reason = "early stopping" if result.stopped_early else "epoch budget reached"
        self._logger.info(
            "Training over (%s): best val_loss %.4f at epoch %d, %d steps in %.1fs",
            reason,
            result.best_validation_loss,
            result.best_epoch + 1,
            result.global_step,
            result.duration_seconds,
        )


class HistoryCallback(Callback):
    """Collect the curves the report figures are drawn from.

    Section 18 requires the training and validation loss curves to be produced
    automatically. Recording them here means the figures are read from what the
    run actually measured, never retyped.
    """

    def __init__(self, *, path: Path | None = None) -> None:
        """Build the callback.

        Args:
            path: Optional file the history is written to at the end of the
                run, as JSON.
        """
        self._path = path
        self.step_losses: list[float] = []
        self.train_losses: list[float] = []
        self.validation_losses: list[float] = []
        self.learning_rates: list[float] = []
        self.epochs: list[EpochMetrics] = []

    def on_step_end(self, state: TrainingState) -> None:
        """Record the loss and the rate of one step.

        Args:
            state: The run state.
        """
        self.step_losses.append(state.last_loss)
        self.learning_rates.append(state.learning_rate)

    def on_epoch_end(self, state: TrainingState, metrics: EpochMetrics) -> None:
        """Record the summary of one epoch.

        Args:
            state: The run state.
            metrics: What the epoch produced.
        """
        self.epochs.append(metrics)
        self.train_losses.append(metrics.train_loss)
        self.validation_losses.append(metrics.validation_loss)

    def on_train_end(self, state: TrainingState, result: TrainingResult) -> None:
        """Write the history to disk when a path was supplied.

        Args:
            state: The run state.
            result: What the whole run produced.
        """
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"result": result.to_dict(), "curves": self.to_dict()}
        with self._path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def to_dict(self) -> dict[str, list[float]]:
        """Render the collected curves as a mapping.

        Returns:
            One list per curve, indexed by step for the step level curves and
            by epoch for the epoch level ones.
        """
        return {
            "step_loss": list(self.step_losses),
            "learning_rate": list(self.learning_rates),
            "train_loss": list(self.train_losses),
            "validation_loss": list(self.validation_losses),
        }


def default_callbacks(
    *,
    log_every_steps: int = 50,
    history_path: Path | None = None,
) -> Sequence[Callback]:
    """Return the callbacks every run installs unless told otherwise.

    Args:
        log_every_steps: Interval between two progress records.
        history_path: Optional destination of the history file.

    Returns:
        A logging callback followed by a history callback.
    """
    return (
        LoggingCallback(every_steps=log_every_steps),
        HistoryCallback(path=history_path),
    )
