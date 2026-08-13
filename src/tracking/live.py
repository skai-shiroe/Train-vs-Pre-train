"""Following a run while it is still going.

Section 19 asks for every experiment to reach MLflow, and :mod:`src.tracking.payload`
answers it out of the record the runner has already written. That order is what
makes the mirror trustworthy, and it is also why nothing appears in the store
before an experiment ends: on a training that runs for hours, the interface
stays empty for hours and the only way to follow the loop is to watch
checkpoint timestamps.

This module adds the missing half without touching the first one. A run is
opened before the training starts, a callback streams what the loop measures
into it, and the finished payload closes the same run. One run per experiment,
readable while it fills.

**The live series and the result series are not the same series.** What the loop
streams is named ``step_*`` and ``epoch_*``; it is the observation the trainer
made at that moment. What :meth:`LiveRun.finish` writes is named after the
record and comes from it alone. They do not share an axis either: the stream is
indexed by optimisation step, the record curves by epoch index. Merging them
would draw two incompatible x axes as one curve, and would let an observation be
read as a result.

**A run exists in the store before its record exists on disk.** This is the one
rule of :mod:`src.tracking.client` that a live run relaxes, and it is worth
being explicit about. What is relaxed is the ordering; what is not relaxed is
the source of the final numbers, which is still the record and nothing else. An
open run carries no score until it is closed, so it cannot be mistaken for a
measurement any more than a ``PARTIAL`` run can.

**A tracking failure never fails a run.** Same rule as the mirror, applied per
call. A store that becomes unreachable mid training stops being asked rather
than printing one warning per logged step: the first failure disables the
callback for the rest of the run.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from src.tracking.payload import TrackedRun
from src.training.callbacks import Callback
from src.training.state import EpochMetrics, TrainingState


@runtime_checkable
class LiveRun(Protocol):
    """A run that is open in the store while the training is still going."""

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Append one point per metric to the open run.

        Args:
            metrics: The values measured at this point.
            step: Abscissa the values are recorded against.
        """

    def finish(self, payload: TrackedRun) -> str | None:
        """Write the finished run and close it.

        Args:
            payload: What the record says, which is the only source of the
                numbers that stay in the store as the result.

        Returns:
            The identifier the store gave the run, or ``None`` when nothing was
            recorded.
        """


@runtime_checkable
class LiveTracker(Protocol):
    """A tracker that can open a run and leave it open.

    Separate from :class:`src.tracking.client.Tracker` on purpose. A tracker
    that only knows how to mirror a finished run stays a valid tracker, and
    :func:`open_live_run` simply finds nothing to open, so the run is traced at
    the end as before instead of failing.
    """

    def open(self, name: str) -> LiveRun:
        """Start a run and return the handle that fills it.

        Args:
            name: Name the run appears under while it is open.

        Returns:
            The open run.
        """


class NullLiveRun:
    """Open run that records nothing, selected when tracking is turned off."""

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Discard one point.

        Args:
            metrics: The values that would have been recorded.
            step: The abscissa they would have been recorded against.
        """

    def finish(self, payload: TrackedRun) -> str | None:
        """Discard the finished run.

        Args:
            payload: The run that would have been recorded.

        Returns:
            ``None``, always. The absence of an identifier is what tells the
            caller that nothing was stored.
        """
        return None


class LiveMetricsCallback(Callback):
    """Stream what the training loop measures into an open run.

    The metric names carry the granularity they were measured at, because the
    store shows them next to the curves rebuilt from the record at the end. A
    ``train_loss`` streamed against the optimisation step and a ``train_loss``
    written against the epoch index would be one series with two abscissae.
    """

    def __init__(self, run: LiveRun, *, every_steps: int = 50) -> None:
        """Build the callback.

        Args:
            run: The open run the points are sent to.
            every_steps: Interval, in optimisation steps, between two points.
                The same interval as the console log by default: a point per
                step would make the store the bottleneck of the loop.

        Raises:
            ValueError: If the interval is not strictly positive.
        """
        if every_steps <= 0:
            raise ValueError(f"every_steps must be strictly positive, got {every_steps}.")
        self._run = run
        self._every_steps = every_steps
        self._broken = False

    def on_step_end(self, state: TrainingState) -> None:
        """Send the state of the loop at the configured interval.

        Args:
            state: The run state.
        """
        if state.global_step % self._every_steps != 0:
            return
        self._send(
            {
                "step_loss": state.last_loss,
                "step_running_loss": state.running_loss,
                "step_learning_rate": state.learning_rate,
                "step_grad_norm": state.grad_norm,
            },
            step=state.global_step,
        )

    def on_epoch_end(self, state: TrainingState, metrics: EpochMetrics) -> None:
        """Send the two losses of a finished epoch.

        This is the signal early stopping decides on, so it is the one worth
        watching while the run goes.

        Args:
            state: The run state.
            metrics: What the epoch produced.
        """
        self._send(
            {
                "epoch_train_loss": metrics.train_loss,
                "epoch_validation_loss": metrics.validation_loss,
            },
            step=metrics.epoch,
        )

    def _send(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Send one point, giving up on the stream after the first failure.

        Args:
            metrics: The values measured at this point.
            step: Abscissa the values are recorded against.
        """
        if self._broken:
            return
        try:
            self._run.log_metrics(metrics, step=step)
        except Exception as error:
            # Caught on purpose, and only once. The measurement is not at
            # stake here, and a store that died at step 50 of 6250 would
            # otherwise print the same warning a hundred times.
            self._broken = True
            print(
                f"live tracking stopped at step {step}: {type(error).__name__}: {error}",
                file=sys.stderr,
            )


def open_live_run(tracker: object | None, name: str) -> LiveRun | None:
    """Open the run an experiment fills while it trains.

    Args:
        tracker: The tracker the runner was given. A tracker that cannot open a
            run, and ``None``, both yield ``None``.
        name: Name the run appears under while it is open.

    Returns:
        The open run, or ``None`` when there is nothing to follow. ``None`` is
        not a degraded mode: the run is still mirrored at the end, which is what
        the store did before this module existed.
    """
    if tracker is None or not isinstance(tracker, LiveTracker):
        return None
    try:
        return tracker.open(name)
    except Exception as error:
        # A store that cannot be reached costs the live view, not the run.
        print(
            f"live tracking unavailable for {name}: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return None


def finish_safely(run: LiveRun, payload: TrackedRun) -> str | None:
    """Close an open run, turning any failure into a warning.

    Args:
        run: The open run.
        payload: What the record says.

    Returns:
        The identifier the store gave the run, or ``None`` when the closing
        failed.
    """
    try:
        return run.finish(payload)
    except Exception as error:
        # Same contract as log_safely: the measurement is already on disk, and
        # losing its mirror is a degraded run rather than a failed one.
        print(
            f"tracking failed for {payload.name}: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return None
