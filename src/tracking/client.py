"""The MLflow side of the tracking, and the two ways of not using it.

Section 19 asks for MLflow. This module is the only place that imports it, so
everything upstream stays testable without a tracking store and the training
image stays the only one that needs the dependency.

**A tracking failure never fails a run.** The result of an experiment is the
record on disk; the tracking store is a mirror of it. An unreachable server, a
full disk under ``mlruns`` or a version mismatch would otherwise turn six hours
of training into a crash after the measurement was already taken. Every
tracked call goes through :func:`log_safely`, which reports the failure on the
error stream and returns nothing.

**Nothing is logged that was not written first.** The runner writes the record,
then traces it. The order matters for the same reason: the store never holds a
run the repository has no record of. :mod:`src.tracking.live` relaxes that
ordering, and only that: a run opened before its training carries no score until
the record closes it, and the numbers that stay in the store still come from the
record alone.

**Tracking is on by default and can be turned off, never faked.** Turning it off
selects :class:`NullTracker`, which returns ``None`` and says so. There is no
mode that pretends to have logged.

MLflow resolves its own tracking URI when none is given, which is what keeps
section 21.1 satisfied: no module outside ``backend/app/core/config.py`` reads
the environment itself. With nothing configured, MLflow 3 resolves to a SQLite
database in the working directory, so an experiment run offline is still traced
and ``docker compose up mlflow`` is not a prerequisite for training.

One URI does not work and is worth knowing about: a ``file://`` path. MLflow 3
put the filesystem tracking backend in maintenance mode and raises rather than
writing to it. A local store is a SQLite database, a shared one is a server.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from src.tracking.live import LiveRun, NullLiveRun
from src.tracking.payload import TrackedRun

#: Experiment the runs are grouped under, matching the default of
#: ``SYNTRA_MLFLOW_EXPERIMENT`` so the training side and the API agree without
#: sharing a settings object.
DEFAULT_EXPERIMENT = "syntra-summarization"


class Tracker(Protocol):
    """Anything that can receive a run.

    The runner depends on this and never on MLflow, which is what lets a test
    assert what would have been logged without starting a store.
    """

    def log(self, payload: TrackedRun) -> str | None:
        """Send one run to the store.

        Args:
            payload: The run to record.

        Returns:
            The identifier the store gave the run, or ``None`` when nothing was
            recorded.
        """


class NullTracker:
    """Tracker that records nothing, selected when tracking is turned off."""

    def log(self, payload: TrackedRun) -> str | None:
        """Discard a run.

        Args:
            payload: The run that would have been recorded.

        Returns:
            ``None``, always. The absence of an identifier is what tells the
            caller that nothing was stored.
        """
        return None

    def open(self, name: str) -> LiveRun:
        """Open a run that records nothing.

        Args:
            name: Name the run would have appeared under.

        Returns:
            A :class:`src.tracking.live.NullLiveRun`. Turning tracking off must
            not put a branch inside the training loop, so the callback is
            installed either way and streams into something inert.
        """
        return NullLiveRun()


class MlflowLiveRun:
    """An MLflow run left open while the training fills it.

    MLflow's fluent API is used rather than a client keyed on the identifier,
    because experiments run one after another in a single process: there is one
    active run at a time, and it is this one.
    """

    def __init__(self, run_id: str) -> None:
        """Build the handle.

        Args:
            run_id: Identifier MLflow gave the run it just started.
        """
        self._run_id = run_id

    @property
    def run_id(self) -> str:
        """Return the identifier of the open run.

        Returns:
            The MLflow run identifier.
        """
        return self._run_id

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Append one point per metric to the open run.

        Args:
            metrics: The values measured at this point.
            step: Abscissa the values are recorded against.
        """
        import mlflow

        mlflow.log_metrics(dict(metrics), step=step)

    def finish(self, payload: TrackedRun) -> str | None:
        """Write the finished run into the open one and close it.

        The run is renamed here rather than at opening: a run that ended
        ``PARTIAL`` or ``FAILED`` carries that in its name, and the status is
        not known while it trains.

        Args:
            payload: What the record says.

        Returns:
            The MLflow run identifier.
        """
        import mlflow

        try:
            mlflow.set_tag("mlflow.runName", payload.name)
            mlflow.set_tags(payload.tags)
            mlflow.log_params(payload.params)
            if payload.metrics:
                mlflow.log_metrics(payload.metrics)
            for index, epoch in enumerate(payload.epochs):
                mlflow.log_metrics(epoch, step=index)
            for artifact in payload.artifacts:
                mlflow.log_artifact(str(Path(artifact)))
        finally:
            # Closed whatever happened. An artefact upload that fails must not
            # leave the run RUNNING for ever, nor make the next experiment of a
            # sweep start nested inside it.
            mlflow.end_run()
        return self._run_id


class MlflowTracker:
    """Tracker writing to an MLflow store."""

    def __init__(
        self, *, tracking_uri: str | None = None, experiment: str = DEFAULT_EXPERIMENT
    ) -> None:
        """Build the tracker.

        Args:
            tracking_uri: Where the store lives. ``None`` leaves MLflow to its
                own resolution, which ends on a local ``mlruns`` directory when
                nothing else is configured.
            experiment: Name the runs are grouped under.
        """
        self.tracking_uri = tracking_uri
        self.experiment = experiment

    def _connect(self) -> None:
        """Point MLflow at the store this tracker writes to.

        MLflow is imported here rather than at module level: the import costs
        seconds and pulls a dependency the inference image does not carry, and
        a run started with tracking turned off must not pay for either.
        """
        import mlflow

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)

    def open(self, name: str) -> MlflowLiveRun:
        """Start a run and leave it open for the training to fill.

        Args:
            name: Name the run appears under while it is open. The finished
                payload renames it, since the status is not known yet.

        Returns:
            The open run.
        """
        import mlflow

        self._connect()
        if mlflow.active_run() is not None:
            # A previous experiment of the sweep whose closing failed. Starting
            # here would nest this run inside that one, and a nested run does
            # not show up as an experiment of the campaign.
            mlflow.end_run()
        active = mlflow.start_run(run_name=name)
        return MlflowLiveRun(str(active.info.run_id))

    def log(self, payload: TrackedRun) -> str | None:
        """Send one finished run to MLflow, in one transaction.

        This is the mirror of a run that is already over, which is what the
        backfill command of :mod:`src.tracking.log` needs. A run followed while
        it trains is opened by :meth:`open` and closed by
        :meth:`MlflowLiveRun.finish` instead.

        Args:
            payload: The run to record.

        Returns:
            The MLflow run identifier.
        """
        import mlflow

        self._connect()

        with mlflow.start_run(run_name=payload.name) as active:
            mlflow.set_tags(payload.tags)
            mlflow.log_params(payload.params)
            if payload.metrics:
                mlflow.log_metrics(payload.metrics)
            for index, epoch in enumerate(payload.epochs):
                mlflow.log_metrics(epoch, step=index)
            for artifact in payload.artifacts:
                mlflow.log_artifact(str(Path(artifact)))
            return str(active.info.run_id)


def build_tracker(
    *,
    enabled: bool = True,
    tracking_uri: str | None = None,
    experiment: str = DEFAULT_EXPERIMENT,
) -> Tracker:
    """Return the tracker a run should use.

    Args:
        enabled: Whether the run is traced at all.
        tracking_uri: Where the store lives.
        experiment: Name the runs are grouped under.

    Returns:
        An :class:`MlflowTracker`, or a :class:`NullTracker` when tracking is
        turned off.
    """
    if not enabled:
        return NullTracker()
    return MlflowTracker(tracking_uri=tracking_uri, experiment=experiment)


def log_safely(tracker: Tracker, payload: TrackedRun) -> str | None:
    """Send a run to the store, turning any failure into a warning.

    Args:
        tracker: The tracker to use.
        payload: The run to record.

    Returns:
        The identifier the store gave the run, or ``None`` when nothing was
        recorded, whether because tracking is off or because it failed.
    """
    try:
        return tracker.log(payload)
    except Exception as error:
        # Every exception is caught on purpose. The measurement is already on
        # disk; losing its mirror is a degraded run, not a failed one.
        print(
            f"tracking failed for {payload.name}: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return None
