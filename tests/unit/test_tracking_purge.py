"""Unit tests of the command that empties the MLflow store.

The store is the only place a measurement still exists once the run
directories have been cleared, so the property that matters most here is the
one that says nothing happened: a command line without ``--yes`` counts and
prints and leaves every run where it was.

The second is that a marked run is not a removed one. ``delete_run`` moves it
out of sight, ``mlflow gc`` removes it, and a collection that failed must be
reported rather than swallowed by an exit code of zero.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.tracking import purge as purge_module
from src.tracking.purge import DEFAULT_MLFLOW_EXPERIMENT, main

pytestmark = pytest.mark.unit


class FakeClient:
    """Client over a store held in a dictionary."""

    def __init__(self, runs: dict[str, list[str]]) -> None:
        self.runs = {name: list(ids) for name, ids in runs.items()}
        self.deleted_runs: list[str] = []
        self.deleted_experiments: list[str] = []
        self.views: list[Any] = []

    def search_experiments(self, view_type: Any = None) -> list[SimpleNamespace]:
        self.views.append(view_type)
        return [
            SimpleNamespace(experiment_id=str(index), name=name)
            for index, name in enumerate(self.runs)
        ]

    def search_runs(
        self, experiment_ids: list[str], run_view_type: Any = None, max_results: int = 0
    ) -> list[SimpleNamespace]:
        self.views.append(run_view_type)
        name = list(self.runs)[int(experiment_ids[0])]
        return [SimpleNamespace(info=SimpleNamespace(run_id=run_id)) for run_id in self.runs[name]]

    def delete_run(self, run_id: str) -> None:
        self.deleted_runs.append(run_id)

    def delete_experiment(self, experiment_id: str) -> None:
        self.deleted_experiments.append(experiment_id)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    """Return a store holding two experiments, wired into the command."""
    client = FakeClient({"syntra-summarization": ["a", "b", "c"], DEFAULT_MLFLOW_EXPERIMENT: []})
    monkeypatch.setattr(purge_module, "build_client", lambda uri: client)
    monkeypatch.setattr(purge_module, "resolve_tracking_uri", lambda: "postgresql://store")
    monkeypatch.setattr(purge_module, "describe_store", lambda uri=None: "tracking postgresql://***")
    monkeypatch.setattr(purge_module, "collect_garbage", lambda uri: (True, "3 runs removed"))
    return client


def test_a_command_line_without_yes_deletes_nothing(
    store: FakeClient, capsys: pytest.CaptureFixture[str]
) -> None:
    # The store is what the report argues from once the run directories are
    # gone. A command that empties it on a typo is not one to keep.
    assert main(["--all"]) == 0

    assert store.deleted_runs == []
    assert store.deleted_experiments == []
    assert "Nothing deleted: add --yes." in capsys.readouterr().out


def test_yes_marks_every_run_of_every_experiment(store: FakeClient) -> None:
    assert main(["--all", "--yes"]) == 0

    assert store.deleted_runs == ["a", "b", "c"]


def test_the_default_experiment_is_emptied_but_never_deleted(store: FakeClient) -> None:
    # MLflow refuses to delete it and recreates it at the next start, so
    # deleting it would be a failure reported on every single run.
    main(["--all", "--yes"])

    assert store.deleted_experiments == ["0"]


def test_one_experiment_can_be_emptied_alone(store: FakeClient) -> None:
    store.runs[DEFAULT_MLFLOW_EXPERIMENT] = ["d"]

    assert main(["--experiment", "syntra-summarization", "--yes"]) == 0

    assert store.deleted_runs == ["a", "b", "c"]
    assert store.deleted_experiments == ["0"]


def test_an_experiment_the_store_does_not_hold_is_an_error(
    store: FakeClient, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--experiment", "absent"]) == 1

    assert store.deleted_runs == []
    assert "No experiment named 'absent'" in capsys.readouterr().err


def test_a_failed_collection_says_the_runs_are_still_there(
    store: FakeClient, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A marked run is not a removed one: the rows and the artefacts are still
    # in place, and only the interface stopped showing them.
    monkeypatch.setattr(purge_module, "collect_garbage", lambda uri: (False, "connection refused"))

    assert main(["--all", "--yes"]) == 1

    assert store.deleted_runs == ["a", "b", "c"]
    assert "still there" in capsys.readouterr().err


def test_a_run_left_running_by_a_killed_campaign_is_taken(store: FakeClient) -> None:
    # Both lifecycle stages are read: a run marked deleted still holds rows,
    # and one left RUNNING is exactly what this command exists to clear.
    main(["--all", "--yes"])

    assert all(view is not None for view in store.views)


def test_the_two_selections_are_exclusive() -> None:
    with pytest.raises(SystemExit):
        main(["--all", "--experiment", "syntra-summarization"])


def test_a_selection_is_required() -> None:
    with pytest.raises(SystemExit):
        main([])
