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
from src.tracking.purge import DEFAULT_MLFLOW_EXPERIMENT, collect_garbage, main

pytestmark = pytest.mark.unit

#: Les deux stades de vie de MLflow, ecrits comme la base les porte.
ACTIVE = "active"
DELETED = "deleted"


class FakeClient:
    """Client over a store held in a dictionary."""

    def __init__(self, runs: dict[str, list[str]]) -> None:
        # Une entree par experience, et par run son stade de vie : "active"
        # tant que rien ne l'a marque, "deleted" ensuite.
        self.runs = {name: {run_id: ACTIVE for run_id in ids} for name, ids in runs.items()}
        self.stages = dict.fromkeys(runs, ACTIVE)
        self.deleted_runs: list[str] = []
        self.deleted_experiments: list[str] = []
        self.views: list[Any] = []

    def search_experiments(self, view_type: Any = None) -> list[SimpleNamespace]:
        self.views.append(view_type)
        return [
            SimpleNamespace(experiment_id=str(index), name=name, lifecycle_stage=self.stages[name])
            for index, name in enumerate(self.runs)
        ]

    def search_runs(
        self, experiment_ids: list[str], run_view_type: Any = None, max_results: int = 0
    ) -> list[SimpleNamespace]:
        self.views.append(run_view_type)
        name = list(self.runs)[int(experiment_ids[0])]
        return [
            SimpleNamespace(info=SimpleNamespace(run_id=run_id, lifecycle_stage=stage))
            for run_id, stage in self.runs[name].items()
        ]

    def delete_run(self, run_id: str) -> None:
        self.deleted_runs.append(run_id)
        for runs in self.runs.values():
            if run_id in runs:
                runs[run_id] = DELETED

    def delete_experiment(self, experiment_id: str) -> None:
        self.deleted_experiments.append(experiment_id)
        self.stages[list(self.runs)[int(experiment_id)]] = DELETED


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


def test_a_second_pass_marks_nothing_and_raises_nothing(
    store: FakeClient, capsys: pytest.CaptureFixture[str]
) -> None:
    # Ce que demande une collecte qui a echoue : relancer. MLflow cherche une
    # experience parmi les actives avant de la supprimer et leve quand elle
    # n'y est plus, donc la seconde passe tomberait sur le travail de la
    # premiere.
    main(["--all", "--yes"])
    store.deleted_runs.clear()
    store.deleted_experiments.clear()
    capsys.readouterr()

    assert main(["--all", "--yes"]) == 0

    assert store.deleted_runs == []
    assert store.deleted_experiments == []
    assert "3 already were" in capsys.readouterr().out


def test_the_collection_is_told_where_the_store_is(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both names are required. mlflow gc refuses to start without a tracking
    # URI, and its own backend store defaults to a local directory: told only
    # half of it, the command reports a clean sweep of a store nobody traced
    # into, and the campaign it was meant to clear is still there.
    seen: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        seen.append(argv)
        return SimpleNamespace(returncode=0, stdout="2 runs removed", stderr="")

    monkeypatch.setattr(purge_module.subprocess, "run", fake_run)

    collected, said = collect_garbage("postgresql://store")

    assert collected
    assert said == "2 runs removed"
    assert seen[0][seen[0].index("--backend-store-uri") + 1] == "postgresql://store"
    assert seen[0][seen[0].index("--tracking-uri") + 1] == "postgresql://store"


def test_a_store_that_is_not_configured_collects_nothing() -> None:
    collected, said = collect_garbage(None)

    assert not collected
    assert "no store is configured" in said


def test_the_two_selections_are_exclusive() -> None:
    with pytest.raises(SystemExit):
        main(["--all", "--experiment", "syntra-summarization"])


def test_a_selection_is_required() -> None:
    with pytest.raises(SystemExit):
        main([])
