"""Unit tests of the command that empties the MLflow store.

The store is the only place a measurement still exists once the run
directories have been cleared, so the property that matters most here is the
one that says nothing happened: a command line without ``--yes`` counts and
prints and leaves every run where it was.

The second is that a marked run is not a removed one. ``delete_run`` moves it
out of sight, ``mlflow gc`` removes it, and a collection that failed must be
reported rather than swallowed by an exit code of zero.

The third is that a run is not the only thing a campaign leaves. A logged model
and a registry version are entities of their own, which ``delete_run`` leaves
and ``mlflow gc`` does not collect. A store emptied of its runs alone still
lists every model of every campaign, pointing at runs that no longer exist.
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

    def __init__(
        self,
        runs: dict[str, list[str]],
        models: dict[str, list[str]] | None = None,
        registry: dict[str, list[tuple[str, str]]] | None = None,
    ) -> None:
        # Une entree par experience, et par run son stade de vie : "active"
        # tant que rien ne l'a marque, "deleted" ensuite.
        self.runs = {name: dict.fromkeys(ids, ACTIVE) for name, ids in runs.items()}
        self.stages = dict.fromkeys(runs, ACTIVE)
        # Les modeles enregistres par MLflow 3 a cote des runs, par experience.
        self.models = {name: list(models.get(name, [])) for name in runs} if models else {}
        # Le registre : par nom de modele, ses (version, run d'origine).
        self.registry = dict(registry or {})
        self.deleted_runs: list[str] = []
        self.deleted_experiments: list[str] = []
        self.deleted_models: list[str] = []
        self.deleted_versions: list[tuple[str, str]] = []
        self.deleted_registered: list[str] = []
        self.views: list[Any] = []
        # L'ordre des suppressions : mlflow gc refuse de retirer un run qu'une
        # version vivante designe encore, donc le registre passe en premier.
        self.order: list[str] = []

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
        self.order.append("run")
        for runs in self.runs.values():
            if run_id in runs:
                runs[run_id] = DELETED

    def search_logged_models(self, experiment_ids: list[str]) -> list[SimpleNamespace]:
        name = list(self.runs)[int(experiment_ids[0])]
        return [SimpleNamespace(model_id=model) for model in self.models.get(name, [])]

    def delete_logged_model(self, model_id: str) -> None:
        self.deleted_models.append(model_id)
        self.order.append("model")
        for models in self.models.values():
            if model_id in models:
                models.remove(model_id)

    def search_registered_models(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=name) for name in self.registry]

    def search_model_versions(self, filter_string: str) -> list[SimpleNamespace]:
        name = filter_string.split("'")[1]
        return [
            SimpleNamespace(name=name, version=version, run_id=run_id)
            for version, run_id in self.registry.get(name, [])
        ]

    def delete_model_version(self, name: str, version: str) -> None:
        self.deleted_versions.append((name, version))
        self.order.append("version")
        self.registry[name] = [pair for pair in self.registry[name] if pair[0] != version]

    def delete_registered_model(self, name: str) -> None:
        self.deleted_registered.append(name)
        self.order.append("registered")
        self.registry.pop(name, None)

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


# ---------------------------------------------------------------------------
# What a run leaves beside itself
# ---------------------------------------------------------------------------


@pytest.fixture
def store_with_models(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    """Return a store whose runs left a logged model and a registry version."""
    client = FakeClient(
        {"syntra-summarization": ["a", "b", "c"], DEFAULT_MLFLOW_EXPERIMENT: []},
        models={"syntra-summarization": ["m-1", "m-2"]},
        registry={"syntra-a": [("1", "a")], "syntra-b": [("1", "b"), ("2", "b")]},
    )
    monkeypatch.setattr(purge_module, "build_client", lambda uri: client)
    monkeypatch.setattr(purge_module, "resolve_tracking_uri", lambda: "postgresql://store")
    monkeypatch.setattr(purge_module, "describe_store", lambda uri=None: "tracking postgresql://***")
    monkeypatch.setattr(purge_module, "collect_garbage", lambda uri: (True, "3 runs removed"))
    return client


def test_a_logged_model_goes_with_the_run_that_produced_it(store_with_models: FakeClient) -> None:
    # delete_run ne les emporte pas et mlflow gc ne les collecte pas : sans
    # cette suppression l'interface liste encore les modeles de la campagne
    # precedente, pointant sur des runs qui n'existent plus.
    assert main(["--all", "--yes"]) == 0

    assert store_with_models.deleted_models == ["m-1", "m-2"]


def test_the_registry_versions_of_a_purged_run_go_too(store_with_models: FakeClient) -> None:
    assert main(["--all", "--yes"]) == 0

    assert store_with_models.deleted_versions == [
        ("syntra-a", "1"),
        ("syntra-b", "1"),
        ("syntra-b", "2"),
    ]
    assert store_with_models.deleted_registered == ["syntra-a", "syntra-b"]


def test_a_registered_model_that_keeps_a_version_is_kept(
    store_with_models: FakeClient,
) -> None:
    # Une version produite par un run que la passe ne touche pas garde son
    # modele : le nom reste utilisable, et son run est toujours la.
    store_with_models.registry["syntra-b"] = [("1", "b"), ("2", "hors-campagne")]

    assert main(["--all", "--yes"]) == 0

    assert store_with_models.deleted_versions == [("syntra-a", "1"), ("syntra-b", "1")]
    assert store_with_models.deleted_registered == ["syntra-a"]


def test_the_registry_goes_before_the_runs(store_with_models: FakeClient) -> None:
    # mlflow gc refuse de retirer un run qu'une version vivante designe encore.
    # L'ordre n'est pas cosmetique : il decide si la collecte passe.
    main(["--all", "--yes"])

    order = store_with_models.order
    assert order.index("version") < order.index("run")
    assert order.index("model") < order.index("run")


def test_the_listing_counts_the_models_it_would_take(
    store_with_models: FakeClient, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--all"]) == 0

    output = capsys.readouterr().out
    assert store_with_models.deleted_models == []
    assert store_with_models.deleted_versions == []
    assert "3 run(s), 2 logged model(s)" in output


def test_a_client_without_the_entity_purges_the_runs_anyway(
    store: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # La borne basse de la plage supportee est MLflow 2, qui ne connait ni le
    # modele journalise ni cette facon de lire le registre. Un magasin ecrit
    # par elle n'en porte simplement pas, et la commande doit vider ses runs
    # plutot que de tomber sur une methode absente.
    monkeypatch.delattr(FakeClient, "search_logged_models")
    monkeypatch.delattr(FakeClient, "search_registered_models")

    assert main(["--all", "--yes"]) == 0

    assert store.deleted_runs == ["a", "b", "c"]
    assert store.deleted_models == []


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
