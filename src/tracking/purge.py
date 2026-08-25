"""Empty the MLflow store, so a campaign can start against nothing.

A campaign that is replayed from the beginning leaves the previous one in the
store beside it. Nothing breaks: the runs carry their own names and dates. But
a listing then holds two answers per experiment, and the interface offers no
way to tell the abandoned attempt from the one the report argues from::

    python -m src.tracking.purge --all              # what would go, nothing goes
    python -m src.tracking.purge --all --yes        # it goes

**A run of this command deletes nothing.** The store is the only place a
measurement exists once ``reports/results`` has been cleared, and a command
that empties it on a typo is not one to keep. Without ``--yes`` the runs are
counted and printed and the store is left as it was.

**Deleting a run in MLflow does not remove it.** ``delete_run`` moves it to the
``deleted`` lifecycle stage: the interface stops showing it, the rows stay in
the database and the artefacts stay on disk. ``mlflow gc`` is what removes
both, and it only ever touches what was already marked. This module runs the
two steps in that order, and reports the second failing without pretending the
first did not happen.

**The Default experiment is emptied, not deleted.** MLflow refuses to delete
it and recreates it at the next start, so deleting it is a failure this command
would report on every run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence

from mlflow.entities import Experiment, LifecycleStage, Run, ViewType
from mlflow.tracking import MlflowClient

from src.tracking.client import DEFAULT_EXPERIMENT
from src.tracking.store import describe_store, resolve_tracking_uri

#: Experiment MLflow creates on its own and refuses to delete.
DEFAULT_MLFLOW_EXPERIMENT = "Default"

#: Runs read at once. A campaign writes nine, and the store of this repository
#: holds a few dozen; one page is the whole store rather than the first of many.
PAGE_SIZE = 1000


def build_client(tracking_uri: str | None) -> MlflowClient:
    """Open a client on the store to empty.

    Args:
        tracking_uri: URI of the store. ``None`` leaves MLflow to its own
            resolution, which is what every other command of the repository
            does.

    Returns:
        The client.
    """
    return MlflowClient(tracking_uri=tracking_uri)


def selected_experiments(client: MlflowClient, name: str | None) -> list[Experiment]:
    """Return the experiments the command was pointed at.

    Args:
        client: Client on the store.
        name: Name of a single experiment, or ``None`` for every one of them.

    Returns:
        The experiments, empty when a named one does not exist. Both lifecycle
        stages are searched: an experiment already marked deleted still holds
        its runs, and leaving them would make ``--all`` a command that empties
        most of the store.
    """
    experiments = client.search_experiments(view_type=ViewType.ALL)
    if name is None:
        return list(experiments)
    return [experiment for experiment in experiments if experiment.name == name]


def runs_of(client: MlflowClient, experiment: Experiment) -> list[Run]:
    """Return every run of an experiment.

    Args:
        client: Client on the store.
        experiment: The experiment to read.

    Returns:
        The runs, whatever their status and whatever their lifecycle stage. A
        run left ``RUNNING`` by a killed campaign is what this command exists
        to clear, and one already marked deleted is still in the database until
        the collection removes it.
    """
    return client.search_runs(
        [experiment.experiment_id], run_view_type=ViewType.ALL, max_results=PAGE_SIZE
    )


def active(entity: Experiment | Run) -> bool:
    """Return whether an experiment or a run is still in the active stage.

    Args:
        entity: An experiment, which carries its stage, or a run, which carries
            it under ``info``.

    Returns:
        ``True`` when it has not been marked deleted yet.
    """
    return getattr(entity, "info", entity).lifecycle_stage == LifecycleStage.ACTIVE


def purge_experiment(client: MlflowClient, experiment: Experiment) -> tuple[int, int]:
    """Mark every run of an experiment deleted, then the experiment itself.

    What is already marked is left alone. MLflow looks an experiment up among
    the active ones before deleting it and raises when it is not there, so a
    second pass over a store whose collection failed would crash on the work
    the first pass did. That second pass is exactly what a failed collection
    asks for.

    Args:
        client: Client on the store.
        experiment: The experiment to empty.

    Returns:
        How many runs this pass marked, and how many the experiment holds. The
        experiment is marked too, unless it is the one MLflow recreates on its
        own, or it was already marked.
    """
    runs = runs_of(client, experiment)
    marked = 0
    for run in runs:
        if not active(run):
            continue
        client.delete_run(run.info.run_id)
        marked += 1

    if experiment.name != DEFAULT_MLFLOW_EXPERIMENT and active(experiment):
        client.delete_experiment(experiment.experiment_id)
    return marked, len(runs)


def collect_garbage(tracking_uri: str | None) -> tuple[bool, str]:
    """Remove for good what was marked deleted, artefacts included.

    ``mlflow gc`` is a command rather than a client call, so it runs as one.
    The interpreter is this one, which is what keeps the subprocess inside the
    virtual environment of the repository.

    The URI is passed twice, under the two names the command asks for.
    ``--backend-store-uri`` says where the rows are, and its own default is a
    local directory, which would report a clean sweep of a store nobody traced
    into. ``--tracking-uri`` is what the command resolves the artefact location
    of every removed run through, and it refuses to start without it. Both hold
    the same value here: this repository traces straight into the database
    rather than through a server.

    Args:
        tracking_uri: URI of the store.

    Returns:
        Whether it succeeded, and what it said. A failure leaves the runs
        marked deleted and invisible in the interface, which is a state to
        report rather than one to hide.
    """
    if tracking_uri is None:
        return False, "no store is configured, so there is nothing to collect."

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlflow",
            "gc",
            "--backend-store-uri",
            tracking_uri,
            "--tracking-uri",
            tracking_uri,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout or completed.stderr).strip()
    return completed.returncode == 0, output


def build_argument_parser() -> argparse.ArgumentParser:
    """Describe the command line of ``python -m src.tracking.purge``.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.tracking.purge",
        description=(
            "Empty the MLflow store of its runs. Prints what would go and "
            "deletes nothing unless --yes is given."
        ),
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--experiment",
        help=f"Name of a single MLflow experiment to empty, for example {DEFAULT_EXPERIMENT}.",
    )
    selection.add_argument(
        "--all", action="store_true", help="Empty every experiment of the store."
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete. Without it the command is a listing.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI. Read from the environment and .env when absent.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Empty the store from the command line.

    Args:
        argv: Arguments to parse. Defaults to the process arguments.

    Returns:
        Process exit code. One when the store holds nothing the command was
        pointed at, or when the collection failed and left the runs marked but
        present.
    """
    args = build_argument_parser().parse_args(argv)
    tracking_uri = args.tracking_uri or resolve_tracking_uri()

    print(f"store            {describe_store(args.tracking_uri)}")
    client = build_client(tracking_uri)
    experiments = selected_experiments(client, None if args.all else args.experiment)

    if not experiments:
        print(f"No experiment named {args.experiment!r} in this store.", file=sys.stderr)
        return 1

    if not args.yes:
        total = 0
        for experiment in experiments:
            count = len(runs_of(client, experiment))
            total += count
            print(f"{experiment.name:<26}{count} run(s)")
        print(f"\n{total} run(s) in {len(experiments)} experiment(s). Nothing deleted: add --yes.")
        return 0

    marked_total = 0
    held_total = 0
    for experiment in experiments:
        marked, held = purge_experiment(client, experiment)
        marked_total += marked
        held_total += held
        kept = " (kept, MLflow recreates it)" if experiment.name == DEFAULT_MLFLOW_EXPERIMENT else ""
        already = f", {held - marked} already were" if held > marked else ""
        print(f"{experiment.name:<26}{marked} run(s) marked deleted{already}{kept}")

    collected, said = collect_garbage(tracking_uri)
    print(f"\n{held_total} run(s) in the deleted stage. mlflow gc: {said or 'nothing to say'}")
    if not collected:
        print(
            "The collection failed. The runs are marked deleted and the interface no "
            "longer shows them, but their rows and their artefacts are still there.",
            file=sys.stderr,
        )
        return 1

    print(f"{held_total} run(s) removed, artefacts included.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
