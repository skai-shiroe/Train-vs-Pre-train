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

from mlflow.entities import Experiment, ViewType
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


def runs_of(client: MlflowClient, experiment: Experiment) -> list[str]:
    """Return the identifiers of every run of an experiment.

    Args:
        client: Client on the store.
        experiment: The experiment to read.

    Returns:
        One identifier per run, whatever its status: a run left ``RUNNING`` by
        a killed campaign is exactly what this command exists to clear.
    """
    runs = client.search_runs(
        [experiment.experiment_id], run_view_type=ViewType.ALL, max_results=PAGE_SIZE
    )
    return [run.info.run_id for run in runs]


def purge_experiment(client: MlflowClient, experiment: Experiment) -> int:
    """Mark every run of an experiment deleted, then the experiment itself.

    Args:
        client: Client on the store.
        experiment: The experiment to empty.

    Returns:
        The number of runs marked. The experiment is marked too, unless it is
        the one MLflow recreates on its own.
    """
    run_ids = runs_of(client, experiment)
    for run_id in run_ids:
        client.delete_run(run_id)

    if experiment.name != DEFAULT_MLFLOW_EXPERIMENT:
        client.delete_experiment(experiment.experiment_id)
    return len(run_ids)


def collect_garbage(tracking_uri: str | None) -> tuple[bool, str]:
    """Remove for good what was marked deleted, artefacts included.

    ``mlflow gc`` is a command rather than a client call, so it runs as one.
    The interpreter is this one, which is what keeps the subprocess inside the
    virtual environment of the repository.

    Args:
        tracking_uri: URI of the store, required by the command itself: its own
            default is a local directory, which would report a clean sweep of a
            store nobody traced into.

    Returns:
        Whether it succeeded, and what it said. A failure leaves the runs
        marked deleted and invisible in the interface, which is a state to
        report rather than one to hide.
    """
    if tracking_uri is None:
        return False, "no store is configured, so there is nothing to collect."

    completed = subprocess.run(
        [sys.executable, "-m", "mlflow", "gc", "--backend-store-uri", tracking_uri],
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

    total = 0
    for experiment in experiments:
        count = purge_experiment(client, experiment)
        total += count
        kept = " (kept, MLflow recreates it)" if experiment.name == DEFAULT_MLFLOW_EXPERIMENT else ""
        print(f"{experiment.name:<26}{count} run(s) marked deleted{kept}")

    collected, said = collect_garbage(tracking_uri)
    print(f"\n{total} run(s) marked. mlflow gc: {said or 'nothing to say'}")
    if not collected:
        print(
            "The collection failed. The runs are marked deleted and the interface no "
            "longer shows them, but their rows and their artefacts are still there.",
            file=sys.stderr,
        )
        return 1

    print(f"{total} run(s) removed, artefacts included.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
