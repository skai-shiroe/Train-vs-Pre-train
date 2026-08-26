"""Send the run records already on disk to the tracking store.

The runner traces every experiment as it finishes it, so this command exists
for the two cases where that did not happen: a run executed while the store was
unreachable, and a store that was created after the runs were. It reads the same
records the ablation tables are built from and pushes them unchanged::

    python -m src.tracking.log --all
    python -m src.tracking.log --experiment scratch_10

**An experiment that never ran produces no run in the store.** It is listed in
the summary with its ``NOT_RUN`` status, because that is what the declared set
says about it, and nothing is sent. A tracking store that held a row for an
experiment nobody executed would put a fabricated run in the one place a reader
goes to count how much of the plan is done.

**Re-sending a record creates a new run in the store.** Nothing is overwritten
and nothing is deduplicated: the store keeps a history, and two runs carrying
the same record and the same commit are two pushes of one measurement, which is
readable. Silently updating an earlier run would rewrite what the store said
yesterday.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from src.experiments.config import DEFAULT_EXPERIMENTS_DIR
from src.experiments.record import (
    STATUS_NOT_RUN,
    STATUS_PARTIAL,
    STATUS_STALE_CONFIG,
    run_directory,
)
from src.experiments.registry import DEFAULT_RESULTS_DIR, ExperimentRow, collect
from src.tracking.client import DEFAULT_EXPERIMENT, Tracker, build_tracker, log_safely
from src.tracking.payload import build_payload


def push(row: ExperimentRow, tracker: Tracker, results_dir: Path) -> str | None:
    """Send one experiment row to the store.

    Args:
        row: A declared experiment, with the record it produced if it produced
            one.
        tracker: The tracker to send to.
        results_dir: Directory the run directories live under.

    Returns:
        The identifier the store gave the run, or ``None`` when the experiment
        never ran or the push failed.
    """
    if row.record is None:
        return None
    if row.status == STATUS_STALE_CONFIG:
        return None

    directory = run_directory(results_dir, row.name, partial=row.status == STATUS_PARTIAL)
    return log_safely(tracker, build_payload(row.record, directory=directory))


def build_argument_parser() -> argparse.ArgumentParser:
    """Describe the command line of ``python -m src.tracking.log``.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.tracking.log",
        description=(
            "Send the run records under reports/results to MLflow. Nothing is "
            "computed: an experiment that never ran is reported and skipped."
        ),
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--experiment", help="Name of a single experiment to send.")
    selection.add_argument(
        "--all", action="store_true", help="Send every declared experiment that ran."
    )

    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=DEFAULT_EXPERIMENTS_DIR,
        help="Directory holding the experiment files.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory holding the run directories.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI. Left to MLflow's own resolution when absent.",
    )
    parser.add_argument(
        "--tracking-experiment",
        default=DEFAULT_EXPERIMENT,
        help="MLflow experiment the runs are grouped under.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Send the records from the command line.

    Args:
        argv: Arguments to parse. Defaults to the process arguments.

    Returns:
        Process exit code. One when a selected experiment could not be sent, so
        an unreachable store is visible rather than silent.
    """
    args = build_argument_parser().parse_args(argv)

    rows = collect(args.experiments_dir, args.results)
    if args.experiment is not None:
        rows = [row for row in rows if row.name == args.experiment]
        if not rows:
            print(f"No experiment named {args.experiment!r} is declared.", file=sys.stderr)
            return 1

    tracker = build_tracker(tracking_uri=args.tracking_uri, experiment=args.tracking_experiment)

    failures = 0
    for row in rows:
        if row.record is None:
            print(f"{row.name:<24} {STATUS_NOT_RUN:<8} nothing to send")
            continue
        if row.status == STATUS_STALE_CONFIG:
            print(f"{row.name:<24} {STATUS_STALE_CONFIG:<8} obsolete record skipped")
            continue

        run_id = push(row, tracker, args.results)
        if run_id is None:
            failures += 1
            print(f"{row.name:<24} {row.status:<8} not sent")
            continue
        print(f"{row.name:<24} {row.status:<8} {run_id}")

    if failures:
        print(f"\n{failures} record(s) could not be sent.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
