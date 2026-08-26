"""The join between what was declared and what was measured.

An experiment exists twice: as a file under ``configs/experiments/``, and as a
directory under ``reports/results/``. A table built from the directories alone
would be shorter than the plan and look complete; a table built from the files
alone would have no numbers. This module puts the two together and gives every
declared experiment exactly one row, whether or not it ever ran.

That is the whole point of the module. Section 44 asks for ``NOT_RUN`` and
``FAILED`` to be carried explicitly, and the only way an unrun experiment can
carry a status is if something knows it was supposed to exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.experiments.config import DEFAULT_EXPERIMENTS_DIR, ExperimentConfig, discover_experiments
from src.experiments.record import STATUS_NOT_RUN, STATUS_STALE_CONFIG, RunRecord, find_record

#: Where the run directories live, per section 17.
DEFAULT_RESULTS_DIR = Path("reports") / "results"


@dataclass(frozen=True, slots=True)
class ExperimentRow:
    """One declared experiment, with the run it produced if it produced one.

    Attributes:
        config: The declaration, read from ``configs/experiments/``.
        record: What the run wrote, or ``None`` when it never ran.
    """

    config: ExperimentConfig
    record: RunRecord | None

    @property
    def architecture_matches(self) -> bool:
        """Return whether the record came from the architecture now declared.

        Only the ``model`` block is compared. A quick run caps its training and
        its evaluation on purpose, so a record produced by
        :mod:`src.experiments.reproduce` legitimately disagrees with the file
        on those two blocks; the model block is the one a run directory kept
        from an earlier version of the project disagrees on, and the one that
        would turn an obsolete score into a row nobody can tell apart from a
        current one.

        Returns:
            ``True`` when there is nothing to contradict: no record, a record
            without a stored configuration, or a stored model block equal to
            the declared one.
        """
        if self.record is None or not self.record.config:
            return True
        stored = self.record.config.get("model")
        return stored is None or stored == self.config.model.model_dump(mode="json")

    @property
    def name(self) -> str:
        """Return the experiment name.

        Returns:
            The name, identical in the declaration and in the record.
        """
        return self.config.name

    @property
    def status(self) -> str:
        """Return the status carried into every table.

        Returns:
            The status of the record, :data:`STATUS_NOT_RUN` when the
            experiment was declared and never executed, or
            :data:`STATUS_STALE_CONFIG` when the run directory was left behind
            by a different architecture.
        """
        if self.record is None:
            return STATUS_NOT_RUN
        return self.record.status if self.architecture_matches else STATUS_STALE_CONFIG

    @property
    def measured(self) -> bool:
        """Return whether this row holds a reportable score.

        Returns:
            ``True`` only for a complete run of the declared architecture.
            Partial, failed, unrun and obsolete rows appear in the tables with
            their status and no number.
        """
        return self.record is not None and self.architecture_matches and self.record.measured

    def rouge(self, variant: str) -> float | None:
        """Return the corpus F-measure of one ROUGE variant.

        Args:
            variant: Variant name, for example ``rougeL``.

        Returns:
            The mean F-measure, or ``None`` when there is nothing to report.
        """
        return self.record.rouge(variant) if self.measured and self.record is not None else None

    def interval(self, variant: str) -> tuple[float, float] | None:
        """Return the bootstrap interval of one ROUGE variant.

        Args:
            variant: Variant name, for example ``rougeL``.

        Returns:
            The ``(low, high)`` bounds, or ``None`` when there is nothing to
            report.
        """
        return self.record.interval(variant) if self.measured and self.record is not None else None

    def in_study(self, study: str) -> bool:
        """Return whether the experiment belongs to a study.

        Args:
            study: Study name, one of :data:`src.experiments.config.STUDIES`.

        Returns:
            ``True`` when the experiment file lists that study.
        """
        return study in self.config.experiment.studies


def collect(
    experiments_dir: Path | str = DEFAULT_EXPERIMENTS_DIR,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> list[ExperimentRow]:
    """Build one row per declared experiment.

    Args:
        experiments_dir: Directory holding the experiment files.
        results_dir: Directory holding the run directories.

    Returns:
        The rows, sorted by experiment name.

    Raises:
        ValueError: If two experiment files declare the same name.
    """
    return [
        ExperimentRow(config=config, record=find_record(Path(results_dir), config.name))
        for config in discover_experiments(experiments_dir)
    ]


def rows_for_study(rows: list[ExperimentRow], study: str) -> list[ExperimentRow]:
    """Keep the rows belonging to one study.

    Args:
        rows: Every declared experiment.
        study: Study name.

    Returns:
        The subset belonging to that study, in the order it was given.
    """
    return [row for row in rows if row.in_study(study)]
