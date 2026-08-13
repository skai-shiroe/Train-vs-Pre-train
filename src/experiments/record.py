"""What one experiment leaves behind, and how it is read back.

The runner writes this record, the registry reads it, and the ablation tables
are built from nothing else. Keeping the format in one module is what stops the
writer and the reader from drifting apart, and it is where section 44 is
enforced in code rather than in a convention.

**A run has a status, never a blank.** Section 44 asks for ``NOT_RUN`` on an
experiment that was never executed and ``FAILED`` on one that crashed. Both are
statuses a table can carry, so a missing measurement stays a visible row instead
of an absent one. An absent row is the dangerous shape: a table of four rows
where six experiments were declared looks complete.

**A partial run cannot occupy the complete slot.** ``--limit`` and a capped
training budget both exist to exercise the chain quickly. Such a run writes to a
directory suffixed with ``_partial`` and is reported as ``PARTIAL``, so it can
never overwrite a full measurement and its score is never read into a
comparison table.

**The settings that produced the score travel with it.** The decoding
configuration and the measurement configuration are stored inside the record.
:mod:`src.experiments.ablation` compares them across the runs of a study and
refuses to build a table out of runs that were not measured the same way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: File the runner writes inside a run directory.
RUN_RECORD_FILE = "run.json"

#: Training curves written next to the record, for the figures of section 18.
HISTORY_FILE = "history.json"

#: Suffix of a run directory that does not hold a reportable measurement.
PARTIAL_SUFFIX = "_partial"

#: A complete run, measured on the whole split.
STATUS_OK = "OK"

#: A run that covered part of the split, or trained under a capped budget.
STATUS_PARTIAL = "PARTIAL"

#: A run that raised before it could be measured.
STATUS_FAILED = "FAILED"

#: An experiment that is declared but was never executed.
STATUS_NOT_RUN = "NOT_RUN"


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Everything one execution of one experiment produced.

    Attributes:
        experiment: Name of the experiment, matching its configuration file.
        status: One of :data:`STATUS_OK`, :data:`STATUS_PARTIAL` or
            :data:`STATUS_FAILED`. :data:`STATUS_NOT_RUN` never appears in a
            file: it describes the absence of one.
        config: The experiment configuration the run was started from.
        dataset: Corpus identification: version, proportion, example counts.
        model: Description returned by the summariser that was measured.
        hardware: What the run executed on, per section 19.
        provenance: The commit the run was started from, per section 19.
            Recorded here rather than recomputed later: a version published or
            traced next week would otherwise carry the commit of that day.
        training: What the training loop produced, or ``None`` for a zero shot
            run.
        evaluation: What the evaluation produced, or ``None`` when the run
            failed before it.
        error: Type and message of the exception that ended a failed run.
        duration_seconds: Wall clock time of the whole run.
    """

    experiment: str
    status: str
    config: dict[str, Any]
    dataset: dict[str, Any]
    model: dict[str, str]
    hardware: dict[str, str]
    provenance: dict[str, str] = field(default_factory=dict)
    training: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    error: str | None = None
    duration_seconds: float = 0.0

    @property
    def measured(self) -> bool:
        """Return whether this record holds a reportable score.

        Returns:
            ``True`` only for a complete run. A partial or failed run has no
            number a comparison may use.
        """
        return self.status == STATUS_OK and self.evaluation is not None

    @property
    def report(self) -> dict[str, Any]:
        """Return the summarisation report of the run.

        Returns:
            The ``report`` block of the evaluation, or an empty mapping when
            the run produced none.
        """
        if self.evaluation is None:
            return {}
        report: Any = self.evaluation.get("report", {})
        return report if isinstance(report, dict) else {}

    def rouge(self, variant: str) -> float | None:
        """Return the corpus F-measure of one ROUGE variant.

        Args:
            variant: Variant name, for example ``rougeL``.

        Returns:
            The mean F-measure, or ``None`` when the run holds no reportable
            score. Returning ``None`` rather than zero matters: a zero is a
            measurement, and a model that was never run did not score zero.
        """
        if not self.measured:
            return None
        scores: Any = self.report.get("rouge", {}).get("scores", {})
        entry = scores.get(variant) if isinstance(scores, dict) else None
        return float(entry["fmeasure"]) if isinstance(entry, dict) else None

    def interval(self, variant: str) -> tuple[float, float] | None:
        """Return the bootstrap interval of one ROUGE variant.

        Args:
            variant: Variant name, for example ``rougeL``.

        Returns:
            The ``(low, high)`` bounds, or ``None`` when the run holds no
            reportable score or the interval was disabled.
        """
        if not self.measured:
            return None
        intervals: Any = self.report.get("rouge", {}).get("intervals", {})
        entry = intervals.get(variant) if isinstance(intervals, dict) else None
        if not isinstance(entry, dict):
            return None
        return float(entry["low"]), float(entry["high"])

    @property
    def measurement_key(self) -> tuple[str, str, str]:
        """Return what makes two runs comparable.

        Two scores may be put in the same table only when they were produced on
        the same split, with the same decoding and the same ROUGE settings.
        Rendering the three as sorted JSON gives a value that can simply be
        compared for equality.

        Returns:
            A triple ``(split, decoding, measurement)``.
        """
        evaluation = self.evaluation or {}
        return (
            str(evaluation.get("split", "")),
            json.dumps(evaluation.get("generation", {}), sort_keys=True),
            json.dumps(evaluation.get("rouge_config", {}), sort_keys=True),
        )

    def to_dict(self) -> dict[str, Any]:
        """Render the record as a JSON serialisable mapping.

        Returns:
            The mapping written to ``run.json``.
        """
        return {
            "experiment": self.experiment,
            "status": self.status,
            "config": self.config,
            "dataset": self.dataset,
            "model": dict(self.model),
            "hardware": dict(self.hardware),
            "provenance": dict(self.provenance),
            "training": self.training,
            "evaluation": self.evaluation,
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 3),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunRecord:
        """Rebuild a record from its serialised form.

        Args:
            payload: Mapping read from ``run.json``.

        Returns:
            The reconstructed record.

        Raises:
            KeyError: If the mandatory fields are missing. A truncated record
                is a defect, not a run without a status.
        """
        return cls(
            experiment=str(payload["experiment"]),
            status=str(payload["status"]),
            config=dict(payload.get("config", {})),
            dataset=dict(payload.get("dataset", {})),
            model={str(key): str(value) for key, value in payload.get("model", {}).items()},
            hardware={str(key): str(value) for key, value in payload.get("hardware", {}).items()},
            provenance={
                str(key): str(value) for key, value in payload.get("provenance", {}).items()
            },
            training=payload.get("training"),
            evaluation=payload.get("evaluation"),
            error=payload.get("error"),
            duration_seconds=float(payload.get("duration_seconds", 0.0)),
        )


def run_directory(results_dir: Path, name: str, *, partial: bool = False) -> Path:
    """Return the directory one run writes to.

    Args:
        results_dir: Directory the run directories are created under.
        name: Experiment name.
        partial: Whether the run covered a subset or trained under a cap.

    Returns:
        The run directory, suffixed when the run is not reportable.
    """
    return Path(results_dir) / (f"{name}{PARTIAL_SUFFIX}" if partial else name)


def write_record(record: RunRecord, directory: Path) -> Path:
    """Write a run record inside a run directory.

    Args:
        record: The record to persist.
        directory: Destination directory. Created when it does not exist.

    Returns:
        The path written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RUN_RECORD_FILE
    path.write_text(
        json.dumps(record.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def read_record(directory: Path) -> RunRecord | None:
    """Read the record of one run directory.

    Args:
        directory: Directory possibly holding a run.

    Returns:
        The record, or ``None`` when the directory holds none.
    """
    path = Path(directory) / RUN_RECORD_FILE
    if not path.is_file():
        return None
    return RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))


def find_record(results_dir: Path, name: str) -> RunRecord | None:
    """Find the record of one experiment, complete run first.

    A complete run wins over a partial one of the same name. The two live in
    different directories, so a quick check run never hides the measurement it
    was a rehearsal for, and never stands in for one either.

    Args:
        results_dir: Directory the run directories live under.
        name: Experiment name.

    Returns:
        The record, or ``None`` when the experiment was never run.
    """
    complete = read_record(run_directory(results_dir, name))
    if complete is not None:
        return complete
    return read_record(run_directory(results_dir, name, partial=True))
