"""What section 19 asks to be traced, read out of a run record.

Section 19 lists the fields every experiment must carry into the tracking
store. All of them are already inside ``run.json``, which is why this module
takes a :class:`src.experiments.record.RunRecord` and produces parameters,
metrics, tags and artefact paths, and touches MLflow nowhere. The mapping is:

===================  =========================================================
Section 19 field     Source
===================  =========================================================
experiment_id        ``experiment``, the name of the configuration file
git_commit           :mod:`src.tracking.provenance`, a tag
seed                 ``config.experiment.seed``
dataset_version      ``dataset.version``, the checksum of the frozen corpus
dataset_percentage   ``dataset.percentage``, absent for the zero shot run
model                ``model``, as the summariser described itself
hyperparameters      ``config.training`` and ``config.evaluation``, flattened
training_duration    ``training.duration_seconds``
BLEU                 out of scope, see below
ROUGE                ``evaluation.report.rouge``
checkpoint           ``training.best_checkpoint``, a parameter, not an upload
hardware             ``hardware``, tags
===================  =========================================================

**BLEU is absent because the project has no BLEU.** Section 2.1 puts translation
and BLEU out of scope and locks the metrics to ROUGE-1, ROUGE-2 and ROUGE-L. A
BLEU column filled with zeros would be a fabricated measurement under section
44, and one filled with a ROUGE would be worse.

**Nothing is computed here.** Every number is copied from the record, including
the commit: the runner reads it once, when the run happens, and stores it. A
tracking layer that recomputed anything would be able to disagree with the file
the tables are built from, and there would be no way to tell which one is the
result. A record pushed a week later would also carry the commit of that day
rather than the one that produced the score.

**Only a complete run logs a quality metric.** A partial run and a failed one
are traced, with their status, their configuration and their duration, and with
no ROUGE at all. A tracking store is sorted by score; a rehearsal over fifty
documents sitting in that ranking is exactly the presented but unmeasured number
section 44 forbids.

**The checkpoint is traced by path, not uploaded.** The weights already live
under ``runs/<experiment>/``, written by the checkpoint manager. Copying several
hundred megabytes into the tracking store for every run would produce a second
copy of the same file, in the place nothing loads from.

**The configuration is parameters, the environment is tags.** A parameter is
what the experiment declared and what a rerun would have to repeat. The
hardware, the commit and the state of the working tree describe where it ran,
which is metadata: a run that reports the same parameters on another GPU is the
same experiment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.evaluation.evaluator import METRICS_FILE, QUALITATIVE_FILE
from src.experiments.record import HISTORY_FILE, RUN_RECORD_FILE, STATUS_OK, RunRecord
from src.metrics.rouge import ROUGE_VARIANTS

#: Files of a run directory that are copied into the tracking store. All four
#: are small text documents. ``predictions.jsonl`` is deliberately absent: it
#: holds one line per scored document, it is already published as a pipeline
#: artefact, and nothing reads it from the tracking store.
TRACKED_ARTIFACTS: tuple[str, ...] = (
    RUN_RECORD_FILE,
    METRICS_FILE,
    HISTORY_FILE,
    QUALITATIVE_FILE,
)

#: Prefix of every tag this project sets. MLflow reserves the ``mlflow`` prefix
#: for its own.
TAG_PREFIX = "syntra"


@dataclass(frozen=True, slots=True)
class TrackedRun:
    """One run, in the shape a tracking store accepts.

    Attributes:
        name: Name the run appears under. A run that is not a complete
            measurement carries its status in that name.
        params: What the experiment declared, flattened to strings.
        metrics: What it produced. Empty for a run that failed before it
            measured anything.
        tags: Where and from which code it ran.
        epochs: One mapping per epoch, logged against the epoch index so the
            training curves are readable in the tracking user interface.
        artifacts: Files to copy into the store, all of which exist.
    """

    name: str
    params: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    epochs: tuple[dict[str, float], ...] = ()
    artifacts: tuple[Path, ...] = ()


def flatten(values: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten a nested mapping into dotted parameter names.

    Args:
        values: The mapping to flatten.
        prefix: Prefix prepended to every key, ending with a dot when not empty.

    Returns:
        A flat mapping of strings. A ``None`` value is dropped rather than
        rendered: an absent parameter shows as an empty cell, whereas the string
        ``None`` reads as a value that was chosen.
    """
    flat: dict[str, str] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, prefix=f"{name}."))
        elif value is None:
            continue
        elif isinstance(value, list | tuple):
            flat[name] = " ".join(str(item) for item in value)
        else:
            flat[name] = str(value)
    return flat


def build_params(record: RunRecord) -> dict[str, str]:
    """Return what the experiment declared.

    Args:
        record: The run record.

    Returns:
        The flattened parameters. The corpus proportion of a zero shot run is
        absent rather than zero: the model was not trained on nothing, it was
        not trained.
    """
    params: dict[str, str] = {"experiment": record.experiment}
    params.update(flatten(record.config))
    params.update(flatten(record.dataset, prefix="dataset."))
    params.update(flatten(record.model, prefix="model."))

    checkpoint = (record.training or {}).get("best_checkpoint")
    if checkpoint is not None:
        params["checkpoint"] = str(checkpoint)
    return params


def build_tags(record: RunRecord) -> dict[str, str]:
    """Return where the run happened and which code produced it.

    Args:
        record: The run record.

    Returns:
        The tags, prefixed with :data:`TAG_PREFIX` so they cannot collide with
        the ones MLflow sets itself.
    """
    model: Any = record.config.get("model", {})
    tags: dict[str, str] = {
        "status": record.status,
        "measured": str(record.measured).lower(),
        "model_type": str(model.get("type", "")) if isinstance(model, Mapping) else "",
        "model_mode": str(model.get("mode", "")) if isinstance(model, Mapping) else "",
    }
    tags.update(record.provenance)
    tags.update(flatten(record.hardware, prefix="hardware."))
    if record.error is not None:
        tags["error"] = record.error
    return {f"{TAG_PREFIX}.{key}": value for key, value in tags.items() if value}


def build_metrics(record: RunRecord) -> dict[str, float]:
    """Return what the run produced.

    Three families are collected under three different conditions. The wall
    clock duration is always known. The training metrics exist as soon as a
    training loop ran, whether or not its result is reportable. The quality
    metrics exist only for a complete run, which is the rule that keeps a
    rehearsal out of a ranking sorted by score.

    Args:
        record: The run record.

    Returns:
        The metrics, all finite floats.
    """
    metrics: dict[str, float] = {"run_duration_seconds": float(record.duration_seconds)}

    training = record.training or {}
    if training:
        metrics["best_epoch"] = float(training["best_epoch"])
        metrics["best_validation_loss"] = float(training["best_validation_loss"])
        metrics["training_steps"] = float(training["global_step"])
        metrics["training_duration_seconds"] = float(training["duration_seconds"])

    if not record.measured:
        return metrics

    evaluation = record.evaluation or {}
    report = record.report
    metrics["evaluation_duration_seconds"] = float(evaluation.get("duration_seconds", 0.0))
    metrics["scored_examples"] = float(report.get("size", 0))
    metrics["empty_predictions"] = float(report.get("empty_predictions", 0))

    for side in ("prediction", "reference"):
        mean = report.get(f"{side}_words", {}).get("mean")
        if mean is not None:
            metrics[f"{side}_words_mean"] = float(mean)

    for variant in ROUGE_VARIANTS:
        score = record.rouge(variant)
        if score is None:
            continue
        metrics[f"{variant}_f"] = score
        bounds = record.interval(variant)
        if bounds is not None:
            metrics[f"{variant}_low"], metrics[f"{variant}_high"] = bounds

    return metrics


def build_epochs(record: RunRecord) -> tuple[dict[str, float], ...]:
    """Return the per epoch curves of a run that trained.

    Args:
        record: The run record.

    Returns:
        One mapping per epoch, in order. Empty for a run that did not train.
    """
    epochs: Any = (record.training or {}).get("epochs", [])
    if not isinstance(epochs, list):
        return ()

    curves: list[dict[str, float]] = []
    for entry in epochs:
        curves.append(
            {
                "train_loss": float(entry["train_loss"]),
                "validation_loss": float(entry["validation_loss"]),
                "learning_rate": float(entry["learning_rate"]),
                "epoch_duration_seconds": float(entry["duration_seconds"]),
            }
        )
    return tuple(curves)


def collect_artifacts(directory: Path | None) -> tuple[Path, ...]:
    """List the files of a run directory that are worth copying.

    Args:
        directory: The run directory, or ``None`` when the caller has none.

    Returns:
        The subset of :data:`TRACKED_ARTIFACTS` that exists, in declaration
        order. A missing file is skipped rather than reported: a zero shot run
        writes no training curve, and a failed one writes only its record.
    """
    if directory is None:
        return ()

    found: list[Path] = []
    for name in TRACKED_ARTIFACTS:
        path = Path(directory) / name
        if path.is_file():
            found.append(path)
    return tuple(found)


def run_name(record: RunRecord) -> str:
    """Return the name the run appears under in the tracking store.

    Args:
        record: The run record.

    Returns:
        The experiment name for a complete run, and the name suffixed with its
        status otherwise. A partial run carries no score, so it cannot climb a
        ranking; the suffix is there so it cannot be read as the experiment
        either.
    """
    if record.status == STATUS_OK:
        return record.experiment
    return f"{record.experiment}:{record.status.lower()}"


def build_payload(record: RunRecord, *, directory: Path | None = None) -> TrackedRun:
    """Turn a run record into everything the tracking store receives.

    Args:
        record: The run record, which is the only source of numbers.
        directory: The run directory, read for the artefacts to copy.

    Returns:
        The payload.
    """
    return TrackedRun(
        name=run_name(record),
        params=build_params(record),
        metrics=build_metrics(record),
        tags=build_tags(record),
        epochs=build_epochs(record),
        artifacts=collect_artifacts(directory),
    )
