"""Put what a run produced into the model registry.

Section 20 asks the backend to resolve a model through a registry rather than
through a path. Something has to fill that registry, and this is it::

    python -m src.experiments.publish --experiment scratch_100 --alias champion

The command reads the run record and the checkpoint of one experiment and writes
one version. It never runs a model and never scores one.

**Only a complete run may be published.** A partial run and a failed one carry
no measurement, so publishing one would put weights behind the API with nothing
to say about them. The registry is where a reader goes to ask which model
produced a summary and how good it was; a version whose answer is ``PARTIAL``
does not belong there.

**The measurement is copied, the checkpoint is stripped.** The scores, the
corpus checksum and the commit come from the record. The weights are reduced to
the state dictionary: the optimiser moments and the generator states exist so a
run can resume, and a served model never resumes.

**A registry entry is a copy, not a reference.** ``runs/`` is scratch space that
the next execution of the same experiment overwrites. A registry pointing into
it would serve weights that changed under it without any version changing.

**Publishing does not promote.** A new version sits beside the others until
``--alias`` or ``--activate`` says otherwise, except for the first version of a
name, which has nothing to displace.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from backend.app.registry.local import (
    WEIGHTS_PAYLOAD_KEY,
    WEIGHTS_PAYLOAD_VERSION,
    LocalModelRegistry,
)
from backend.app.registry.models import (
    ALIASES,
    DEFAULT_REGISTRY_ROOT,
    PRETRAINED,
    SCRATCH,
    ModelDraft,
    ModelSource,
    ModelVersion,
)
from src.data.config import load_pipeline_config
from src.experiments.config import DEFAULT_EXPERIMENTS_DIR
from src.experiments.record import STATUS_OK, RunRecord
from src.experiments.registry import DEFAULT_RESULTS_DIR, collect
from src.experiments.run import DEFAULT_RUNS_DIR
from src.metrics.rouge import ROUGE_VARIANTS
from src.training.checkpoint import CheckpointManager, load_checkpoint


def model_block(record: RunRecord) -> dict[str, Any]:
    """Return the model block of the experiment that produced a record.

    Args:
        record: The run record.

    Returns:
        The block, empty when the record carries no configuration.
    """
    block: Any = record.config.get("model", {})
    return dict(block) if isinstance(block, dict) else {}


def default_name(record: RunRecord) -> str:
    """Return the registry name an experiment publishes under by default.

    Section 20 names two models. Which side of the comparison a run belongs to
    is what decides, not the experiment name: ``scratch_10`` and ``scratch_100``
    are two versions of one model, not two models.

    Args:
        record: The run record.

    Returns:
        :data:`SCRATCH` or :data:`PRETRAINED`.
    """
    return SCRATCH if model_block(record).get("type") == "scratch" else PRETRAINED


def model_source(record: RunRecord) -> ModelSource:
    """Return where the weights of a run live.

    Args:
        record: The run record.

    Returns:
        :data:`ModelSource.HUGGING_FACE` for the zero shot baseline, whose
        weights were never written by this project, and
        :data:`ModelSource.CHECKPOINT` for anything that trained.
    """
    block = model_block(record)
    if block.get("type") == "pretrained" and block.get("mode") == "zero_shot":
        return ModelSource.HUGGING_FACE
    return ModelSource.CHECKPOINT


def registry_metrics(record: RunRecord) -> dict[str, float]:
    """Return the scores stored beside a published version.

    Args:
        record: The run record.

    Returns:
        One entry per ROUGE variant the run measured. A run with no reportable
        score yields nothing, which is why publishing one is refused earlier.
    """
    metrics: dict[str, float] = {}
    for variant in ROUGE_VARIANTS:
        score = record.rouge(variant)
        if score is not None:
            metrics[f"{variant}_f"] = score
    return metrics


def build_draft(record: RunRecord, *, name: str, description: str | None = None) -> ModelDraft:
    """Describe the version an experiment publishes.

    Args:
        record: The run record.
        name: Registry name the version belongs to.
        description: One line describing the version. Defaults to the
            description of the experiment file.

    Returns:
        The draft.

    Raises:
        FileNotFoundError: If the data pipeline configuration the run used has
            since been removed. Its tokeniser block is part of what the API
            needs to reproduce the preprocessing of the training.
        ValueError: If the version names a Hugging Face identifier without a
            pinned revision.
    """
    block = model_block(record)
    pipeline = load_pipeline_config(Path(str(record.dataset["config"])))

    hf_id: str | None = None
    if block.get("type") == "pretrained":
        hf_id = str(block.get("hf_id") or pipeline.tokenizer.hf_id)

    experiment: Any = record.config.get("experiment", {})
    return ModelDraft(
        name=name,
        source=model_source(record),
        hf_id=hf_id,
        revision=block.get("revision"),
        architecture=block,
        tokenizer=pipeline.tokenizer.model_dump(mode="json"),
        experiment=record.experiment,
        dataset_version=str(record.dataset.get("version") or "") or None,
        git_commit=record.provenance.get("git_commit"),
        metrics=registry_metrics(record),
        description=(
            description if description is not None else str(experiment.get("description", ""))
        ),
    )


def strip_checkpoint(checkpoint: Path, destination: Path) -> Path:
    """Write the weights of a checkpoint, and nothing else.

    Args:
        checkpoint: Checkpoint written by the training loop.
        destination: File the reduced payload is written to.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        ValueError: If it was written by another payload version.
    """
    payload = load_checkpoint(checkpoint, map_location="cpu")
    reduced = {
        "version": WEIGHTS_PAYLOAD_VERSION,
        WEIGHTS_PAYLOAD_KEY: payload[WEIGHTS_PAYLOAD_KEY],
    }
    # nosec B614 - writing is not a deserialisation risk, and the payload holds
    # tensors and one integer, which is what lets the reader stay on
    # weights_only=True.
    torch.save(reduced, destination)  # nosec B614
    return destination


def publish_experiment(
    experiment: str,
    *,
    registry: LocalModelRegistry,
    experiments_dir: Path = DEFAULT_EXPERIMENTS_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    name: str | None = None,
    description: str | None = None,
    alias: str | None = None,
    activate: bool = False,
) -> ModelVersion:
    """Publish the result of one experiment.

    Args:
        experiment: Name of the experiment to publish.
        registry: The registry to write to.
        experiments_dir: Directory holding the experiment files.
        results_dir: Directory holding the run directories.
        runs_dir: Directory holding the checkpoints.
        name: Registry name. Defaults to the side of the comparison the
            experiment belongs to.
        description: One line describing the version.
        alias: Deployment alias to point at the new version.
        activate: Whether the new version becomes the one its name serves.

    Returns:
        The published version.

    Raises:
        ValueError: If the experiment is not declared, if it never ran, if its
            run is not a complete measurement, or if the alias is unknown.
        FileNotFoundError: If the checkpoint the record names is missing.
    """
    rows = [row for row in collect(experiments_dir, results_dir) if row.name == experiment]
    if not rows:
        raise ValueError(f"No experiment named {experiment!r} is declared under {experiments_dir}.")

    record = rows[0].record
    if record is None:
        raise ValueError(
            f"Experiment {experiment!r} has never run, so there are no weights to publish."
        )
    if record.status != STATUS_OK or not record.measured:
        raise ValueError(
            f"Experiment {experiment!r} is {record.status}, so it carries no measurement. Only a "
            "complete run may be published: the registry answers what a served model scored, "
            "and a version with no score has no answer."
        )
    if alias is not None and alias not in ALIASES:
        raise ValueError(f"Unknown alias {alias!r}. Available: {', '.join(ALIASES)}.")

    draft = build_draft(record, name=name or default_name(record), description=description)

    if draft.source is not ModelSource.CHECKPOINT:
        published = registry.publish(draft, activate=activate)
    else:
        checkpoint = CheckpointManager(Path(runs_dir) / experiment).best_path
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Experiment {experiment!r} was measured but its checkpoint is gone: "
                f"{checkpoint}. The registry stores a copy of the weights, so it cannot be "
                "filled from the record alone."
            )
        with tempfile.TemporaryDirectory() as staging:
            weights = strip_checkpoint(checkpoint, Path(staging) / "weights.pt")
            published = registry.publish(draft, weights=weights, activate=activate)

    if alias is not None:
        registry.promote(alias, published.name, published.version)
    return published


def format_version(version: ModelVersion, alias: str | None, registry: LocalModelRegistry) -> str:
    """Render the summary printed after a publication.

    Args:
        version: The published version.
        alias: The alias it was promoted to, if any.
        registry: The registry it was written to.

    Returns:
        A short multi line report.
    """
    index = registry.read_index()
    lines = [
        f"published        {version.label}",
        f"source           {version.source.value}",
        f"experiment       {version.experiment}",
        f"commit           {version.git_commit or 'unknown'}",
        f"serves by default {index.current.get(version.name, 'nothing')}",
    ]
    if version.artifact is not None:
        lines.append(f"artifact         {version.artifact}")
        lines.append(f"checksum         {version.checksum}")
    for variant in ROUGE_VARIANTS:
        score = version.metrics.get(f"{variant}_f")
        if score is not None:
            lines.append(f"{variant:<16} {score:.4f}")
    if alias is not None:
        lines.append(f"alias            {alias} points at {version.label}")
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    """Describe the command line of ``python -m src.experiments.publish``.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments.publish",
        description=(
            "Publish the weights and the measurement of one complete run into the "
            "model registry of section 20."
        ),
    )
    parser.add_argument("--experiment", required=True, help="Experiment to publish.")
    parser.add_argument(
        "--name",
        default=None,
        help=f"Registry name. Defaults to {SCRATCH} or {PRETRAINED}, from the model block.",
    )
    parser.add_argument(
        "--registry", type=Path, default=DEFAULT_REGISTRY_ROOT, help="Root of the registry."
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
        "--runs", type=Path, default=DEFAULT_RUNS_DIR, help="Directory holding the checkpoints."
    )
    parser.add_argument(
        "--alias", default=None, choices=ALIASES, help="Point a deployment alias at this version."
    )
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Make this version the one its name serves. Implicit for the first one.",
    )
    parser.add_argument("--description", default=None, help="One line describing the version.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Publish one experiment from the command line.

    Args:
        argv: Arguments to parse. Defaults to the process arguments.

    Returns:
        Process exit code. One when the publication was refused, with the
        reason on the error stream.
    """
    args = build_argument_parser().parse_args(argv)
    registry = LocalModelRegistry(args.registry)

    try:
        version = publish_experiment(
            args.experiment,
            registry=registry,
            experiments_dir=args.experiments_dir,
            results_dir=args.results,
            runs_dir=args.runs,
            name=args.name,
            description=args.description,
            alias=args.alias,
            activate=args.activate,
        )
    except (ValueError, FileNotFoundError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(format_version(version, args.alias, registry))
    return 0


if __name__ == "__main__":
    sys.exit(main())
