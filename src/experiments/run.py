"""One experiment, end to end: build, train, evaluate, record.

Section 14 asks for a single command per experiment::

    python -m src.experiments.run --config configs/experiments/scratch_10.yaml

Everything that experiment varies is in that file. This module only decides
where the artefacts go and how a failure is recorded.

**The evaluated weights are read back from the checkpoint.** When training
ends, the weights in memory are those of the last epoch, and early stopping
exists precisely because that is not the best epoch. Both models are therefore
rebuilt from the best checkpoint before being measured. The reload costs one
model construction and buys two things: the reported score belongs to the
weights the run selected, and a checkpoint that cannot be reloaded fails here,
during the run that wrote it, rather than the day someone tries to reuse it.

**Both sides go through the same three steps.** The from scratch Transformer and
``t5-small`` differ in how they are built and in their loss adapter, and in
nothing else: same corpus, same loader, same trainer, same decoding, same
metric. The zero shot baseline skips the training step because its weights never
move, not because it takes another path.

**A failure is a result.** Section 44 asks for ``FAILED`` on an experiment that
crashed. In ``--all`` mode an experiment that raises does not stop the sweep: it
writes a record carrying its status and its error, and the run continues. The
record is written to the canonical directory even when an older successful one
sits there, because a score the current code can no longer produce is not a
result any more.

**The record is written first, then traced.** Section 19 asks for every
experiment to reach MLflow, and :mod:`src.tracking` builds what it sends out of
the record that was just written. Tracking therefore cannot invent a number, and
a store that is unreachable costs the mirror of a measurement rather than the
measurement itself. A failed run is traced too: an experiment that crashed is
part of what the plan produced.

**An untraceable campaign says so before it starts, and starts anyway.** The
provenance of a run reaches its record as a tag, which is only read once the
measurement is over. The campaign of August 2026 was learnt that way: it ran from
a directory without a ``.git``, and the commit its records name resolves nowhere,
so the results of the report cannot be tied to a state of the code. The warning
:func:`untraceable_warning` builds is printed before the first experiment, since
that is the only moment where knowing costs nothing. It does not refuse to run:
an unrecorded commit is a degraded run, and an experiment nobody is allowed to
start is worse than one that has to be documented.

**The command line owns the logging configuration.**
:class:`src.training.callbacks.LoggingCallback` reports the progress of the loop
on the ``syntra.training`` logger, but a module that calls
:func:`logging.basicConfig` when it is imported steals a decision belonging to
whoever runs it. :func:`main` configures the root logger because it is the entry
point; importing this module still changes nothing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from src.data.config import DataPipelineConfig, load_pipeline_config
from src.data.dataset import (
    build_dataloader,
    load_manifest,
    load_split,
    load_training_proportion,
)
from src.data.example import Example
from src.data.tokenize import build_tokenizer
from src.evaluation.evaluator import EvaluationResult, evaluate_summarizer
from src.evaluation.evaluator import write_results as write_evaluation
from src.experiments.config import (
    DEFAULT_EXPERIMENTS_DIR,
    ExperimentConfig,
    ScratchModelConfig,
    discover_experiments,
    load_experiment_config,
)
from src.experiments.record import (
    HISTORY_FILE,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_PARTIAL,
    RunRecord,
    run_directory,
    write_record,
)
from src.metrics.rouge import ROUGE_VARIANTS
from src.models.pretrained.base import BaselineConfig, PretrainedSummarizer
from src.models.pretrained.factory import build_summarizer, get_baseline_class
from src.models.scratch.summarizer import ScratchSummarizer
from src.models.scratch.transformer import ScratchTransformer
from src.tracking.client import DEFAULT_EXPERIMENT, Tracker, build_tracker, log_safely
from src.tracking.live import LiveMetricsCallback, LiveRun, finish_safely, open_live_run
from src.tracking.model import log_model_safely, should_log
from src.tracking.store import describe_store
from src.tracking.payload import build_payload
from src.tracking.provenance import UNKNOWN, describe_provenance
from src.training.callbacks import default_callbacks
from src.training.checkpoint import CheckpointManager, load_checkpoint
from src.training.sampler import build_training_dataloader
from src.training.state import TrainingResult
from src.training.trainer import make_scratch_batch_loss, train_model
from src.utils.device import describe_hardware, resolve_device
from src.utils.seed import set_seed

#: Where the run directories are created, per section 17.
DEFAULT_RESULTS_DIR = Path("reports") / "results"

#: Where the checkpoints of a run are written. Outside the results directory
#: because the two have different lifetimes: a run directory is a few kilobytes
#: of records that the tables read, a checkpoint is hundreds of megabytes that
#: the next run of the same experiment overwrites.
DEFAULT_RUNS_DIR = Path("runs")

#: The two concrete summarisers an experiment can build. The evaluator only
#: needs :class:`src.evaluation.protocol.Summarizer`; the trainer needs the
#: module underneath, which the protocol deliberately does not expose.
AnySummarizer = ScratchSummarizer | PretrainedSummarizer


@dataclass(frozen=True, slots=True)
class Corpus:
    """The frozen corpus, loaded once for one experiment.

    Attributes:
        pipeline: The data pipeline configuration that located everything else.
        tokenizer: The shared tokeniser.
        dataset_version: Checksum of the working corpus, traced with the run.
        train: Training examples of the requested proportion. Empty for a run
            that does not train.
        validation: Validation split. Empty for a run that does not train.
        evaluation: Examples of the split the run is scored on.
    """

    pipeline: DataPipelineConfig
    tokenizer: Any
    dataset_version: str
    train: tuple[Example, ...]
    validation: tuple[Example, ...]
    evaluation: tuple[Example, ...]

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary the from scratch model must cover.

        ``len`` rather than ``vocab_size``: it counts the tokens added after
        the vocabulary was built, and an embedding table one row short would
        raise on the first document that uses one.

        Returns:
            The number of identifiers the tokeniser can produce.
        """
        return int(len(self.tokenizer))


def load_corpus(config: ExperimentConfig) -> Corpus:
    """Load everything the experiment reads from disk.

    Args:
        config: The experiment configuration.

    Returns:
        The corpus, with the training split reduced to the declared proportion.

    Raises:
        FileNotFoundError: If the corpus was never built. Run ``make data``.
        KeyError: If the declared proportion is absent from the manifest.
    """
    pipeline = load_pipeline_config(config.dataset.config)
    manifest = load_manifest(pipeline.paths.processed)

    train: tuple[Example, ...] = ()
    validation: tuple[Example, ...] = ()
    if config.dataset.percentage is not None:
        train = load_training_proportion(pipeline.paths.processed, config.dataset.percentage)
        validation = load_split(pipeline.paths.processed, "validation")

    return Corpus(
        pipeline=pipeline,
        tokenizer=build_tokenizer(pipeline.tokenizer.hf_id),
        dataset_version=manifest.dataset_version,
        train=train,
        validation=validation,
        evaluation=load_split(pipeline.paths.processed, config.evaluation.split),
    )


def build_experiment_summarizer(
    config: ExperimentConfig,
    corpus: Corpus,
    *,
    state_dict: Mapping[str, torch.Tensor] | None = None,
) -> AnySummarizer:
    """Build the summariser of an experiment, trained or not.

    Args:
        config: The experiment configuration.
        corpus: The loaded corpus, which carries the tokeniser.
        state_dict: Weights of the best checkpoint. When supplied the
            summariser is marked as trained, which is what the run record
            reports. The mark is never inferred: a randomly initialised
            Transformer produces summaries too, and filing its score under the
            trained model would be a fabricated result.

    Returns:
        The summariser, on the CPU.
    """
    if isinstance(config.model, ScratchModelConfig):
        architecture = config.model.architecture(
            vocab_size=corpus.vocab_size,
            pad_token_id=int(corpus.tokenizer.pad_token_id),
            eos_token_id=int(corpus.tokenizer.eos_token_id),
        )
        model = ScratchTransformer(architecture)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return ScratchSummarizer(
            model,
            corpus.tokenizer,
            corpus.pipeline.tokenizer,
            trained=state_dict is not None,
        )

    baseline_config = BaselineConfig.from_tokenizer_config(
        corpus.pipeline.tokenizer,
        hf_id=config.model.hf_id,
        revision=config.model.revision,
    )
    if state_dict is None:
        return build_summarizer(config.model.baseline, baseline_config, fine_tuned=False)
    return get_baseline_class(config.model.baseline).from_checkpoint(baseline_config, state_dict)


def train_experiment(
    config: ExperimentConfig,
    corpus: Corpus,
    summarizer: AnySummarizer,
    *,
    output_dir: Path,
    history_path: Path,
    resume_from: Path | None = None,
    live: LiveRun | None = None,
) -> TrainingResult:
    """Train the model of an experiment on its corpus proportion.

    Args:
        config: The experiment configuration. Its training block is required.
        corpus: The loaded corpus.
        summarizer: The untrained summariser wrapping the model to optimise.
        output_dir: Directory receiving the checkpoints.
        history_path: File the training curves are written to.
        resume_from: Checkpoint or directory to continue from.
        live: Run to stream the loop into, so the training is followable before
            it ends. ``None`` trains exactly as before and the store sees the
            run once it is over.

    Returns:
        What the run produced.

    Raises:
        ValueError: If the configuration carries no training block. The loader
            already refuses such a file, so reaching this is a programming
            error rather than a user one.
    """
    if config.training is None:
        raise ValueError(f"Experiment {config.name!r} has no training block.")

    settings = config.training
    train_loader, _ = build_training_dataloader(
        corpus.train,
        corpus.tokenizer,
        corpus.pipeline.tokenizer,
        batch_size=settings.batch_size,
        seed=settings.seed,
        group_by_length=settings.group_by_length,
        mega_batch_factor=settings.mega_batch_factor,
        num_workers=settings.num_workers,
    )
    validation_loader = build_dataloader(
        corpus.validation,
        corpus.tokenizer,
        corpus.pipeline.tokenizer,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
    )

    batch_loss = (
        make_scratch_batch_loss(settings.label_smoothing)
        if isinstance(summarizer, ScratchSummarizer)
        else summarizer.batch_loss(settings.label_smoothing)
    )

    callbacks = list(
        default_callbacks(log_every_steps=settings.log_every_steps, history_path=history_path)
    )
    if live is not None:
        callbacks.append(LiveMetricsCallback(live, every_steps=settings.log_every_steps))

    return train_model(
        summarizer.model,
        settings,
        train_loader=train_loader,
        validation_loader=validation_loader,
        output_dir=output_dir,
        batch_loss=batch_loss,
        callbacks=callbacks,
        metadata={
            "experiment": config.name,
            "dataset_version": corpus.dataset_version,
            "dataset_percentage": config.dataset.percentage or 0,
            "seed": config.experiment.seed,
        },
        resume_from=resume_from,
    )


def evaluate_experiment(
    config: ExperimentConfig,
    corpus: Corpus,
    summarizer: AnySummarizer,
    *,
    limit: int | None = None,
) -> EvaluationResult:
    """Score the final weights of an experiment on its evaluation split.

    Args:
        config: The experiment configuration.
        corpus: The loaded corpus.
        summarizer: The summariser to measure, already on its device.
        limit: Score the first ``limit`` examples only, which marks the run as
            partial.

    Returns:
        The evaluation result.
    """
    examples: Sequence[Example] = corpus.evaluation
    split_size = len(examples)
    if limit is not None:
        examples = examples[:limit]

    settings = config.evaluation
    return evaluate_summarizer(
        summarizer,
        examples,
        split=settings.split,
        split_size=split_size,
        generation=settings.generation(),
        rouge=settings.rouge(config.experiment.seed),
        batch_size=settings.batch_size,
    )


def best_weights(name: str, result: TrainingResult, checkpoints: Path) -> Path:
    """Return the checkpoint holding the weights the run selected.

    A run that resumed a finished training runs no epoch, so it reports no best
    checkpoint of its own while the previous run left one on disk. Falling back
    to it is what makes ``--resume`` idempotent: rerunning a finished experiment
    re-measures the model it produced instead of refusing to.

    Args:
        name: Experiment name, used in the error message.
        result: What the training loop produced.
        checkpoints: Directory the checkpoints were written to.

    Returns:
        The path of the best checkpoint.

    Raises:
        RuntimeError: If there are no weights to measure at all.
    """
    if result.best_checkpoint is not None:
        return result.best_checkpoint

    stored = CheckpointManager(checkpoints).best_path
    if stored.is_file():
        return stored

    raise RuntimeError(
        f"Experiment {name!r} produced no checkpoint, so there are no weights to "
        "measure. A training whose validation loss never improved on its first "
        "epoch is the usual cause."
    )


def trace_model(record: RunRecord, summarizer: AnySummarizer, live: LiveRun | None) -> None:
    """Put the measured weights into the run that is still open.

    Called between the record being written and the run being closed, for the
    ordering :mod:`src.tracking.client` states: nothing reaches the store that
    was not written first, and the checkpoint these weights come from is on
    disk before this runs.

    **Only into an open run.** MLflow logs a model into whatever run is active,
    and starting one here would file the weights under a second run with no
    score in it. Without a live run there is nothing to log into, and the
    weights stay where they already are, under ``runs/``.

    Args:
        record: The record that was just written, read for its status.
        summarizer: The evaluated summariser, holding the weights the reported
            score was measured on.
        live: The run opened before the training, or ``None``.
    """
    if live is None or not should_log(record.status):
        return
    log_model_safely(summarizer, experiment=record.experiment)


def trace(
    record: RunRecord,
    directory: Path,
    tracker: Tracker | None,
    live: LiveRun | None = None,
) -> None:
    """Mirror a finished run into the tracking store.

    Args:
        record: The record that was just written to disk.
        directory: The run directory, read for the artefacts to copy.
        tracker: The tracker to send to, or ``None`` when the caller traces
            nothing. The default is ``None`` on purpose: a test that runs an
            experiment must not reach a store, and the command line is what
            turns tracing on.
        live: The run opened before the training, when there is one. Closing it
            with the payload is what keeps one run per experiment rather than
            one open run beside one finished mirror.
    """
    if tracker is None:
        return
    payload = build_payload(record, directory=directory)
    if live is not None:
        finish_safely(live, payload)
        return
    log_safely(tracker, payload)


def execute(
    config: ExperimentConfig,
    *,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    limit: int | None = None,
    resume: bool = False,
    tracker: Tracker | None = None,
    live: LiveRun | None = None,
    corpus: Corpus | None = None,
) -> RunRecord:
    """Run one experiment and write everything it produced.

    Args:
        config: The experiment configuration.
        results_dir: Directory the run directory is created under.
        runs_dir: Directory the checkpoints are written under.
        limit: Score the first ``limit`` examples only.
        resume: Continue the training from the checkpoints already in the run
            directory, when there are any.
        tracker: Where the finished run is mirrored, per section 19.
        live: Run opened by the caller before the training starts. Owned by
            :func:`run_one` rather than here, because an experiment that raises
            must still close the run it opened, and only the caller that
            catches the failure can do that.
        corpus: The corpus to run on, when the caller has one. Defaults to what
            the experiment file points at, which is the only corpus a reported
            run may use. The parameter exists for the quick mode of
            :mod:`src.experiments.reproduce`, which hands over a truncated
            corpus to exercise the chain in seconds; every run it produces is
            partial, so no table can pick one up.

    Returns:
        The record that was written.
    """
    started = perf_counter()
    set_seed(config.experiment.seed)

    corpus = load_corpus(config) if corpus is None else corpus
    partial = limit is not None or config.capped
    directory = run_directory(results_dir, config.name, partial=partial)

    training_result: TrainingResult | None = None
    state_dict: Mapping[str, torch.Tensor] | None = None

    if config.trains:
        checkpoints = Path(runs_dir) / config.name
        training_result = train_experiment(
            config,
            corpus,
            build_experiment_summarizer(config, corpus),
            output_dir=checkpoints,
            history_path=directory / HISTORY_FILE,
            resume_from=checkpoints if resume and checkpoints.is_dir() else None,
            live=live,
        )
        payload = load_checkpoint(
            best_weights(config.name, training_result, checkpoints), map_location="cpu"
        )
        state_dict = payload["model"]

    device = resolve_device(config.training.device if config.training else "auto")
    summarizer = build_experiment_summarizer(config, corpus, state_dict=state_dict).to(device)
    evaluation = evaluate_experiment(config, corpus, summarizer, limit=limit)

    record = RunRecord(
        experiment=config.name,
        status=STATUS_PARTIAL if (partial or evaluation.partial) else STATUS_OK,
        config=config.to_dict(),
        dataset={
            "config": str(config.dataset.config),
            "name": corpus.pipeline.name,
            "version": corpus.dataset_version,
            "percentage": config.dataset.percentage,
            "train_examples": len(corpus.train),
            "validation_examples": len(corpus.validation),
        },
        model=evaluation.model,
        hardware=describe_hardware(),
        provenance=describe_provenance().to_dict(),
        training=training_result.to_dict() if training_result else None,
        evaluation=evaluation.to_dict(),
        duration_seconds=perf_counter() - started,
    )

    write_evaluation(evaluation, directory)
    write_record(record, directory)
    trace_model(record, summarizer, live)
    trace(record, directory, tracker, live)
    return record


def failure_record(config: ExperimentConfig, error: Exception, duration: float) -> RunRecord:
    """Build the record of an experiment that raised.

    Args:
        config: The experiment that was attempted.
        error: The exception that ended it.
        duration: Wall clock time spent before the failure.

    Returns:
        A record carrying :data:`src.experiments.record.STATUS_FAILED` and no
        score. Section 44 asks for the failure to be visible; a run with an
        empty score column and a status is visible, an absent row is not.
    """
    return RunRecord(
        experiment=config.name,
        status=STATUS_FAILED,
        config=config.to_dict(),
        dataset={"config": str(config.dataset.config), "percentage": config.dataset.percentage},
        model={},
        hardware=describe_hardware(),
        provenance=describe_provenance().to_dict(),
        error=f"{type(error).__name__}: {error}",
        duration_seconds=duration,
    )


def run_one(
    config: ExperimentConfig,
    *,
    results_dir: Path,
    runs_dir: Path,
    limit: int | None,
    resume: bool,
    tracker: Tracker | None = None,
    corpus: Corpus | None = None,
) -> RunRecord:
    """Run one experiment, turning a crash into a recorded failure.

    Args:
        config: The experiment configuration.
        results_dir: Directory the run directory is created under.
        runs_dir: Directory the checkpoints are written under.
        limit: Score the first ``limit`` examples only.
        resume: Continue from the checkpoints of the run directory.
        tracker: Where the finished run is mirrored, per section 19.
        corpus: The corpus to run on, passed straight to :func:`execute`.

    Returns:
        The record that was written, successful or failed.
    """
    started = perf_counter()
    # Opened here rather than inside execute: an experiment that raises must
    # still close the run it opened, and this is the frame that catches the
    # failure. A tracker that cannot open one yields None, and the run is
    # mirrored at the end exactly as it was before.
    live = open_live_run(tracker, config.name)
    try:
        return execute(
            config,
            results_dir=results_dir,
            runs_dir=runs_dir,
            limit=limit,
            resume=resume,
            tracker=tracker,
            live=live,
            corpus=corpus,
        )
    except Exception as error:
        # Every exception is caught on purpose: recording the failure is the
        # contract of section 44, and a sweep must not stop at its first crash.
        record = failure_record(config, error, perf_counter() - started)
        directory = run_directory(results_dir, config.name)
        write_record(record, directory)
        trace(record, directory, tracker, live)
        print(f"{config.name}: {record.error}", file=sys.stderr)
        return record


def _configure_logging(verbose: bool) -> None:
    """Attach a handler to the root logger, for the command line only.

    Without this call the root logger keeps its default level and the INFO
    records of :class:`src.training.callbacks.LoggingCallback` reach no output:
    a long training would print its opening banner, then nothing until the
    final summary. The store has been followable during a run since
    :mod:`src.tracking.live`; this is the console catching up.

    Args:
        verbose: Whether to emit debug level records.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


def untraceable_warning(provenance: Mapping[str, str]) -> str | None:
    """Describe what the results of a campaign started now could not be tied to.

    Args:
        provenance: What :func:`src.tracking.provenance.describe_provenance`
            answered, rendered as tags.

    Returns:
        The warning to show before anything runs, or ``None`` when the commit is
        known and the working tree clean.
    """
    reasons = []
    if provenance.get("git_commit", UNKNOWN) == UNKNOWN:
        reasons.append("git cannot name the commit this code is at")
    if provenance.get("git_dirty") == "true":
        reasons.append("the working tree carries uncommitted changes")
    if not reasons:
        return None

    return (
        f"warning: {', and '.join(reasons)}. The records this run writes will not "
        f"tie their numbers to a state of the code anyone can check out again."
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Describe the command line of ``python -m src.experiments.run``.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments.run",
        description=(
            "Run one experiment, or every declared experiment, and write its results "
            "under reports/results."
        ),
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--config", type=Path, help="Experiment configuration to run.")
    selection.add_argument(
        "--all",
        action="store_true",
        help="Run every experiment declared under the experiments directory, in name order.",
    )

    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=DEFAULT_EXPERIMENTS_DIR,
        help="Directory holding the experiment files, read by --all.",
    )
    parser.add_argument(
        "--results", type=Path, default=DEFAULT_RESULTS_DIR, help="Where the run directories go."
    )
    parser.add_argument(
        "--runs", type=Path, default=DEFAULT_RUNS_DIR, help="Where the checkpoints go."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate the first N examples only. Marks the run as partial.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the training from the checkpoints already in the run directory.",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Do not send the run to MLflow. The record is written either way.",
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
    parser.add_argument("--verbose", action="store_true", help="Emit debug level logs.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one experiment, or all of them, from the command line.

    Args:
        argv: Arguments to parse. Defaults to the process arguments.

    Returns:
        Process exit code. One when at least one experiment failed, so that a
        sweep cannot end green with a missing measurement.
    """
    args = build_argument_parser().parse_args(argv)
    _configure_logging(args.verbose)

    warning = untraceable_warning(describe_provenance().to_dict())
    if warning is not None:
        print(warning, file=sys.stderr)

    configs = (
        discover_experiments(args.experiments_dir)
        if args.all
        else [load_experiment_config(args.config)]
    )
    if not configs:
        print(f"No experiment file found under {args.experiments_dir}.", file=sys.stderr)
        return 1

    tracker = build_tracker(
        enabled=not args.no_tracking,
        tracking_uri=args.tracking_uri,
        experiment=args.tracking_experiment,
    )
    # Printed before the first run rather than after the last: a campaign that
    # traced into a local SQLite file when it meant to reach the shared
    # database is a thing to discover in the first second, not in six hours.
    if not args.no_tracking:
        print(f"store            {describe_store(args.tracking_uri)}")

    failures = 0
    for config in configs:
        print(f"\n=== {config.name} ({config.variant}) ===")
        record = run_one(
            config,
            results_dir=args.results,
            runs_dir=args.runs,
            limit=args.limit,
            resume=args.resume,
            tracker=tracker,
        )
        if record.status == STATUS_FAILED:
            failures += 1
            continue

        print(format_record(record))
        directory = run_directory(
            args.results, config.name, partial=record.status == STATUS_PARTIAL
        )
        print(f"written to       {directory}")

    if failures:
        print(f"\n{failures} experiment(s) failed.", file=sys.stderr)
        return 1
    return 0


def format_record(record: RunRecord) -> str:
    """Render the one screen summary of a finished run.

    Args:
        record: The record just written.

    Returns:
        A short multi line report. A partial run says so, and its scores are
        not printed as a result.
    """
    identity = record.model.get("model") or record.model.get("baseline") or "unknown"
    lines = [
        f"status           {record.status}",
        f"model            {identity} ({record.model.get('mode', 'unknown')})",
    ]

    if record.training is not None:
        lines.append(
            f"training         best epoch {record.training['best_epoch']}, "
            f"validation loss {record.training['best_validation_loss']:.4f}, "
            f"{record.training['global_step']} steps"
        )

    if record.evaluation is None:
        return "\n".join(lines)

    report = record.report
    lines.append(
        f"split            {record.evaluation['split']}, "
        f"{report.get('size', 0)} of {record.evaluation['split_size']} examples"
    )
    for variant in ROUGE_VARIANTS:
        score = record.rouge(variant)
        if score is None:
            continue
        bounds = record.interval(variant)
        spread = f"  [{bounds[0]:.4f}, {bounds[1]:.4f}]" if bounds else ""
        lines.append(f"{variant:<16} {score:.4f}{spread}")

    lines.append(f"empty summaries  {report.get('empty_predictions', 0)}")
    if record.status == STATUS_PARTIAL:
        lines.append("PARTIAL RUN: this run is not a result and is excluded from the tables.")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
