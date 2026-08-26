"""The single command of section 41: the whole chain, in one of two modes.

``make reproduce`` is what a reader runs after cloning the repository::

    make reproduce MODE=quick   # verifies the chain, produces no result
    make reproduce MODE=full    # the campaign itself, hours of GPU

The steps section 41 asks for are the ones chained here: preparing the corpus,
running the experiments, evaluating them, replaying the ablations, aggregating
the results and drawing the figures. Every one of them already had a command;
what was missing is the order between them and the two modes.

**Quick mode is a pipeline test, not a small campaign.** Its runs train for two
optimisation steps on a truncated corpus and are scored on eight test
documents, so every record it writes carries ``PARTIAL`` and is refused by the
tables through the rules the aggregation already applies. Section 44 calls a
presented but unmeasured number a fabricated result; the way to honour that
here is not to produce a smaller number, it is to produce one no table accepts.

**Quick mode writes nowhere near the campaign.** Its records go under
``reports/quick/``, its checkpoints to ``runs/quick/``, its figures and its
generated tables beside them, and the page it injects the headline table into
is a copy of the README. A verification pass that overwrote the twelve measured
runs, or the tables of the report, would cost more than it proves. It traces
nothing either: MLflow holds the campaign, and a run that is not a result has
no business in it.

**The corpus step is skipped when the corpus is already built.** ``make data``
is idempotent, so rebuilding produces the same files and the same
``dataset_version`` at the price of the download and the tokenisation.
``--rebuild-data`` forces it, and the ``dataset_version`` is printed either way
because it is what says which corpus the rest of the chain read.

**A step that fails stops the chain.** Aggregating the tables after a sweep
that crashed would publish the previous campaign under the current code. The
summary names the step that failed, keeps the ones already done, and the exit
code is one.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

from src.data import fragments as corpus_fragments
from src.data.build import build as build_corpus
from src.data.config import load_pipeline_config
from src.data.dataset import MANIFEST_NAME
from src.experiments import ablation, figures, fragments
from src.experiments.config import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_EXPERIMENTS_DIR,
    STUDIES,
    ExperimentConfig,
    campaign_order,
    discover_experiments,
)
from src.experiments.record import STATUS_FAILED, RunRecord
from src.experiments.run import (
    DEFAULT_RESULTS_DIR,
    DEFAULT_RUNS_DIR,
    Corpus,
    load_corpus,
    run_one,
)
from src.tracking.client import DEFAULT_EXPERIMENT, build_tracker
from src.tracking.store import describe_store
from src.utils.device import release_accelerator
from src.utils.markdown import write as write_fragments

#: The two modes of section 41. ``quick`` exercises the mechanism, ``full``
#: produces the results.
MODE_QUICK = "quick"
MODE_FULL = "full"
MODES: tuple[str, ...] = (MODE_QUICK, MODE_FULL)

#: A step that ran and produced what it was asked for.
DONE = "DONE"

#: A step that had nothing left to do, which is not the same as a step that
#: succeeded and must not be printed as one.
SKIPPED = "SKIPPED"

#: A step that raised, or whose own verdict was a failure.
FAILED = "FAILED"

#: Optimisation steps a quick run is allowed. Two rather than one: a single
#: step never exercises the scheduler moving between two of them.
QUICK_MAX_STEPS = 2

#: Training examples kept by a quick run. The proportion declared by the
#: experiment file is still read, then cut here, so the run record reports the
#: proportion it was asked for next to the handful of examples it saw.
QUICK_TRAIN_EXAMPLES = 32

#: Validation examples kept by a quick run. The validation pass runs once per
#: epoch over the whole split, which on a laptop without a GPU is the longest
#: part of an otherwise instant run.
QUICK_VALIDATION_EXAMPLES = 16

#: Test documents a quick run is scored on. Passed as the evaluation limit
#: rather than cut from the split, so the record keeps the real size of the
#: split beside the eight documents it actually read, and is marked partial.
QUICK_EVALUATION_EXAMPLES = 8

#: Batch size of a quick run, on both sides.
QUICK_BATCH_SIZE = 4

#: Token budget of a quick summary. Long enough to leave the first token,
#: short enough that beam search is not what the mode measures.
QUICK_MAX_NEW_TOKENS = 16

#: Bootstrap resamples of a quick run. Non zero: the interval code path is part
#: of what the mode verifies.
QUICK_BOOTSTRAP_SAMPLES = 50

#: Where the published figures are copied for the documentation, alongside
#: ``reports/figures``. A full run is what draws both: ``make figures`` writes
#: the copy the report reads, the only one the repository tracks.
DOCS_FIGURES_DIR = Path("docs") / "assets" / "figures"

#: Root of everything a quick run writes. One directory, outside every path the
#: campaign and the documentation read.
QUICK_ROOT = Path("reports") / "quick"

#: What one step reports, and what the caller decides on.
Outcome = tuple[str, str]

#: A step, as the chain sees it: something to run that reports an outcome.
Action = Callable[[], Outcome]


@dataclass(frozen=True, slots=True)
class Destinations:
    """Where one run of the chain writes everything it produces.

    Attributes:
        results: Directory the run directories are created under.
        runs: Directory the checkpoints are written under.
        figures: Directories the four figures are drawn into. Two in full mode:
            the report reads one, the documentation site the other.
        fragments: Directory the generated Markdown tables go to.
        readme: Page carrying the headline table between its markers.
        report: Page carrying the tables its sections argue from, from both
            generators.
    """

    results: Path
    runs: Path
    figures: tuple[Path, ...]
    fragments: Path
    readme: Path
    report: Path


#: Destinations of a campaign: the paths the report, the site and the freshness
#: check all read.
FULL_DESTINATIONS = Destinations(
    results=DEFAULT_RESULTS_DIR,
    runs=DEFAULT_RUNS_DIR,
    figures=(figures.DEFAULT_FIGURES_DIR, DOCS_FIGURES_DIR),
    fragments=fragments.DEFAULT_FRAGMENTS_DIR,
    readme=fragments.DEFAULT_README,
    report=fragments.DEFAULT_REPORT,
)

#: Destinations of a verification pass: everything under one directory that
#: nothing else reads, copies of the two injected pages included.
QUICK_DESTINATIONS = Destinations(
    results=QUICK_ROOT / "results",
    runs=Path("runs") / "quick",
    figures=(QUICK_ROOT / "figures",),
    fragments=QUICK_ROOT / "_generated",
    readme=QUICK_ROOT / "README.md",
    report=QUICK_ROOT / "RAPPORT.md",
)


@dataclass(frozen=True, slots=True)
class Settings:
    """One invocation of the chain.

    Attributes:
        mode: :data:`MODE_QUICK` or :data:`MODE_FULL`.
        experiments_dir: Directory holding the experiment files.
        data_config: Data pipeline configuration the corpus is built from.
        destinations: Where the chain writes.
        tracking: Whether finished runs are mirrored into MLflow. Ignored in
            quick mode, which never traces anything.
        rebuild_data: Whether to rebuild a corpus that is already on disk.
    """

    mode: str
    experiments_dir: Path = DEFAULT_EXPERIMENTS_DIR
    data_config: Path = DEFAULT_DATA_CONFIG
    destinations: Destinations = FULL_DESTINATIONS
    tracking: bool = True
    rebuild_data: bool = False

    @property
    def quick(self) -> bool:
        """Return whether this is a verification pass.

        Returns:
            ``True`` in quick mode.
        """
        return self.mode == MODE_QUICK

    @property
    def traces(self) -> bool:
        """Return whether the runs of this invocation reach MLflow.

        Returns:
            ``True`` only for a full run that was not asked to stay offline. A
            quick run never traces: its records are partial by construction,
            and a store holding them beside the campaign would need a reader to
            tell the two apart.
        """
        return self.tracking and not self.quick


@dataclass(frozen=True, slots=True)
class StepResult:
    """What one step of the chain produced.

    Attributes:
        name: Step name, as printed.
        status: :data:`DONE`, :data:`SKIPPED` or :data:`FAILED`.
        detail: One line saying what happened, printed next to the status.
        duration_seconds: Wall clock time the step took.
    """

    name: str
    status: str
    detail: str
    duration_seconds: float

    @property
    def failed(self) -> bool:
        """Return whether this step stopped the chain.

        Returns:
            ``True`` when the step failed.
        """
        return self.status == FAILED


def destinations_for(mode: str) -> Destinations:
    """Return where a mode writes.

    Args:
        mode: :data:`MODE_QUICK` or :data:`MODE_FULL`.

    Returns:
        The destinations of that mode.

    Raises:
        ValueError: If the mode is unknown. The command line already restricts
            it; a caller of this module does not.
    """
    if mode == MODE_QUICK:
        return QUICK_DESTINATIONS
    if mode == MODE_FULL:
        return FULL_DESTINATIONS
    raise ValueError(f"Unknown mode {mode!r}. Available: {', '.join(MODES)}.")


def quick_experiment(config: ExperimentConfig) -> ExperimentConfig:
    """Cut an experiment down to what a pipeline test needs.

    The seed, the split, the architecture and the declared proportion are left
    alone: they are what the file says the experiment is, and a verification
    pass that changed them would exercise a different experiment. What is cut
    is the budget, on both sides.

    ``max_steps`` is what marks the run as capped, which
    :class:`src.experiments.record.RunRecord` turns into ``PARTIAL``. The zero
    shot baseline has no training block to cap, so it is the evaluation limit
    that keeps it out of the tables.

    Args:
        config: The declared experiment.

    Returns:
        The same experiment, with a training budget of
        :data:`QUICK_MAX_STEPS` steps and a decoding budget short enough that
        the run ends in seconds.
    """
    evaluation = config.evaluation.model_copy(
        update={
            "batch_size": QUICK_BATCH_SIZE,
            "max_new_tokens": QUICK_MAX_NEW_TOKENS,
            "num_beams": 1,
            "bootstrap_samples": QUICK_BOOTSTRAP_SAMPLES,
        }
    )
    if config.training is None:
        return config.model_copy(update={"evaluation": evaluation})

    training = config.training.model_copy(
        update={
            "epochs": 1,
            "batch_size": QUICK_BATCH_SIZE,
            "gradient_accumulation_steps": 1,
            "max_steps": QUICK_MAX_STEPS,
            "num_workers": 0,
            "log_every_steps": 1,
        }
    )
    return config.model_copy(update={"training": training, "evaluation": evaluation})


def quick_corpus(corpus: Corpus) -> Corpus:
    """Truncate the splits a quick run trains and validates on.

    The evaluation split is left whole on purpose: the run is scored on its
    first :data:`QUICK_EVALUATION_EXAMPLES` documents through the evaluation
    limit, which is what makes the record say eight of a thousand rather than
    eight of eight. A record claiming a complete split is the one thing this
    mode must not write.

    Args:
        corpus: The corpus the experiment declared, already loaded.

    Returns:
        The same corpus with a handful of training and validation examples.
    """
    return replace(
        corpus,
        train=corpus.train[:QUICK_TRAIN_EXAMPLES],
        validation=corpus.validation[:QUICK_VALIDATION_EXAMPLES],
    )


def prepare_corpus(data_config: Path, *, rebuild: bool) -> Outcome:
    """Build the frozen corpus, or report the one already there.

    Args:
        data_config: The data pipeline configuration.
        rebuild: Whether to rebuild a corpus that is already on disk.

    Returns:
        The outcome, carrying the ``dataset_version`` the rest of the chain
        reads. Version rather than a bare ``ok``: it is the only thing that
        says two machines ran on the same corpus.
    """
    pipeline = load_pipeline_config(data_config)
    manifest_path = pipeline.paths.processed / MANIFEST_NAME

    if manifest_path.is_file() and not rebuild:
        version = json.loads(manifest_path.read_text(encoding="utf-8"))["dataset_version"]
        return SKIPPED, f"corpus already built, dataset_version {version[:12]} (--rebuild-data)"

    manifest = build_corpus(pipeline)
    return DONE, f"corpus built, dataset_version {str(manifest['dataset_version'])[:12]}"


def run_experiments(settings: Settings) -> Outcome:
    """Run every declared experiment, from scratch family first.

    The order is :func:`src.experiments.config.campaign_order`, not the name
    order of the files: a sweep that stops on a crash or a keyboard interrupt
    has then measured the from scratch family before touching ``t5-small``.

    Args:
        settings: The invocation.

    Returns:
        The outcome. A sweep where one experiment crashed reports
        :data:`FAILED` and names it: the sweep itself does not stop, because
        section 44 asks for the failure to be recorded, but the chain does.
    """
    configs = campaign_order(discover_experiments(settings.experiments_dir))
    if not configs:
        return FAILED, f"no experiment file under {settings.experiments_dir}"

    tracker = build_tracker(
        enabled=settings.traces,
        tracking_uri=None,
        experiment=DEFAULT_EXPERIMENT,
    )
    if settings.traces:
        print(f"store            {describe_store()}")

    records: list[RunRecord] = []
    for config in configs:
        prepared = quick_experiment(config) if settings.quick else config
        print(f"\n=== {prepared.name} ({prepared.variant}) ===")
        records.append(
            run_one(
                prepared,
                results_dir=settings.destinations.results,
                runs_dir=settings.destinations.runs,
                limit=QUICK_EVALUATION_EXAMPLES if settings.quick else None,
                resume=False,
                tracker=tracker,
                corpus=quick_corpus(load_corpus(prepared)) if settings.quick else None,
            )
        )
        # Le run precedent a rendu ses tenseurs, pas la memoire que l'allocateur
        # en cache. Sans cette ligne, une campagne de douze experiences accumule
        # ce que les huit premieres ont reserve, et la carte du poste de
        # reference tombe sur un OOM pilote a la sixieme alors que chacune tient
        # largement seule.
        release_accelerator()

    failures = [record.experiment for record in records if record.status == STATUS_FAILED]
    if failures:
        return FAILED, f"{len(failures)} of {len(records)} failed: {', '.join(failures)}"
    return DONE, f"{len(records)} experiments run, statuses in {settings.destinations.results}"


def aggregate_tables(settings: Settings) -> Outcome:
    """Rebuild the registry table, the two ablation tables and the selections.

    Args:
        settings: The invocation.

    Returns:
        The outcome, counting the files written.
    """
    written = ablation.aggregate(
        experiments_dir=settings.experiments_dir,
        results_dir=settings.destinations.results,
        output_dir=settings.destinations.results,
        studies=STUDIES,
    )
    return DONE, f"{len(written)} tables under {settings.destinations.results}"


def draw_figures(settings: Settings) -> Outcome:
    """Draw the four figures of section 18, once per destination.

    Args:
        settings: The invocation.

    Returns:
        The outcome, counting the images written.
    """
    drawn = [
        path
        for output_dir in settings.destinations.figures
        for path in figures.build(
            experiments_dir=settings.experiments_dir,
            results_dir=settings.destinations.results,
            output_dir=output_dir,
        )
    ]
    directories = ", ".join(str(path) for path in settings.destinations.figures)
    return DONE, f"{len(drawn)} figures under {directories}"


def prepare_page(settings: Settings, source: Path, destination: Path) -> Path:
    """Return the page a generated table is injected into.

    In full mode that is the page of the repository, which is the point of the
    step. In quick mode it is a copy: the regions have to exist for the
    injection to run at all, and rewriting the README or the report with the
    scores of a verification pass is exactly what the mode exists not to do.

    Args:
        settings: The invocation.
        source: The tracked page, read in quick mode to seed the copy.
        destination: Where the page to inject into belongs for this mode.

    Returns:
        The path of the page to inject into.
    """
    if not settings.quick:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    return destination


def synchronise_documentation(settings: Settings) -> Outcome:
    """Regenerate every Markdown table derived from the run records and the corpus.

    Both generators run: the campaign tables come from the run records, the
    corpus tables from the manifest the first step wrote or found. A chain that
    refreshed one and not the other would leave a page describing the previous
    build beside a page describing this one.

    **They are written in two passes, and the order is not free.** Both inject
    into the report, each into its own regions, and each renders the whole page
    from what is on disk. Rendering both before writing would hand the single
    write two full pages built from the same starting text, and the last one in
    the mapping would silently drop the other's tables. So the campaign tables
    are written first, and the corpus generator then reads the page they
    produced.

    Args:
        settings: The invocation.

    Returns:
        The outcome, counting the files written.
    """
    pipeline = load_pipeline_config(settings.data_config)
    destinations = settings.destinations
    report = prepare_page(settings, fragments.DEFAULT_REPORT, destinations.report)

    written = write_fragments(
        fragments.build(
            experiments_dir=settings.experiments_dir,
            results_dir=destinations.results,
            output_dir=destinations.fragments,
            readme=prepare_page(settings, fragments.DEFAULT_README, destinations.readme),
            report=report,
        )
    )
    written += write_fragments(
        corpus_fragments.build(
            processed_dir=pipeline.paths.processed,
            output_dir=destinations.fragments,
            report=report,
        )
    )

    # The report is written by both passes. Counting it once keeps the line
    # reporting how many tables exist rather than how many writes happened.
    return DONE, f"{len(set(written))} generated tables under {destinations.fragments}"


def steps(settings: Settings) -> tuple[tuple[str, Action], ...]:
    """Return the chain of section 41, in order.

    Evaluation is not a step of its own: every experiment is scored by the run
    that produced it, on the split its file declares, which is what keeps the
    weights that were measured and the weights that were selected the same
    weights.

    Args:
        settings: The invocation.

    Returns:
        One ``(name, action)`` pair per step, ready to be run in order.
    """
    return (
        ("data", lambda: prepare_corpus(settings.data_config, rebuild=settings.rebuild_data)),
        ("experiments", lambda: run_experiments(settings)),
        ("tables", lambda: aggregate_tables(settings)),
        ("figures", lambda: draw_figures(settings)),
        ("documentation", lambda: synchronise_documentation(settings)),
    )


def run_step(name: str, action: Action) -> StepResult:
    """Run one step and turn a crash into a recorded failure.

    Args:
        name: Step name, as printed.
        action: What the step does.

    Returns:
        The result. Every exception is caught: the chain reports which step
        broke and what it raised, which is more useful than a traceback ending
        in a library the reader did not call.
    """
    started = perf_counter()
    try:
        status, detail = action()
    except Exception as error:  # noqa: BLE001 - the chain reports the step that broke
        return StepResult(
            name, FAILED, f"{type(error).__name__}: {error}", perf_counter() - started
        )
    return StepResult(name, status, detail, perf_counter() - started)


def reproduce(settings: Settings) -> list[StepResult]:
    """Run the chain, stopping at the first step that fails.

    Args:
        settings: The invocation.

    Returns:
        One result per step attempted. The list is shorter than the chain when
        a step failed, and the summary says which one.
    """
    results: list[StepResult] = []
    for name, action in steps(settings):
        print(f"\n--- {name} ---")
        result = run_step(name, action)
        print(f"{result.status:<8} {name}: {result.detail}")
        results.append(result)
        if result.failed:
            break
    return results


def format_summary(settings: Settings, results: Sequence[StepResult]) -> str:
    """Render the report printed once the chain is over.

    Args:
        settings: The invocation.
        results: What each attempted step produced.

    Returns:
        A short multi line report. A quick run says in words that it measured
        nothing, because a green chain is the one thing a reader could mistake
        for a campaign.
    """
    lines = [f"mode             {settings.mode}", f"steps            {len(results)} of 5 attempted"]
    lines.extend(
        f"  {result.name:<16} {result.status:<8} {result.duration_seconds:7.1f}s  {result.detail}"
        for result in results
    )

    failed = [result for result in results if result.failed]
    if failed:
        lines.append(f"\nThe chain stopped on step {failed[0].name!r}. Nothing after it ran.")
        return "\n".join(lines)

    if settings.quick:
        lines.append(
            "\nQUICK RUN: every record is PARTIAL and stays out of every table. "
            f"Nothing outside {QUICK_ROOT} was written, and nothing was traced. "
            "The scientific results come from MODE=full."
        )
    return "\n".join(lines)


def _configure_logging(verbose: bool) -> None:
    """Attach a handler to the root logger, for the command line only.

    The chain is long and its steps log on their own loggers; without a handler
    a full run would print its banner and then nothing for hours.

    Args:
        verbose: Whether to emit debug level records.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Describe the command line of ``python -m src.experiments.reproduce``.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments.reproduce",
        description=(
            "Replay the whole scientific chain: corpus, experiments, ablations, tables "
            "and figures. MODE=quick verifies the mechanism and produces no result; "
            "MODE=full produces the campaign."
        ),
    )
    parser.add_argument(
        "--mode",
        default=MODE_QUICK,
        choices=MODES,
        help="quick verifies the chain on a truncated corpus, full runs the campaign.",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=DEFAULT_EXPERIMENTS_DIR,
        help="Directory holding the experiment files.",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help="Data pipeline configuration the corpus is built from.",
    )
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Rebuild the corpus even when it is already on disk.",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Do not send the runs to MLflow. A quick run never sends them anyway.",
    )
    parser.add_argument("--verbose", action="store_true", help="Emit debug level logs.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the chain from the command line.

    Args:
        argv: Arguments to parse. Defaults to the process arguments.

    Returns:
        Process exit code. One when a step failed, so that a chain that stopped
        halfway cannot end green.
    """
    args = build_argument_parser().parse_args(argv)
    _configure_logging(args.verbose)

    settings = Settings(
        mode=args.mode,
        experiments_dir=args.experiments_dir,
        data_config=args.data_config,
        destinations=destinations_for(args.mode),
        tracking=not args.no_tracking,
        rebuild_data=args.rebuild_data,
    )

    results = reproduce(settings)
    print(f"\n{format_summary(settings, results)}")
    return 1 if any(result.failed for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
