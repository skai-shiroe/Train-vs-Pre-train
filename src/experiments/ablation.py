"""The ablation tables of sections 11, 12 and 17.

``make ablation`` turns the run records into the four files section 17 asks
for::

    reports/results/experiments.csv
    reports/results/ablation_dataset_size.csv
    reports/results/ablation_architecture.csv
    reports/results/qualitative_examples.json

Nothing is computed here. Every number is copied from a run record, and a cell
with no measurement behind it is left empty next to a status column. Section 17
forbids editing these files to improve a result; a generator that could invent
one would make that rule unenforceable.

**Every declared experiment gets a row.** Including the ones that never ran.
A six row table where nine experiments were declared reads as a finished study.

**Only a complete run fills a score column.** A partial run and a failed one
keep their row and their status, and their score cells stay empty. A score over
fifty documents presented next to scores over a thousand is section 44's
fabricated result with extra steps.

**Runs that were not measured the same way are not put in one table.** The
decoding budget, the beam width and the ROUGE settings travel inside each
record. If two runs of a study disagree on any of them, the table is refused
rather than written: a gap between two models measured differently says nothing
about the models, and nothing on the page would show it.

**The zero shot baseline is one row, without a proportion.** Section 11 says the
zero shot measurement does not depend on the training corpus size. Its
``dataset_percentage`` cell is empty rather than zero, because the model was not
trained on nothing, it was not trained at all. The curve of section 18 draws it
as a horizontal reference line.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from src.evaluation.evaluator import QUALITATIVE_FILE
from src.experiments.config import (
    DEFAULT_EXPERIMENTS_DIR,
    PRETRAINED_ZERO_SHOT,
    STUDIES,
    ScratchModelConfig,
)
from src.experiments.record import run_directory
from src.experiments.registry import DEFAULT_RESULTS_DIR, ExperimentRow, collect, rows_for_study
from src.metrics.rouge import REPORTED_VARIANT

#: Registry of every declared experiment, measured or not.
EXPERIMENTS_CSV = "experiments.csv"

#: Ablation of section 11: performance against the training corpus size.
DATASET_SIZE_CSV = "ablation_dataset_size.csv"

#: Ablation of section 12: performance against the depth of the encoder decoder.
ARCHITECTURE_CSV = "ablation_architecture.csv"

#: Examples selected for the qualitative analysis, gathered across the runs.
QUALITATIVE_JSON = "qualitative_examples.json"

#: Number of decimals kept in the tables. Four is what a ROUGE is read to; six
#: keeps the figures from being drawn off a rounded value.
DECIMALS = 6

#: Columns of the registry table.
EXPERIMENT_COLUMNS: tuple[str, ...] = (
    "experiment",
    "variant",
    "studies",
    "status",
    "dataset_percentage",
    "train_examples",
    "parameters",
    "split",
    "split_size",
    "scored_examples",
    "rouge1_f",
    "rouge2_f",
    "rougeL_f",
    "rougeL_low",
    "rougeL_high",
    "empty_predictions",
    "prediction_words_mean",
    "reference_words_mean",
    "best_epoch",
    "best_validation_loss",
    "training_steps",
    "training_seconds",
    "evaluation_seconds",
    "seed",
)

#: Columns of the corpus size ablation.
DATASET_SIZE_COLUMNS: tuple[str, ...] = (
    "variant",
    "dataset_percentage",
    "train_examples",
    "experiment",
    "status",
    "rouge1_f",
    "rouge2_f",
    "rougeL_f",
    "rougeL_low",
    "rougeL_high",
)

#: Columns of the architecture ablation.
ARCHITECTURE_COLUMNS: tuple[str, ...] = (
    "experiment",
    "encoder_layers",
    "decoder_layers",
    "d_model",
    "num_heads",
    "d_ff",
    "parameters",
    "status",
    "rouge1_f",
    "rouge2_f",
    "rougeL_f",
    "rougeL_low",
    "rougeL_high",
)


def cell(value: Any) -> str:
    """Render one value for a table.

    Args:
        value: The value, possibly absent.

    Returns:
        The empty string when the value is ``None``, so that a missing
        measurement reads as missing rather than as a zero, and a rounded
        decimal for a float.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{round(value, DECIMALS)}"
    return str(value)


def write_table(path: Path, columns: Sequence[str], rows: Iterable[dict[str, Any]]) -> Path:
    """Write one CSV table.

    Args:
        path: Destination file. Parent directories are created.
        columns: Header, in order.
        rows: Mappings keyed by column name.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: cell(row.get(column)) for column in columns})
    return path


def check_comparable(rows: Sequence[ExperimentRow], study: str) -> None:
    """Refuse a study whose measured runs were not scored the same way.

    Args:
        rows: The rows of one study.
        study: Study name, used in the error message.

    Raises:
        ValueError: If two measured runs disagree on the split, the decoding
            configuration or the ROUGE settings.
    """
    grouped: dict[tuple[str, str, str], list[str]] = {}
    for row in rows:
        if row.measured and row.record is not None:
            grouped.setdefault(row.record.measurement_key, []).append(row.name)

    if len(grouped) <= 1:
        return

    detail = " | ".join(
        f"{', '.join(sorted(names))}: split={key[0]} generation={key[1]} rouge={key[2]}"
        for key, names in grouped.items()
    )
    raise ValueError(
        f"The runs of the {study} study were not measured the same way, so their scores "
        f"cannot be compared: {detail}. Rerun the diverging experiments with the same "
        "evaluation block, or the gap between them will report the settings rather than "
        "the models."
    )


def scores(row: ExperimentRow) -> dict[str, Any]:
    """Return the score cells of one row.

    Args:
        row: The experiment row.

    Returns:
        One entry per score column, all ``None`` when the row holds no
        reportable measurement.
    """
    interval = row.interval(REPORTED_VARIANT)
    return {
        "rouge1_f": row.rouge("rouge1"),
        "rouge2_f": row.rouge("rouge2"),
        "rougeL_f": row.rouge(REPORTED_VARIANT),
        "rougeL_low": interval[0] if interval else None,
        "rougeL_high": interval[1] if interval else None,
    }


def experiment_row(row: ExperimentRow) -> dict[str, Any]:
    """Render one row of the registry table.

    Args:
        row: The experiment row.

    Returns:
        A mapping keyed by :data:`EXPERIMENT_COLUMNS`.
    """
    record = row.record
    evaluation: dict[str, Any] = (record.evaluation or {}) if record else {}
    training: dict[str, Any] = (record.training or {}) if record else {}
    dataset: dict[str, Any] = record.dataset if record else {}
    report = record.report if record else {}

    fields: dict[str, Any] = {
        "experiment": row.name,
        "variant": row.config.variant,
        "studies": " ".join(row.config.experiment.studies),
        "status": row.status,
        "dataset_percentage": row.config.dataset.percentage,
        "train_examples": dataset.get("train_examples"),
        "parameters": (record.model.get("parameters") if record else None),
        "split": evaluation.get("split") or row.config.evaluation.split,
        "split_size": evaluation.get("split_size"),
        "scored_examples": report.get("size"),
        "empty_predictions": report.get("empty_predictions") if row.measured else None,
        "prediction_words_mean": (
            report.get("prediction_words", {}).get("mean") if row.measured else None
        ),
        "reference_words_mean": (
            report.get("reference_words", {}).get("mean") if row.measured else None
        ),
        "best_epoch": training.get("best_epoch"),
        "best_validation_loss": training.get("best_validation_loss"),
        "training_steps": training.get("global_step"),
        "training_seconds": training.get("duration_seconds"),
        "evaluation_seconds": evaluation.get("duration_seconds"),
        "seed": row.config.experiment.seed,
    }
    fields.update(scores(row))
    return fields


def dataset_size_row(row: ExperimentRow) -> dict[str, Any]:
    """Render one row of the corpus size ablation.

    Args:
        row: The experiment row.

    Returns:
        A mapping keyed by :data:`DATASET_SIZE_COLUMNS`. The proportion of the
        zero shot baseline stays empty: it was not trained on nothing, it was
        not trained.
    """
    dataset: dict[str, Any] = row.record.dataset if row.record else {}
    fields: dict[str, Any] = {
        "variant": row.config.variant,
        "dataset_percentage": row.config.dataset.percentage,
        "train_examples": dataset.get("train_examples"),
        "experiment": row.name,
        "status": row.status,
    }
    fields.update(scores(row))
    return fields


def architecture_row(row: ExperimentRow) -> dict[str, Any]:
    """Render one row of the architecture ablation.

    Args:
        row: The experiment row.

    Returns:
        A mapping keyed by :data:`ARCHITECTURE_COLUMNS`.

    Raises:
        TypeError: If the experiment is not a from scratch run. Section 12
            varies the depth of the hand written Transformer; a baseline whose
            architecture is fixed by its checkpoint has nothing to vary.
    """
    model = row.config.model
    if not isinstance(model, ScratchModelConfig):
        raise TypeError(
            f"Experiment {row.name!r} declares the architecture study but is not a from "
            "scratch run. The study varies the depth of the hand written Transformer."
        )

    fields: dict[str, Any] = {
        "experiment": row.name,
        "encoder_layers": model.encoder_layers,
        "decoder_layers": model.decoder_layers,
        "d_model": model.d_model,
        "num_heads": model.num_heads,
        "d_ff": model.d_ff,
        "parameters": (row.record.model.get("parameters") if row.record else None),
        "status": row.status,
    }
    fields.update(scores(row))
    return fields


def sort_dataset_size(rows: Sequence[ExperimentRow]) -> list[ExperimentRow]:
    """Order the corpus size ablation for reading.

    Args:
        rows: The rows of the study.

    Returns:
        The trained variants first, by variant then by proportion, and the zero
        shot reference last: it is a horizontal line under the curve, not a
        point on it.
    """

    def key(row: ExperimentRow) -> tuple[int, str, int]:
        zero_shot = row.config.variant == PRETRAINED_ZERO_SHOT
        return (1 if zero_shot else 0, row.config.variant, row.config.dataset.percentage or 0)

    return sorted(rows, key=key)


def sort_architecture(rows: Sequence[ExperimentRow]) -> list[ExperimentRow]:
    """Order the architecture ablation by the quantity it varies.

    Args:
        rows: The rows of the study.

    Returns:
        The rows sorted by depth, then by width, so the table reads as the
        sweep it is.
    """

    def key(row: ExperimentRow) -> tuple[int, int, int, str]:
        model = row.config.model
        if isinstance(model, ScratchModelConfig):
            return (model.encoder_layers, model.decoder_layers, model.d_model, row.name)
        return (0, 0, 0, row.name)

    return sorted(rows, key=key)


def collect_qualitative(rows: Sequence[ExperimentRow], results_dir: Path) -> dict[str, Any]:
    """Gather the qualitative selections of the measured runs.

    Section 16 asks for a qualitative analysis next to the scores. Each run
    writes its own selection; this puts them side by side so the same test
    documents can be compared across models.

    Args:
        rows: Every declared experiment.
        results_dir: Directory the run directories live under.

    Returns:
        A mapping from experiment name to the selection that run wrote. An
        experiment without a complete run is absent: there is no summary to
        show for a model that never produced one.
    """
    gathered: dict[str, Any] = {}
    for row in rows:
        if not row.measured:
            continue
        path = run_directory(results_dir, row.name) / QUALITATIVE_FILE
        if path.is_file():
            gathered[row.name] = json.loads(path.read_text(encoding="utf-8"))
    return gathered


def write_qualitative(gathered: dict[str, Any], path: Path) -> Path:
    """Write the gathered qualitative selections.

    Args:
        gathered: Mapping produced by :func:`collect_qualitative`.
        path: Destination file.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gathered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def aggregate(
    *,
    experiments_dir: Path,
    results_dir: Path,
    output_dir: Path,
    studies: Sequence[str],
) -> list[Path]:
    """Build the registry table and the requested study tables.

    Args:
        experiments_dir: Directory holding the experiment files.
        results_dir: Directory holding the run directories.
        output_dir: Directory the tables are written to.
        studies: Studies to build a table for.

    Returns:
        The paths written, in the order they were produced.

    Raises:
        ValueError: If a study holds runs that were not measured the same way.
        TypeError: If a non scratch experiment declares the architecture study.
    """
    rows = collect(experiments_dir, results_dir)
    written = [
        write_table(
            Path(output_dir) / EXPERIMENTS_CSV,
            EXPERIMENT_COLUMNS,
            (experiment_row(row) for row in rows),
        )
    ]

    for study in studies:
        selected = rows_for_study(rows, study)
        check_comparable(selected, study)

        if study == "dataset_size":
            written.append(
                write_table(
                    Path(output_dir) / DATASET_SIZE_CSV,
                    DATASET_SIZE_COLUMNS,
                    (dataset_size_row(row) for row in sort_dataset_size(selected)),
                )
            )
        else:
            written.append(
                write_table(
                    Path(output_dir) / ARCHITECTURE_CSV,
                    ARCHITECTURE_COLUMNS,
                    (architecture_row(row) for row in sort_architecture(selected)),
                )
            )

    written.append(
        write_qualitative(
            collect_qualitative(rows, Path(results_dir)), Path(output_dir) / QUALITATIVE_JSON
        )
    )
    return written


def format_status(rows: Sequence[ExperimentRow]) -> str:
    """Render the summary printed after an aggregation.

    Args:
        rows: Every declared experiment.

    Returns:
        A short multi line report stating how much of the plan is measured.
    """
    measured = [row for row in rows if row.measured]
    lines = [
        f"declared         {len(rows)} experiments",
        f"measured         {len(measured)}",
    ]
    for row in rows:
        score = row.rouge(REPORTED_VARIANT)
        value = f"{score:.4f}" if score is not None else "-"
        lines.append(f"  {row.name:<24} {row.status:<8} {REPORTED_VARIANT} {value}")

    if not measured:
        lines.append(
            "No experiment has been run, so every score column is empty. "
            "The tables describe the plan, not a result."
        )
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    """Describe the command line of ``python -m src.experiments.ablation``.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments.ablation",
        description=(
            "Aggregate the run records into the tables of section 17. Nothing is "
            "computed here: a cell with no run behind it stays empty."
        ),
    )
    parser.add_argument(
        "--study",
        default="all",
        choices=(*STUDIES, "all"),
        help="Study table to build. The registry table is written every time.",
    )
    parser.add_argument(
        "--experiments",
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
        "--output", type=Path, default=None, help="Where the tables go. Defaults to --results."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Aggregate the results from the command line.

    Args:
        argv: Arguments to parse. Defaults to the process arguments.

    Returns:
        Process exit code. Zero even when nothing has been measured: an empty
        table on a repository where no experiment has run is the correct
        output, and the summary says so in words.
    """
    args = build_argument_parser().parse_args(argv)
    studies = STUDIES if args.study == "all" else (args.study,)

    written = aggregate(
        experiments_dir=args.experiments,
        results_dir=args.results,
        output_dir=args.output or args.results,
        studies=studies,
    )

    print(format_status(collect(args.experiments, args.results)))
    for path in written:
        print(f"written          {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
