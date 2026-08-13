"""The figures of section 18, drawn from the run records.

``python -m src.experiments.figures`` turns the records into the four images
section 18 asks for::

    reports/figures/performance_vs_dataset_size.png
    reports/figures/training_loss.png
    reports/figures/validation_loss.png
    reports/figures/model_comparison.png

The main one answers the question the project exists to answer: how the from
scratch Transformer and the fine tuned baseline move as the training corpus
grows, with the zero shot measurement underneath as a reference.

**Nothing is computed here.** Every point is copied from a run record, like the
tables of :mod:`src.experiments.ablation`. A figure is the most quoted output of
the campaign and the least verifiable one: a reader sees a curve, not the file
it came from. Recomputing a score at drawing time would let the image disagree
with the table beside it, with no way to tell which one is the result.

**Only a complete run becomes a point.** A partial run and a failed one carry no
reportable score, so they are not drawn, and the figure names them in a footnote
rather than dropping them silently. An image is separated from its table the
moment it is pasted into a slide; one that hides what is missing reads as a
finished study.

**The intervals are drawn, always.** The bootstrap bounds travel with every
score in the record. Without them a reader ranks two models on a gap of two
thousandths, which is exactly what the architecture ablation shows is not
established.

**The zero shot baseline is a horizontal line, not a point.** Its weights do not
depend on the training corpus, so placing it at any proportion would claim a
measurement that was never taken. Section 11 gives it one row in the table and
this is its equivalent on the curve.

**Runs measured differently are not put on one figure.** The check the tables
apply is applied here too, and by the same function. A gap between two models
scored under different decoding settings reports the settings, and a curve shows
that even less than a table does.

The labels are French because the figures are read beside the French report;
the code around them stays English like the rest of ``src``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.experiments.ablation import check_comparable
from src.experiments.config import (
    DEFAULT_EXPERIMENTS_DIR,
    PRETRAINED_FINE_TUNED,
    PRETRAINED_ZERO_SHOT,
    SCRATCH,
)
from src.experiments.registry import DEFAULT_RESULTS_DIR, ExperimentRow, collect, rows_for_study
from src.metrics.rouge import REPORTED_VARIANT, ROUGE_VARIANTS

#: Where the figures go, per section 6.
DEFAULT_FIGURES_DIR = Path("reports") / "figures"

#: Performance against the training corpus size, the main figure of section 18.
PERFORMANCE_FIGURE = "performance_vs_dataset_size.png"

#: Training loss of every run that trained.
TRAINING_LOSS_FIGURE = "training_loss.png"

#: Validation loss of every run that trained.
VALIDATION_LOSS_FIGURE = "validation_loss.png"

#: ROUGE of every measured run, side by side.
MODEL_COMPARISON_FIGURE = "model_comparison.png"

#: Resolution of the written images. High enough to stay readable in a report,
#: low enough that the four files stay under a megabyte together.
DPI = 150

#: How the two trained families are named on a figure.
VARIANT_LABELS: dict[str, str] = {
    SCRATCH: "Transformer from scratch",
    PRETRAINED_FINE_TUNED: "T5-small fine-tune",
    PRETRAINED_ZERO_SHOT: "T5-small zero-shot",
}

#: How the ROUGE variants are named on a figure.
ROUGE_LABELS: dict[str, str] = {
    "rouge1": "ROUGE-1",
    "rouge2": "ROUGE-2",
    "rougeL": "ROUGE-L",
}

#: One colour per family, reused across the figures so a family keeps its hue
#: from one image to the next. A figure whose entity is the run rather than the
#: family shades this colour per run, see :func:`run_styles`.
VARIANT_COLOURS: dict[str, str] = {
    SCRATCH: "#c1440e",
    PRETRAINED_FINE_TUNED: "#1f4e79",
    PRETRAINED_ZERO_SHOT: "#6b6b6b",
}

#: Markers cycled inside a family, so two runs of one family are told apart by
#: shape as well as by shade. Shade alone stops being readable past three runs.
RUN_MARKERS: tuple[str, ...] = ("o", "s", "^", "D")

#: How far the last run of a family is lightened from the family colour. Beyond
#: this the palest curve stops reading against a white background.
MAX_LIGHTENING = 0.55

#: One colour per ROUGE variant, for the comparison figure.
ROUGE_COLOURS: dict[str, str] = {
    "rouge1": "#4c72b0",
    "rouge2": "#dd8452",
    "rougeL": "#55a868",
}

#: Reminder printed on every figure that carries a proportion. Section 2.2 asks
#: for it wherever a percentage of the corpus appears, because the number is a
#: share of the 20 000 example working corpus and not of XSum.
CORPUS_NOTE = "100 % = 20 000 exemples du corpus de travail, non XSum complet"


def pyplot() -> Any:
    """Return the plotting module, configured for a headless run.

    matplotlib is imported here rather than at module level for the reason
    MLflow is in :mod:`src.tracking.client`: the import costs seconds, and a
    module that only reads records should not pay it. The drawing backend is
    fixed to ``Agg`` before ``pyplot`` is imported, so the command draws the
    same way whether or not the machine has a display.

    Returns:
        The ``matplotlib.pyplot`` module.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def lighten(colour: str, factor: float) -> tuple[float, float, float]:
    """Return a lighter shade of a colour.

    Args:
        colour: The colour to lighten, in any form matplotlib accepts.
        factor: How far to move towards white, between zero and one.

    Returns:
        The shade, as red, green and blue components.
    """
    from matplotlib.colors import to_rgb

    red, green, blue = to_rgb(colour)
    return (
        red + (1.0 - red) * factor,
        green + (1.0 - green) * factor,
        blue + (1.0 - blue) * factor,
    )


def run_styles(rows: Sequence[ExperimentRow]) -> dict[str, tuple[Any, str]]:
    """Assign one colour and one marker to every run.

    The loss figures draw one line per run, not one per family, so the family
    colour alone would render the four from scratch runs as one thick orange
    band with a legend nobody can use. The hue still says which family a curve
    belongs to; the shade and the marker say which run.

    Args:
        rows: Every declared experiment.

    Returns:
        A mapping from experiment name to its colour and marker. Runs are
        shaded in name order, so redrawing the same records twice produces the
        same image.
    """
    families: dict[str, list[str]] = {}
    for row in rows:
        families.setdefault(row.config.variant, []).append(row.name)

    styles: dict[str, tuple[Any, str]] = {}
    for variant, names in families.items():
        base = VARIANT_COLOURS.get(variant, "#333333")
        span = max(len(names) - 1, 1)
        for index, name in enumerate(sorted(names)):
            styles[name] = (
                lighten(base, MAX_LIGHTENING * index / span),
                RUN_MARKERS[index % len(RUN_MARKERS)],
            )
    return styles


def measured(rows: Sequence[ExperimentRow]) -> list[ExperimentRow]:
    """Keep the rows that carry a reportable score.

    Args:
        rows: The rows to filter.

    Returns:
        The complete runs, in the order given.
    """
    return [row for row in rows if row.measured]


def missing_note(rows: Sequence[ExperimentRow]) -> str | None:
    """Return the footnote naming the experiments a figure could not draw.

    Args:
        rows: Every row the figure was built from.

    Returns:
        The footnote, or ``None`` when every declared experiment is measured.
        The status travels with the name: a reader has to be able to tell an
        experiment that failed from one nobody ran.
    """
    absent = [f"{row.name} ({row.status})" for row in rows if not row.measured]
    if not absent:
        return None
    return "Sans mesure reportable : " + ", ".join(absent)


def annotate(figure: Any, note: str | None) -> None:
    """Write a footnote under a figure.

    Args:
        figure: The figure to write on.
        note: The text, or ``None`` to write nothing.
    """
    if note is None:
        return
    figure.text(0.01, 0.01, note, fontsize=7, color="#555555", ha="left", va="bottom")


def curve(
    rows: Sequence[ExperimentRow], variant: str
) -> tuple[list[int], list[float], list[list[float]]]:
    """Return the points of one family, ordered by corpus proportion.

    Args:
        rows: Every row of the corpus size study.
        variant: Family to extract, :data:`SCRATCH` or
            :data:`PRETRAINED_FINE_TUNED`.

    Returns:
        The proportions, the scores, and the two error bar lengths expected by
        matplotlib, which are distances from the point rather than bounds. A
        run whose record holds no interval contributes a zero length bar rather
        than being dropped: the score is measured, the interval is what is
        missing.
    """
    selected = [
        row
        for row in measured(rows)
        if row.config.variant == variant and row.config.dataset.percentage is not None
    ]
    selected.sort(key=lambda row: row.config.dataset.percentage or 0)

    percentages: list[int] = []
    scores: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    for row in selected:
        score = row.rouge(REPORTED_VARIANT)
        if score is None:
            continue
        percentages.append(int(row.config.dataset.percentage or 0))
        scores.append(score)
        bounds = row.interval(REPORTED_VARIANT)
        lower.append(score - bounds[0] if bounds else 0.0)
        upper.append(bounds[1] - score if bounds else 0.0)

    return percentages, scores, [lower, upper]


def zero_shot(rows: Sequence[ExperimentRow]) -> tuple[float, tuple[float, float] | None] | None:
    """Return the zero shot reference of the corpus size study.

    Args:
        rows: Every row of the study.

    Returns:
        The score and its interval, or ``None`` when the baseline was never
        measured.
    """
    for row in measured(rows):
        if row.config.variant != PRETRAINED_ZERO_SHOT:
            continue
        score = row.rouge(REPORTED_VARIANT)
        if score is not None:
            return score, row.interval(REPORTED_VARIANT)
    return None


def draw_performance(rows: Sequence[ExperimentRow], path: Path) -> Path:
    """Draw performance against the training corpus size.

    This is the figure section 18 calls the main one, and the deliverable the
    project is judged on: two curves, three proportions, one reference line.

    Args:
        rows: Every row of the corpus size study.
        path: Destination file.

    Returns:
        The path written.
    """
    plt = pyplot()
    figure, axes = plt.subplots(figsize=(9.0, 5.5))

    drawn = 0
    for variant in (PRETRAINED_FINE_TUNED, SCRATCH):
        percentages, scores, errors = curve(rows, variant)
        if not percentages:
            continue
        drawn += 1
        axes.errorbar(
            percentages,
            scores,
            yerr=errors,
            marker="o",
            markersize=6,
            capsize=4,
            linewidth=2,
            color=VARIANT_COLOURS[variant],
            label=VARIANT_LABELS[variant],
        )
        for percentage, score in zip(percentages, scores, strict=True):
            axes.annotate(
                f"{score:.4f}",
                (percentage, score),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=8,
                color=VARIANT_COLOURS[variant],
            )

    reference = zero_shot(rows)
    if reference is not None:
        drawn += 1
        score, bounds = reference
        axes.axhline(
            score,
            linestyle="--",
            linewidth=1.5,
            color=VARIANT_COLOURS[PRETRAINED_ZERO_SHOT],
            label=f"{VARIANT_LABELS[PRETRAINED_ZERO_SHOT]} ({score:.4f})",
        )
        if bounds is not None:
            axes.axhspan(
                bounds[0], bounds[1], color=VARIANT_COLOURS[PRETRAINED_ZERO_SHOT], alpha=0.12
            )

    axes.set_title("Performance selon la taille du corpus d'entrainement")
    axes.set_ylabel(f"{ROUGE_LABELS[REPORTED_VARIANT]} (F), IC 95 %")
    axes.grid(True, linestyle=":", alpha=0.5)

    if drawn:
        axes.set_xlabel(f"Part du corpus d'entrainement (%)\n{CORPUS_NOTE}")
        axes.set_xticks([10, 50, 100])
        axes.set_xticklabels(["10 %", "50 %", "100 %"])
        # Upper left is the only quadrant no curve crosses: the reference line
        # and its band sit low, and both curves climb from the left.
        axes.legend(loc="upper left", frameon=True)
    else:
        empty(axes, "Aucun run complet : rien a tracer.")

    annotate(figure, missing_note(rows))
    return save(figure, path)


def epoch_series(row: ExperimentRow, key: str) -> tuple[list[int], list[float]]:
    """Return one per epoch curve of a run.

    Args:
        row: The experiment row.
        key: Field to read from each epoch entry, ``train_loss`` or
            ``validation_loss``.

    Returns:
        The epoch numbers, counted from one, and the values. Empty for a run
        that never trained, which is what keeps the zero shot baseline off the
        loss figures.
    """
    record = row.record
    if record is None:
        return [], []

    entries: Any = (record.training or {}).get("epochs", [])
    if not isinstance(entries, list):
        return [], []

    epochs: list[int] = []
    values: list[float] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or key not in entry:
            continue
        epochs.append(int(entry.get("epoch", index)) + 1)
        values.append(float(entry[key]))
    return epochs, values


def draw_loss(
    rows: Sequence[ExperimentRow], path: Path, *, key: str, title: str, label: str
) -> Path:
    """Draw one loss curve per run that trained.

    Args:
        rows: Every declared experiment.
        path: Destination file.
        key: Field read from each epoch entry.
        title: Title of the figure.
        label: Label of the vertical axis.

    Returns:
        The path written.
    """
    plt = pyplot()
    figure, axes = plt.subplots(figsize=(9.0, 5.5))

    styles = run_styles(rows)
    drawn = 0
    for row in rows:
        epochs, values = epoch_series(row, key)
        if not epochs:
            continue
        colour, marker = styles[row.name]
        axes.plot(
            epochs,
            values,
            marker=marker,
            markersize=5,
            linewidth=1.6,
            color=colour,
            label=row.name,
        )
        drawn += 1

    axes.set_title(title)
    axes.set_xlabel("Epoque")
    axes.set_ylabel(label)
    axes.grid(True, linestyle=":", alpha=0.5)
    if drawn:
        axes.legend(fontsize=8, ncol=2, frameon=True)
    else:
        empty(axes, "Aucun run entraine : rien a tracer.")

    annotate(figure, missing_note(rows))
    return save(figure, path)


def draw_model_comparison(rows: Sequence[ExperimentRow], path: Path) -> Path:
    """Draw the three ROUGE variants of every measured run, side by side.

    Args:
        rows: Every declared experiment.
        path: Destination file.

    Returns:
        The path written.
    """
    plt = pyplot()
    figure, axes = plt.subplots(figsize=(11.0, 5.5))

    complete = measured(rows)
    if not complete:
        empty(axes, "Aucun run complet : rien a comparer.")
    else:
        width = 0.26
        positions = range(len(complete))
        for offset, variant in enumerate(ROUGE_VARIANTS):
            scores = [row.rouge(variant) or 0.0 for row in complete]
            errors_low: list[float] = []
            errors_high: list[float] = []
            for row, score in zip(complete, scores, strict=True):
                bounds = row.interval(variant)
                errors_low.append(score - bounds[0] if bounds else 0.0)
                errors_high.append(bounds[1] - score if bounds else 0.0)
            axes.bar(
                [position + (offset - 1) * width for position in positions],
                scores,
                width=width,
                yerr=[errors_low, errors_high],
                capsize=3,
                color=ROUGE_COLOURS[variant],
                label=ROUGE_LABELS[variant],
            )
        axes.set_xticks(list(positions))
        axes.set_xticklabels([row.name for row in complete], rotation=30, ha="right", fontsize=8)
        axes.legend(frameon=True)

    axes.set_title("ROUGE par experience, avec intervalles bootstrap a 95 %")
    axes.set_ylabel("F-mesure")
    axes.grid(True, axis="y", linestyle=":", alpha=0.5)

    annotate(figure, missing_note(rows))
    return save(figure, path)


def empty(axes: Any, message: str) -> None:
    """State on the image itself that there was nothing to draw.

    An empty pair of axes under a title reads as a measurement of zero. This
    writes the reason across the middle instead.

    Args:
        axes: The axes to write on.
        message: What to say.
    """
    axes.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, color="#555555")
    axes.set_xticks([])
    axes.set_yticks([])


def save(figure: Any, path: Path) -> Path:
    """Write one figure and release it.

    Args:
        figure: The figure to write.
        path: Destination file. Parent directories are created.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=DPI)
    pyplot().close(figure)
    return path


def build(
    *,
    experiments_dir: Path = DEFAULT_EXPERIMENTS_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    output_dir: Path = DEFAULT_FIGURES_DIR,
) -> list[Path]:
    """Draw the four figures of section 18.

    Args:
        experiments_dir: Directory holding the experiment files.
        results_dir: Directory holding the run directories.
        output_dir: Directory the images are written to.

    Returns:
        The paths written, in the order they were produced.

    Raises:
        ValueError: If the runs of the corpus size study, or the measured runs
            as a whole, were not scored the same way. The rule is the one the
            tables apply, and it matters more here: a curve makes a gap look
            like a result even harder than a column does.
    """
    rows = collect(experiments_dir, results_dir)
    dataset_size = rows_for_study(rows, "dataset_size")

    check_comparable(dataset_size, "dataset_size")
    check_comparable(rows, "model comparison")

    output = Path(output_dir)
    return [
        draw_performance(dataset_size, output / PERFORMANCE_FIGURE),
        draw_loss(
            rows,
            output / TRAINING_LOSS_FIGURE,
            key="train_loss",
            title="Perte d'entrainement par epoque",
            label="Perte d'entrainement",
        ),
        draw_loss(
            rows,
            output / VALIDATION_LOSS_FIGURE,
            key="validation_loss",
            title="Perte de validation par epoque",
            label="Perte de validation",
        ),
        draw_model_comparison(rows, output / MODEL_COMPARISON_FIGURE),
    ]


def build_argument_parser() -> argparse.ArgumentParser:
    """Describe the command line of ``python -m src.experiments.figures``.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments.figures",
        description=(
            "Draw the four figures of section 18 from the run records. Nothing is "
            "computed: an experiment with no complete run is named in a footnote "
            "rather than drawn."
        ),
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
        "--output",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Where the images go.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Draw the figures from the command line.

    Args:
        argv: Arguments to parse. Defaults to the process arguments.

    Returns:
        Process exit code. One when a figure was refused because its runs are
        not comparable, with the reason on the error stream.
    """
    args = build_argument_parser().parse_args(argv)

    try:
        written = build(
            experiments_dir=args.experiments,
            results_dir=args.results,
            output_dir=args.output,
        )
    except ValueError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1

    rows = collect(args.experiments, args.results)
    print(f"measured         {len(measured(rows))} of {len(rows)} declared experiments")
    note = missing_note(rows)
    if note is not None:
        print(f"footnote         {note}")
    for path in written:
        print(f"written          {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
