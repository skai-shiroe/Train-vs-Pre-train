"""The numbers of the documentation, emitted from the run records.

Section 38.4 asks for the synchronisation between the code and what is derived
from it to be checked mechanically. The pages were the one place where that did
not hold. Their tables were retyped from the run records by hand, so a new
campaign moved the records, the CSV tables and the figures, and left six pages
stating the values of the previous one. Nothing failed, and a wrong number is
worse than a missing one::

    python -m src.experiments.fragments

writes the Markdown fragments the documentation includes::

    docs/_generated/campaign.md         the provenance stamp
    docs/_generated/plan.md             every declared experiment
    docs/_generated/dataset_size.md     the corpus size ablation
    docs/_generated/architecture.md     the depth ablation
    docs/_generated/capitalisation.md   the lowercase rates
    docs/_generated/metrics.md          the three ROUGE variants, per run
    docs/_generated/pretrained.md       the four ``t5-small`` runs
    docs/_generated/families.md         the two families at each proportion
    docs/_generated/headline.md         the result table of a landing page

and refreshes the table ``README.md`` carries between its markers.

**A number is computed or narrated, never both.** The prose of the report is
the reading of the results and cannot be generated; it stays written by hand.
What it stops holding is the values. Every table it used to carry is now one
include, and :mod:`scripts.check_docs_sync` fails when a fragment on disk is
not what these records produce.

**The same table is included, not retyped, wherever it appears twice.** The
baseline page and the conformity page both showed the four ``t5-small`` runs,
and the landing page showed a subset of what the README showed. Two pages
retyping one measurement is two chances to state the previous campaign, so they
now share one fragment and the divergence has nowhere to live.

**The README is injected rather than included.** It is read on the forge, from
the repository root, where no MkDocs extension runs and a ``--8<--`` line would
render as itself. So the region between its markers is rewritten in place and
compared like a fragment. The markers are HTML comments: they say the table is
generated, and they render nowhere.

**The source is the run record, like the tables and the figures.** Same reader,
same comparability check, one rounding. A fragment built from
``experiments.csv`` would be a copy of a copy, rounded twice, and free to
disagree with the figure beside it.

**Only a complete run fills a cell.** A partial, failed or unrun experiment
keeps its row and leaves its score cells empty, and the status column appears
as soon as one row is in that state. A campaign that is not finished has to
read as unfinished, which is what section 44 asks of every derived artefact.

**The capitalisation table is measured here, not copied.** It is the only
number of the report that no run record carries, so it is recomputed from the
``predictions.jsonl`` each run wrote. That file is an artefact of the
evaluation, not a rerun of it: nothing is generated, decoded or scored again.

The strings are French because the fragments are read on the documentation
site, and they are accented, which is the one module of ``src`` that departs
from the plain ASCII the rest of the package holds to. A figure label carries
its own frame and reads as a caption whatever its spelling; these lines land in
the middle of the accented prose of the report, where an unaccented column
header reads as a defect of the page. The code around them stays English.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from src.evaluation.evaluator import PREDICTIONS_FILE
from src.experiments.ablation import check_comparable, sort_architecture, sort_dataset_size
from src.experiments.config import (
    DEFAULT_EXPERIMENTS_DIR,
    PRETRAINED_FINE_TUNED,
    PRETRAINED_ZERO_SHOT,
    SCRATCH,
    PretrainedModelConfig,
    ScratchModelConfig,
)
from src.experiments.record import run_directory
from src.experiments.registry import DEFAULT_RESULTS_DIR, ExperimentRow, collect, rows_for_study
from src.metrics.rouge import REPORTED_VARIANT, ROUGE_VARIANTS
from src.utils.markdown import (
    FRAGMENTS_DIR,
    MISSING,
    banner,
    fragment,
    number,
    share,
    stale,
    table,
    write,
)

#: What rewrites these fragments, for the banner they carry.
COMMAND = "python -m src.experiments.fragments"

#: Where the fragments go, shared with the generator of the corpus page.
DEFAULT_FRAGMENTS_DIR = FRAGMENTS_DIR

#: What the campaign as a whole is made of, for section 1 of the report.
CAMPAIGN_FRAGMENT = "campaign.md"

#: Performance against the training corpus size, for section 6.
DATASET_SIZE_FRAGMENT = "dataset_size.md"

#: Performance against the depth of the encoder decoder, for section 8.
ARCHITECTURE_FRAGMENT = "architecture.md"

#: Share of predictions starting on a lowercase character, for section 9.
CAPITALISATION_FRAGMENT = "capitalisation.md"

#: Every declared experiment and where it scored, for the plan of the
#: experiments page.
PLAN_FRAGMENT = "plan.md"

#: The three ROUGE variants of every run, for the evaluation page. The report
#: reads ROUGE-L alone; the page that describes the metric shows all three,
#: because the claim that fine tuning changes the format rather than the
#: language is read on ROUGE-2.
METRICS_FRAGMENT = "metrics.md"

#: The four ``t5-small`` runs, for the baseline page and requirement 2 of the
#: conformity page. One fragment, two includes.
PRETRAINED_FRAGMENT = "pretrained.md"

#: The two families side by side at each proportion, for requirement 3 of the
#: conformity page.
FAMILIES_FRAGMENT = "families.md"

#: The result table of a landing page, for the site index and the README.
HEADLINE_FRAGMENT = "headline.md"

#: The page that carries a generated table without being built by MkDocs.
DEFAULT_README = Path("README.md")

#: The name of the region the README reserves for the headline table.
HEADLINE_REGION = "headline"

#: How an injected region opens and closes. Named, so a page could carry a
#: second one, and so a mismatched pair is a parse error rather than a silently
#: swallowed page.
REGION_BEGIN = "<!-- syntra:begin {name} -->"
REGION_END = "<!-- syntra:end {name} -->"

#: How the ablation studies are named on a page. The experiment files carry the
#: keys, the reader gets the words.
STUDY_LABELS: dict[str, str] = {
    "dataset_size": "taille du corpus",
    "architecture": "architecture",
}

#: How a family is named in a table whose entity is the model rather than the
#: experiment. :mod:`src.experiments.figures` holds its own spelling of these:
#: a figure label is drawn by Matplotlib into an image and stays ASCII, these
#: land in accented prose.
VARIANT_LABELS: dict[str, str] = {
    SCRATCH: "from scratch",
    PRETRAINED_FINE_TUNED: "`t5-small` fine-tuné",
    PRETRAINED_ZERO_SHOT: "`t5-small` zero-shot",
}

#: How the two modes of the pre-trained baseline are named.
MODE_LABELS: dict[str, str] = {
    "zero_shot": "zero-shot",
    "fine_tuned": "fine-tuné",
}

#: How the ROUGE variants are named in a column header. Duplicated from
#: :mod:`src.experiments.figures` rather than imported: that module pulls
#: Matplotlib, and the freshness check has to run wherever the documentation is
#: built, which is not where the figures are drawn.
ROUGE_LABELS: dict[str, str] = {
    "rouge1": "ROUGE-1",
    "rouge2": "ROUGE-2",
    "rougeL": "ROUGE-L",
}

#: Written at the top of every fragment.
BANNER = banner(COMMAND)

#: Decimals kept for a ROUGE score. Four is what the metric is read to, and what
#: the figures annotate their points with.
SCORE_DECIMALS = 4

#: What fills a cell whose quantity does not apply. The zero shot baseline was
#: not trained on nothing, it was not trained at all, so its corpus cells say so
#: rather than showing a zero.
NOT_APPLICABLE = "sans objet"


def score(value: float | None) -> str:
    """Render one ROUGE score.

    Args:
        value: The F-measure, or ``None`` when the run holds none.

    Returns:
        The score with a decimal comma, or :data:`MISSING`.
    """
    if value is None:
        return MISSING
    return f"{value:.{SCORE_DECIMALS}f}".replace(".", ",")


def interval(bounds: tuple[float, float] | None) -> str:
    """Render one bootstrap interval.

    Args:
        bounds: The ``(low, high)`` pair, or ``None`` when the run holds none.

    Returns:
        The interval in brackets, or :data:`MISSING`. Reported even when the
        score is: a score without its interval is what lets a reader rank two
        models on two thousandths.
    """
    if bounds is None:
        return MISSING
    return f"[{score(bounds[0])}, {score(bounds[1])}]"


def seconds(value: float | None) -> str:
    """Render a duration in whole seconds.

    Args:
        value: The duration, or ``None`` when the run never trained.

    Returns:
        The rounded duration followed by its unit, or :data:`MISSING`.
    """
    if value is None:
        return MISSING
    return f"{number(round(float(value)))} s"


def needs_status(rows: Sequence[ExperimentRow]) -> bool:
    """Return whether a table has to carry a status column.

    Args:
        rows: The rows the table is built from.

    Returns:
        ``True`` as soon as one row holds no reportable score. A finished
        campaign renders without the column, an unfinished one cannot: the
        empty cells would be the only sign, and an empty cell is not a reason.
    """
    return any(not row.measured for row in rows)


def scored_cells(row: ExperimentRow) -> list[str]:
    """Return the score cells shared by every table.

    Args:
        row: The experiment row.

    Returns:
        The reported ROUGE variant and its bootstrap interval.
    """
    return [score(row.rouge(REPORTED_VARIANT)), interval(row.interval(REPORTED_VARIANT))]


def percentage_cell(row: ExperimentRow) -> str:
    """Render the training proportion of one experiment.

    Args:
        row: The experiment row.

    Returns:
        The proportion, or :data:`NOT_APPLICABLE` for the zero shot baseline,
        which was not trained on nothing but not trained at all.
    """
    percentage = row.config.dataset.percentage
    return f"{percentage} %" if percentage is not None else NOT_APPLICABLE


def emphasis(cell: str) -> str:
    """Mark a cell as the best result of its table.

    Args:
        cell: The rendered cell.

    Returns:
        The cell in bold, or unchanged when it holds nothing. Bold on an empty
        cell renders as four asterisks and reads as a defect of the page.
    """
    return f"**{cell}**" if cell else cell


def best_name(rows: Sequence[ExperimentRow]) -> str | None:
    """Return the experiment holding the highest reported score.

    Args:
        rows: The rows of the table.

    Returns:
        Its name, or ``None`` when nothing was measured. Ties are broken on the
        name so that two equal scores do not swap the emphasis between runs.
    """
    scored = [
        (value, row.name) for row in rows if (value := row.rouge(REPORTED_VARIANT)) is not None
    ]
    return max(scored)[1] if scored else None


def by_score(rows: Sequence[ExperimentRow]) -> list[ExperimentRow]:
    """Order rows from the best reported score to the worst.

    Args:
        rows: The rows to order.

    Returns:
        The measured rows by descending score, then the unmeasured ones by
        name. A result table is read as a ranking, and an experiment that did
        not run has no rank; it keeps its row at the bottom with its status.
    """

    def key(row: ExperimentRow) -> tuple[int, float, str]:
        value = row.rouge(REPORTED_VARIANT)
        return (1, 0.0, row.name) if value is None else (0, -value, row.name)

    return sorted(rows, key=key)


def pretrained_rows(rows: Sequence[ExperimentRow]) -> list[ExperimentRow]:
    """Keep the experiments running the pre-trained baseline.

    Args:
        rows: Every declared experiment.

    Returns:
        The subset whose model is ``t5-small``, ordered from the untouched
        weights to the full corpus. The zero shot run comes first here and last
        in the ablation tables: there it is a horizontal line under a curve,
        here it is the starting point the fine tuning moves away from.
    """

    def key(row: ExperimentRow) -> tuple[int, int, str]:
        percentage = row.config.dataset.percentage
        return (0 if percentage is None else 1, percentage or 0, row.name)

    pretrained = [row for row in rows if isinstance(row.config.model, PretrainedModelConfig)]
    return sorted(pretrained, key=key)


def relative(base: float | None, other: float | None) -> str:
    """Render how far one score sits above another, as a percentage.

    Args:
        base: The score compared against, from scratch in the families table.
        other: The score compared.

    Returns:
        The signed relative gap in whole percent, or :data:`MISSING` when
        either side is absent. The sign is always written: a table column named
        for a gap must not let a regression read as a gain.
    """
    if base is None or other is None or base <= 0:
        return MISSING
    return f"{100.0 * (other - base) / base:+.0f} %"


def plan_table(rows: Sequence[ExperimentRow]) -> str:
    """Render every declared experiment and what it scored.

    This one carries the status column unconditionally, unlike the ablation
    tables. It is the table that answers how much of the plan is done, so the
    column is the point of it rather than a sign that something is missing.

    Args:
        rows: Every declared experiment, already ordered.

    Returns:
        The Markdown table.
    """
    columns = ["Expérience", "Variante", "Corpus", "Études", "Statut", "ROUGE-L"]
    body = [
        [
            f"`{row.name}`",
            f"`{row.config.variant}`",
            percentage_cell(row),
            ", ".join(STUDY_LABELS.get(study, study) for study in row.config.experiment.studies)
            or NOT_APPLICABLE,
            f"`{row.status}`",
            score(row.rouge(REPORTED_VARIANT)),
        ]
        for row in rows
    ]
    return table(columns, body)


def dataset_size_table(rows: Sequence[ExperimentRow]) -> str:
    """Render the corpus size ablation of section 6.

    Args:
        rows: The rows of the corpus size study, already ordered.

    Returns:
        The Markdown table.
    """
    status = needs_status(rows)
    columns = ["Variante", "Corpus", "Exemples", "ROUGE-L", "IC 95 %"]
    if status:
        columns.append("Statut")

    body: list[list[str]] = []
    for row in rows:
        percentage = row.config.dataset.percentage
        trained = percentage is not None
        dataset: dict[str, Any] = row.record.dataset if row.record else {}
        cells = [
            f"`{row.config.variant}`",
            percentage_cell(row),
            number(dataset.get("train_examples")) if trained else NOT_APPLICABLE,
            *scored_cells(row),
        ]
        if status:
            cells.append(f"`{row.status}`")
        body.append(cells)

    return table(columns, body)


def architecture_table(rows: Sequence[ExperimentRow]) -> str:
    """Render the architecture ablation of section 8.

    Args:
        rows: The rows of the architecture study, already ordered.

    Returns:
        The Markdown table. The training time sits beside the score because the
        result of the study is that the deeper models cost more and score no
        better; without the cost column the table only shows the second half.
    """
    status = needs_status(rows)
    columns = ["Expérience", "Couches", "Paramètres", "ROUGE-L", "IC 95 %", "Entraînement"]
    if status:
        columns.append("Statut")

    body: list[list[str]] = []
    for row in rows:
        model = row.config.model
        depth = (
            f"{model.encoder_layers} + {model.decoder_layers}"
            if isinstance(model, ScratchModelConfig)
            else MISSING
        )
        training: dict[str, Any] = (row.record.training or {}) if row.record else {}
        cells = [
            f"`{row.name}`",
            depth,
            number(row.record.model.get("parameters") if row.record else None),
            *scored_cells(row),
            seconds(training.get("duration_seconds")),
        ]
        if status:
            cells.append(f"`{row.status}`")
        body.append(cells)

    return table(columns, body)


def metrics_table(rows: Sequence[ExperimentRow]) -> str:
    """Render the three ROUGE variants of every run, for the evaluation page.

    Args:
        rows: Every declared experiment, already ordered.

    Returns:
        The Markdown table. The interval is on the reported variant only: the
        evaluator bootstraps all three, but a row carrying three intervals is
        read as three independent results rather than one score with its
        uncertainty and two figures supporting it.
    """
    status = needs_status(rows)
    reported = ROUGE_LABELS[REPORTED_VARIANT]
    columns = ["Expérience", *(ROUGE_LABELS[variant] for variant in ROUGE_VARIANTS)]
    columns.append(f"IC 95 % sur {reported}")
    if status:
        columns.append("Statut")

    best = best_name(rows)
    body: list[list[str]] = []
    for row in rows:
        cells = [f"`{row.name}`"]
        for variant in ROUGE_VARIANTS:
            cell = score(row.rouge(variant))
            highlight = variant == REPORTED_VARIANT and row.name == best
            cells.append(emphasis(cell) if highlight else cell)
        cells.append(interval(row.interval(REPORTED_VARIANT)))
        if status:
            cells.append(f"`{row.status}`")
        body.append(cells)

    return table(columns, body)


def pretrained_table(rows: Sequence[ExperimentRow]) -> str:
    """Render the four ``t5-small`` runs, for the baseline and conformity pages.

    Args:
        rows: The pre-trained experiments, already ordered.

    Returns:
        The Markdown table. The mode column is the point of it: requirement 2
        asks for zero shot and fine tuned as two distinct experiments, and a
        table of four scores without the mode does not show that they are.
    """
    status = needs_status(rows)
    columns = ["Expérience", "Mode", "Corpus", ROUGE_LABELS[REPORTED_VARIANT], "IC 95 %"]
    if status:
        columns.append("Statut")

    body: list[list[str]] = []
    for row in rows:
        model = row.config.model
        mode = MODE_LABELS[model.mode] if isinstance(model, PretrainedModelConfig) else MISSING
        cells = [f"`{row.name}`", mode, percentage_cell(row), *scored_cells(row)]
        if status:
            cells.append(f"`{row.status}`")
        body.append(cells)

    return table(columns, body)


def gap_reason(label: str, row: ExperimentRow | None) -> str:
    """Say why one side of a family comparison carries no score.

    Args:
        label: How the family is named in the column that stayed empty.
        row: The experiment of that family at that proportion, or ``None`` when
            no experiment declares it.

    Returns:
        The reason, or an empty string when the cell holds a measurement. A
        proportion that only one family declares is a different absence from a
        run that failed, and the table has to distinguish them: the first is a
        hole in the plan, the second a hole in the campaign.
    """
    if row is None:
        return f"{label} : non déclarée"
    return "" if row.measured else f"`{row.name}` `{row.status}`"


def families_table(rows: Sequence[ExperimentRow]) -> str:
    """Render the two families side by side at each proportion.

    This is the table requirement 3 is answered with, and the relative gap is
    the answer: it does not close as the corpus grows.

    Args:
        rows: The rows of the corpus size study.

    Returns:
        The Markdown table, or a sentence saying no proportion is declared. The
        zero shot baseline is absent by construction: its weights do not depend
        on the size of the training corpus, so it is one measurement and not
        one per proportion, and a row repeating it three times would read as
        three.
    """
    scratch_label, pretrained_label = (
        VARIANT_LABELS[SCRATCH],
        VARIANT_LABELS[PRETRAINED_FINE_TUNED],
    )
    declared = {
        (row.config.variant, row.config.dataset.percentage): row
        for row in rows
        if row.config.dataset.percentage is not None
        and row.config.variant in (SCRATCH, PRETRAINED_FINE_TUNED)
    }
    proportions = sorted({percentage for _, percentage in declared})
    if not proportions:
        return "Aucune proportion n'est déclarée dans les deux familles : la table est vide."

    columns = ["Proportion", scratch_label, pretrained_label, "Écart relatif"]
    pairs = [
        (
            percentage,
            declared.get((SCRATCH, percentage)),
            declared.get((PRETRAINED_FINE_TUNED, percentage)),
        )
        for percentage in proportions
    ]
    reasons = [
        ", ".join(
            reason
            for reason in (
                gap_reason(scratch_label, scratch),
                gap_reason(pretrained_label, trained),
            )
            if reason
        )
        for _, scratch, trained in pairs
    ]
    status = any(reasons)
    if status:
        columns.append("Statut")

    body: list[list[str]] = []
    for (percentage, scratch, trained), reason in zip(pairs, reasons, strict=True):
        base = scratch.rouge(REPORTED_VARIANT) if scratch else None
        other = trained.rouge(REPORTED_VARIANT) if trained else None
        cells = [f"{percentage} %", score(base), score(other), relative(base, other)]
        if status:
            cells.append(reason)
        body.append(cells)

    return table(columns, body)


def headline_table(rows: Sequence[ExperimentRow]) -> str:
    """Render the result table a landing page carries.

    Args:
        rows: The rows of the corpus size study, already ordered.

    Returns:
        The Markdown table. Its entity is the model rather than the experiment:
        a reader arriving on the index or the README has not met the
        experiment names yet, and ``pretrained_ft_100`` answers nothing they
        came to ask.
    """
    status = needs_status(rows)
    columns = ["Modèle", "Corpus", ROUGE_LABELS[REPORTED_VARIANT], "IC 95 %"]
    if status:
        columns.append("Statut")

    best = best_name(rows)
    body: list[list[str]] = []
    for row in rows:
        cell = score(row.rouge(REPORTED_VARIANT))
        cells = [
            VARIANT_LABELS[row.config.variant],
            percentage_cell(row),
            emphasis(cell) if row.name == best else cell,
            interval(row.interval(REPORTED_VARIANT)),
        ]
        if status:
            cells.append(f"`{row.status}`")
        body.append(cells)

    return table(columns, body)


def injected(path: Path, name: str, body: str) -> str:
    """Return a hand written page with its marked region rewritten.

    Args:
        path: The page to refresh.
        name: The region name its markers carry.
        body: The Markdown to put between them.

    Returns:
        The whole page, so that it is compared and written like a fragment. The
        prose around the markers is read from disk and kept: this rewrites one
        region, it does not generate a page.

    Raises:
        ValueError: If the page is missing, or carries no such region. A
            generated table that silently found nowhere to go would leave the
            page stating the previous campaign, which is what the markers are
            there to prevent.
    """
    if not path.is_file():
        raise ValueError(f"{path} does not exist, so its `{name}` table has nowhere to go.")

    text = path.read_text(encoding="utf-8")
    begin, end = REGION_BEGIN.format(name=name), REGION_END.format(name=name)
    start, stop = text.find(begin), text.find(end)
    if start < 0 or stop < start:
        raise ValueError(
            f"{path} carries no `{name}` region: it must hold {begin} then {end}. "
            "A generated table is only as safe as the marker saying where it belongs."
        )
    return f"{text[: start + len(begin)]}\n{fragment(body, COMMAND)}{text[stop:]}"


def capitalisation(path: Path) -> tuple[int, int, int] | None:
    """Count the summaries of one run that start on a lowercase character.

    Args:
        path: The ``predictions.jsonl`` an evaluation wrote.

    Returns:
        The lowercase predictions, the lowercase references and the total, or
        ``None`` when the run wrote no predictions file. An empty prediction is
        counted in the total and not as lowercase: it is a different defect,
        and the record reports it separately.
    """
    if not path.is_file():
        return None

    predictions = 0
    references = 0
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        total += 1
        predictions += 1 if str(entry.get("prediction", ""))[:1].islower() else 0
        references += 1 if str(entry.get("reference", ""))[:1].islower() else 0
    return predictions, references, total


def capitalisation_table(rows: Sequence[ExperimentRow], results_dir: Path) -> str:
    """Render the capitalisation table of section 9.

    The reference row is the control. Without it the prediction rates are three
    numbers with nothing to be high against, and the claim that the convention
    is learned rather than mangled by the decoding has nothing behind it.

    Args:
        rows: Every declared experiment.
        results_dir: Directory the run directories live under.

    Returns:
        The Markdown table, or a sentence saying no run wrote predictions. The
        rows are ordered the way the corpus size ablation is, by family then by
        proportion, because the point of the table is that the rate falls as
        the corpus grows. Sorted by name, ``scratch_100`` would sit between
        ``scratch_10`` and ``scratch_50`` and hide it.
    """
    columns = ["Modèle", "Prédictions commençant par une minuscule"]
    body: list[list[str]] = []
    reference_rate = MISSING

    for row in sort_dataset_size(rows):
        if not row.measured:
            continue
        counted = capitalisation(run_directory(results_dir, row.name) / PREDICTIONS_FILE)
        if counted is None:
            continue
        predictions, references, total = counted
        body.append([f"`{row.name}`", share(predictions, total)])
        # Every run of a table is scored on the same split, which is what
        # check_comparable enforces, so the reference rate is one number and
        # not one per row.
        reference_rate = share(references, total)

    if not body:
        return "Aucun run complet n'a écrit de prédictions : la table est vide."

    body.append(["Références", reference_rate])
    return table(columns, body)


def distinct(values: Iterable[str]) -> list[str]:
    """Return the distinct values of a sequence, in first seen order.

    Args:
        values: The values to reduce.

    Returns:
        The values without repetition. Order is kept rather than sorted so the
        output follows the experiment order and stays stable between runs.
    """
    seen: dict[str, None] = {}
    for value in values:
        if value:
            seen.setdefault(value, None)
    return list(seen)


def statuses(rows: Sequence[ExperimentRow]) -> str:
    """Render how many experiments are in each status.

    Args:
        rows: Every declared experiment.

    Returns:
        One entry per status present, most frequent first, then by name so two
        statuses of equal count keep a stable order.
    """
    counted: dict[str, int] = {}
    for row in rows:
        counted[row.status] = counted.get(row.status, 0) + 1
    ordered = sorted(counted.items(), key=lambda entry: (-entry[1], entry[0]))
    return ", ".join(f"`{name}` {count}" for name, count in ordered)


def campaign_table(rows: Sequence[ExperimentRow]) -> str:
    """Render what the campaign as a whole is made of, for section 1.

    This is the provenance stamp. The scores of the report are only worth what
    the reader knows about where they come from, and the commit is part of that
    answer even when, especially when, it is ``unknown``.

    Nothing here describes where the records were read from. A fragment that
    named its own input directory would render differently depending on whether
    the command was given a relative or an absolute path, which is a generated
    file that disagrees with itself for a reason no reader could see.

    Args:
        rows: Every declared experiment.

    Returns:
        The Markdown table.
    """
    measured = [row for row in rows if row.measured]
    records = [row.record for row in measured if row.record is not None]

    total_seconds = sum(record.duration_seconds for record in records)
    corpora = distinct(str(record.dataset.get("version", "")) for record in records)
    commits = distinct(record.provenance.get("git_commit", "") for record in records)
    hardware = distinct(record.hardware.get("gpu_name", "") for record in records)
    frameworks = distinct(record.hardware.get("torch_version", "") for record in records)

    body = [
        ["Expériences déclarées", number(len(rows))],
        ["Runs complets", number(len(measured))],
        ["Statuts", statuses(rows)],
        ["Calcul cumulé", f"{number(round(total_seconds / 60))} minutes"],
        ["Empreinte du corpus", ", ".join(f"`{value}`" for value in corpora) or MISSING],
        ["Commit des runs", ", ".join(f"`{value}`" for value in commits) or MISSING],
        ["Matériel", ", ".join(hardware) or MISSING],
        ["Torch", ", ".join(f"`{value}`" for value in frameworks) or MISSING],
    ]
    return table(["Ce que porte la campagne", "Valeur"], body)


def build(
    *,
    experiments_dir: Path = DEFAULT_EXPERIMENTS_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    output_dir: Path = DEFAULT_FRAGMENTS_DIR,
    readme: Path,
) -> dict[Path, str]:
    """Render every generated table of the documentation.

    Args:
        experiments_dir: Directory holding the experiment files.
        results_dir: Directory holding the run directories.
        output_dir: Directory the fragments belong in. Used to key the result;
            nothing is written here.
        readme: The page carrying a generated table between markers instead of
            an include, because MkDocs does not build it. Required, and
            deliberately without a default: it is the one target that is a
            tracked, hand written page rather than a generated directory, and a
            caller that forgot it would rewrite the README of the repository
            with whatever campaign it was pointed at. The command line supplies
            :data:`DEFAULT_README`, a test supplies its own.

    Returns:
        A mapping from destination path to the text that belongs in it.
        Rendering and writing are split so the freshness check can compare
        without producing a file, and so a refused study never leaves half the
        tables rewritten.

    Raises:
        ValueError: If the runs of a study were not measured the same way, or
            if the README carries no region to inject into. The comparability
            check the tables and the figures apply is applied here too, and by
            the same function.
    """
    rows = collect(experiments_dir, results_dir)
    results = Path(results_dir)
    output = Path(output_dir)

    dataset_size = rows_for_study(rows, "dataset_size")
    architecture = rows_for_study(rows, "architecture")
    check_comparable(dataset_size, "dataset_size")
    check_comparable(architecture, "architecture")

    # Rendered once and placed twice: the site index includes it, the README
    # has it injected. Two renderings could not disagree, but two call sites
    # could drift in what they are given, which is the same defect one step up.
    headline = headline_table(by_score(dataset_size))

    return {
        output / CAMPAIGN_FRAGMENT: fragment(campaign_table(rows), COMMAND),
        output / PLAN_FRAGMENT: fragment(plan_table(sort_dataset_size(rows)), COMMAND),
        output
        / DATASET_SIZE_FRAGMENT: fragment(
            dataset_size_table(sort_dataset_size(dataset_size)), COMMAND
        ),
        output
        / ARCHITECTURE_FRAGMENT: fragment(
            architecture_table(sort_architecture(architecture)), COMMAND
        ),
        output / CAPITALISATION_FRAGMENT: fragment(capitalisation_table(rows, results), COMMAND),
        output / METRICS_FRAGMENT: fragment(metrics_table(by_score(rows)), COMMAND),
        output / PRETRAINED_FRAGMENT: fragment(pretrained_table(pretrained_rows(rows)), COMMAND),
        output / FAMILIES_FRAGMENT: fragment(families_table(dataset_size), COMMAND),
        output / HEADLINE_FRAGMENT: fragment(headline, COMMAND),
        Path(readme): injected(Path(readme), HEADLINE_REGION, headline),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Describe the command line of ``python -m src.experiments.fragments``.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments.fragments",
        description=(
            "Emit the Markdown fragments the documentation includes, from the run records, "
            "and refresh the table the README carries between its markers. Nothing is "
            "computed except the capitalisation rate, which is counted from the predictions "
            "each evaluation wrote."
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
        default=DEFAULT_FRAGMENTS_DIR,
        help="Where the fragments go.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=DEFAULT_README,
        help="The page whose marked region carries the headline table.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report what is out of date and write nothing. Exits non zero on a stale fragment.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Emit or check the fragments from the command line.

    Args:
        argv: Arguments to parse. Defaults to the process arguments.

    Returns:
        Process exit code. One when a study was refused because its runs are
        not comparable, and one under ``--check`` when a fragment is stale.
    """
    args = build_argument_parser().parse_args(argv)

    try:
        rendered = build(
            experiments_dir=args.experiments,
            results_dir=args.results,
            output_dir=args.output,
            readme=args.readme,
        )
    except ValueError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1

    if args.check:
        outdated = stale(rendered)
        for path in outdated:
            print(f"stale            {path}")
        if outdated:
            print(
                f"\n{len(outdated)} generated table(s) out of date, run: make report-sync",
                file=sys.stderr,
            )
            return 1
        print(f"up to date       {len(rendered)} generated files")
        return 0

    for path in write(rendered):
        print(f"written          {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
