"""The numbers of the corpus, emitted from the corpus record.

Section 38.4 asks for the synchronisation between the code and what is derived
from it to be checked mechanically. :mod:`src.experiments.fragments` did that
for the tables the campaign produces, and left the corpus out. Its
distributions, its truncation counts and its checksums were retyped from
``statistics.json`` and ``manifest.json`` by hand, so rebuilding the corpus
moved the record and left the tables describing the previous build::

    python -m src.data.fragments

writes the Markdown tables of the corpus::

    reports/_generated/checksums.md    the fingerprints of the built corpus
    reports/_generated/statistics.md   the measured distributions, per split
    reports/_generated/truncation.md   what the token ceilings cut off

and refreshes the distributions ``RAPPORT.md`` carries between its markers, in
the section that argues the corpus from them. The injection mechanism is the one
:mod:`src.experiments.fragments` documents, and the report carries the regions of
both generators: they never touch the same one.

**The record is what ``make data`` wrote, not a recomputation.** Nothing here
opens a JSONL file, counts a word or loads a tokenizer. ``python -m
src.data.build`` measured the corpus once and wrote what it measured beside it;
this reads that file. A table rebuilt by remeasuring would be a second
measurement free to disagree with the first, which is the defect one level up
from the one being fixed.

**The notebook is a reader, like this module.** ``notebooks/01_eda_xsum.ipynb``
reads the same two files rather than recomputing them, and its outputs are
stripped at every commit. So these tables cannot take their numbers from the
notebook: they take them from what the notebook itself reads.

**Nothing here names a ceiling.** The truncation table counts what was cut and
does not say at how many tokens, because the record holds the counts and not the
configuration they were measured under. Reading ``configs/data/xsum.yaml`` for
the figure would pair today's setting with yesterday's counts and render a
sentence no file on disk supports. The ceilings stay in the prose above the
table, where they are a decision rather than a measurement.

This module does not import the rest of :mod:`src.data`. Reaching the two
filenames through :mod:`src.data.dataset` would pull Torch and Transformers into
the freshness check, which has to run wherever the repository is checked out,
and that is not where a corpus is prepared. Two filenames are copied instead.

The strings are French because the tables are read beside the French report,
and accented for the same reason :mod:`src.experiments.fragments` gives: they
land in the middle of accented prose. The code around them stays English.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.markdown import (
    FRAGMENTS_DIR,
    MISSING,
    decimal,
    fragment,
    inject_regions,
    number,
    share,
    stale,
    table,
    write,
)

#: What rewrites these fragments, for the banner they carry.
COMMAND = "python -m src.data.fragments"

#: Where the built corpus and its record live. The value ``configs/data/xsum.yaml``
#: gives to ``paths.processed``.
DEFAULT_PROCESSED_DIR = Path("data") / "processed" / "xsum"

#: Where the fragments go, shared with the generator of the campaign tables.
DEFAULT_FRAGMENTS_DIR = FRAGMENTS_DIR

#: The two files ``python -m src.data.build`` writes beside the corpus. Copied
#: from :mod:`src.data.dataset` and :mod:`src.data.build` rather than imported,
#: for the reason the module docstring gives.
MANIFEST_NAME = "manifest.json"
STATISTICS_NAME = "statistics.json"

#: The fingerprints of the corpus currently built.
CHECKSUMS_FRAGMENT = "checksums.md"

#: The measured distributions, per split.
STATISTICS_FRAGMENT = "statistics.md"

#: What the token ceilings cut off.
TRUNCATION_FRAGMENT = "truncation.md"

#: The report, which carries the measured distributions between markers. Section
#: 1 argues the corpus from them, and a table describing the previous build would
#: put the argument on a corpus nobody trained on.
DEFAULT_REPORT = Path("RAPPORT.md")

#: The name of the region the report reserves for the statistics table.
STATISTICS_REGION = "statistics"

#: The splits, in the order every table shows them. Train first
#: because it is the one the ablations resize; test last because it is the one
#: the scores are read on.
SPLIT_NAMES = ("train", "validation", "test")

#: How a split is named in a column header.
SPLIT_LABELS: dict[str, str] = {
    "train": "Entraînement",
    "validation": "Validation",
    "test": "Test",
}

#: How the fingerprint of the three splits taken together is named. It is the
#: value tracked with every experiment, so it is written as the identifier it is
#: rather than translated.
VERSION_LABEL = "`dataset_version`"

#: Decimals kept for a length. One, which is what the tables showed when the
#: values were typed by hand, and one more than the quantity is read to: a
#: median of thirty tokens is not measured to a hundredth.
LENGTH_DECIMALS = 1

#: Decimals kept for the compression ratio. It sits around a tenth, so one
#: decimal would render three splits as the same number.
RATIO_DECIMALS = 3


@dataclass(frozen=True, slots=True)
class Quantity:
    """One row of the statistics table.

    Attributes:
        label: How the row is named in the first column.
        section: Top level key of the record holding it.
        series: Distribution it belongs to, or ``None`` when the value sits
            directly on the split, which is how the compression ratio is stored.
        statistic: Key of the value inside its series.
        decimals: Decimals to render it with.
    """

    label: str
    section: str
    series: str | None
    statistic: str
    decimals: int


#: The rows of the statistics table, in the order they are shown: what a reader
#: counts in words first, then what the model actually consumes, then the ratio
#: the two make. Only the quantities the report argues from are here; the record
#: holds minima and maxima that no decision reads.
QUANTITIES: tuple[Quantity, ...] = (
    Quantity(
        "Mots par document, médiane",
        "characters_and_words",
        "source_words",
        "median",
        LENGTH_DECIMALS,
    ),
    Quantity(
        "Mots par résumé, médiane",
        "characters_and_words",
        "target_words",
        "median",
        LENGTH_DECIMALS,
    ),
    Quantity("Tokens par document, moyenne", "tokens", "source_tokens", "mean", LENGTH_DECIMALS),
    Quantity("Tokens par document, médiane", "tokens", "source_tokens", "median", LENGTH_DECIMALS),
    Quantity("Tokens par document, p95", "tokens", "source_tokens", "p95", LENGTH_DECIMALS),
    Quantity("Tokens par résumé, médiane", "tokens", "target_tokens", "median", LENGTH_DECIMALS),
    Quantity("Tokens par résumé, p95", "tokens", "target_tokens", "p95", LENGTH_DECIMALS),
    Quantity(
        "Taux de compression",
        "characters_and_words",
        None,
        "compression_ratio",
        RATIO_DECIMALS,
    ),
)


def read_record(path: Path) -> dict[str, Any]:
    """Read one JSON record written beside the corpus.

    Args:
        path: The file to read.

    Returns:
        Its content as a mapping.

    Raises:
        ValueError: If the file is missing, unreadable or not a JSON object. A
            generator that treated an absent record as an empty one would render
            a table of empty cells and report success, which section 44 forbids
            more plainly than it forbids a wrong number.
    """
    if not path.is_file():
        raise ValueError(f"{path} does not exist, run: make data")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not readable JSON: {error}") from error

    if not isinstance(payload, dict):
        raise ValueError(f"{path} holds a {type(payload).__name__}, expected an object.")
    return payload


def split_record(statistics: dict[str, Any], section: str, split: str) -> dict[str, Any]:
    """Return the part of a record describing one split.

    Args:
        statistics: The corpus statistics.
        section: Top level key to read under.
        split: Name of the split.

    Returns:
        The mapping for that split, empty when the record holds none. A split
        the record never described renders as empty cells rather than raising:
        the table says what was measured, and a missing split is one of the
        things it can say.
    """
    section_record = statistics.get(section)
    if not isinstance(section_record, dict):
        return {}
    split_payload = section_record.get(split)
    return split_payload if isinstance(split_payload, dict) else {}


def measured(statistics: dict[str, Any], split: str, quantity: Quantity) -> float | None:
    """Read one measured value out of the corpus statistics.

    Args:
        statistics: The corpus statistics.
        split: Name of the split.
        quantity: What to read.

    Returns:
        The value, or ``None`` when the record does not hold it.
    """
    holder = split_record(statistics, quantity.section, split)
    if quantity.series is not None:
        series = holder.get(quantity.series)
        holder = series if isinstance(series, dict) else {}

    value = holder.get(quantity.statistic)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def split_size(statistics: dict[str, Any], split: str) -> int | None:
    """Return how many examples one split holds.

    Args:
        statistics: The corpus statistics.
        split: Name of the split.

    Returns:
        The count, or ``None`` when the record does not hold it. Read from the
        statistics rather than from the manifest so that a percentage and the
        count it divides come from one file and cannot disagree.
    """
    size = split_record(statistics, "characters_and_words", split).get("size")
    return int(size) if isinstance(size, int) and not isinstance(size, bool) else None


def truncated(statistics: dict[str, Any], split: str, key: str) -> int | None:
    """Return how many examples of one split were cut by a token ceiling.

    Args:
        statistics: The corpus statistics.
        split: Name of the split.
        key: Which count to read, sources or targets.

    Returns:
        The count, or ``None`` when the record does not hold it.
    """
    value = split_record(statistics, "tokens", split).get(key)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def cut_cell(statistics: dict[str, Any], split: str, key: str) -> str:
    """Render how much of one split a ceiling cut.

    Args:
        statistics: The corpus statistics.
        split: Name of the split.
        key: Which count to read, sources or targets.

    Returns:
        The count out of the split size with its percentage, or :data:`MISSING`
        when either is absent. The count is kept beside the percentage: a share
        alone hides how many examples it stands for, and the two splits of a
        thousand examples move by a tenth of a point per example.
    """
    count = truncated(statistics, split, key)
    size = split_size(statistics, split)
    if count is None or size is None or size <= 0:
        return MISSING
    return f"{number(count)} sur {number(size)} ({share(count, size)})"


def code(value: object) -> str:
    """Render a value as inline code.

    Args:
        value: The value, or ``None`` when the record holds none.

    Returns:
        The value between backticks, or :data:`MISSING`. A checksum is read
        character by character when it is compared, so it is never rendered as
        prose.
    """
    return f"`{value}`" if value else MISSING


def checksums_table(manifest: dict[str, Any]) -> str:
    """Render the fingerprints of the corpus currently built.

    Args:
        manifest: The corpus manifest.

    Returns:
        The Markdown table. The version of the three splits taken together comes
        first because it is the value tracked with every experiment; the per
        split hashes follow because they are what localises a change the version
        only signals.
    """
    columns = ["Empreinte", "SHA-256"]
    checksums = manifest.get("split_checksums")
    checksums = checksums if isinstance(checksums, dict) else {}

    body = [[VERSION_LABEL, code(manifest.get("dataset_version"))]]
    body.extend([SPLIT_LABELS[split], code(checksums.get(split))] for split in SPLIT_NAMES)
    return table(columns, body)


def statistics_table(statistics: dict[str, Any]) -> str:
    """Render the measured distributions of the working corpus.

    Args:
        statistics: The corpus statistics.

    Returns:
        The Markdown table, one row per quantity and one column per split.
    """
    columns = ["Grandeur", *(SPLIT_LABELS[split] for split in SPLIT_NAMES)]
    body = [
        [
            quantity.label,
            *(
                decimal(measured(statistics, split, quantity), quantity.decimals)
                for split in SPLIT_NAMES
            ),
        ]
        for quantity in QUANTITIES
    ]
    return table(columns, body)


def truncation_table(statistics: dict[str, Any]) -> str:
    """Render what the token ceilings cut off, per split.

    The targets sit beside the sources because the report argues that one ceiling
    is expensive and the other is essentially free. With the source column
    alone, half of that argument rests on a number no table carries.

    Args:
        statistics: The corpus statistics.

    Returns:
        The Markdown table.
    """
    columns = ["Split", "Documents tronqués", "Résumés tronqués"]
    body = [
        [
            SPLIT_LABELS[split],
            cut_cell(statistics, split, "truncated_sources"),
            cut_cell(statistics, split, "truncated_targets"),
        ]
        for split in SPLIT_NAMES
    ]
    return table(columns, body)


def build(
    *,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    output_dir: Path = DEFAULT_FRAGMENTS_DIR,
    report: Path,
) -> dict[Path, str]:
    """Render every generated table of the corpus.

    Args:
        processed_dir: Directory holding the built corpus and its record.
        output_dir: Directory the fragments belong in. Used to key the result;
            nothing is written here.
        report: The page whose marked region carries the statistics table.
            Required, and deliberately without a default: it is a tracked, hand
            written page rather than a generated directory, and a caller that
            forgot it would rewrite the report of the repository with whatever
            corpus it was pointed at. The command line supplies
            :data:`DEFAULT_REPORT`, a test supplies its own.

    Returns:
        A mapping from destination path to the text that belongs in it.
        Rendering and writing are split so the freshness check can compare
        without producing a file.

    Raises:
        ValueError: If a record is missing or unreadable, or if the report
            carries no region to inject into.
    """
    processed = Path(processed_dir)
    output = Path(output_dir)

    manifest = read_record(processed / MANIFEST_NAME)
    statistics = read_record(processed / STATISTICS_NAME)

    # Rendered once and placed twice, for the reason the campaign generator
    # gives: two copies of one measurement is two chances to describe the
    # previous build.
    distributions = statistics_table(statistics)

    return {
        output / CHECKSUMS_FRAGMENT: fragment(checksums_table(manifest), COMMAND),
        output / STATISTICS_FRAGMENT: fragment(distributions, COMMAND),
        output / TRUNCATION_FRAGMENT: fragment(truncation_table(statistics), COMMAND),
        Path(report): inject_regions(Path(report), {STATISTICS_REGION: distributions}, COMMAND),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Describe the command line of ``python -m src.data.fragments``.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.data.fragments",
        description=(
            "Emit the Markdown tables of the corpus, from the manifest and the "
            "statistics that python -m src.data.build wrote beside the corpus. Nothing is "
            "measured again."
        ),
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory holding the built corpus and its record.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FRAGMENTS_DIR,
        help="Where the fragments go.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="The page whose marked region carries the statistics table.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report what is out of date and write nothing. Exits non zero on a stale fragment.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Emit or check the corpus fragments from the command line.

    Args:
        argv: Arguments to parse. Defaults to the process arguments.

    Returns:
        Process exit code. One when a record is missing or unreadable, and one
        under ``--check`` when a fragment is stale.
    """
    args = build_argument_parser().parse_args(argv)

    try:
        rendered = build(
            processed_dir=args.processed, output_dir=args.output, report=args.report
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
                f"\n{len(outdated)} generated table(s) out of date, run: make corpus-sync",
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
