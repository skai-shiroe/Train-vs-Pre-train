"""Markdown rendering shared by the table generators.

Two modules emit tables into ``reports/_generated``, and they read different
records. :mod:`src.experiments.fragments` reads the run records a campaign
wrote; :mod:`src.data.fragments` reads the corpus record ``python -m
src.data.build`` wrote beside the frozen corpus. What they have in common is
not what they measure, it is how a measurement reaches a table.

That part lives here rather than in a copy on each side. A thousands separator,
the banner marking a file as generated, and the comparison deciding whether a
fragment on disk is stale are not two decisions that happen to agree. Two
spellings of twenty thousand in two tables read side by side is the defect the
generators exist to remove, and reintroducing it one level below them would be
a strange place to stop caring.

The banner is the one thing that must differ: it names the command that rewrites
the file, and pointing a reader at the wrong one is worse than not naming it at
all. So it is a function of the command rather than a constant.

The rendered strings are French, like the pages. The code around them is
English, like the rest of ``src``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

#: What fills a cell whose quantity was never measured. Empty rather than a dash
#: or a zero: a dash is a character a reader may take for a value, and section 44
#: forbids presenting a missing result as a measured one.
MISSING = ""

#: Where the fragments go, for every generator. Beside the records they are
#: rendered from, under a name starting with an underscore so the directory
#: reads as generated rather than written.
FRAGMENTS_DIR = Path("reports") / "_generated"


def banner(command: str) -> str:
    """Render the comment that marks a file as generated.

    Args:
        command: What rewrites the file, as it is typed.

    Returns:
        An HTML comment. A generated file that does not say so gets edited by
        hand exactly once, and the edit is lost without a trace at the next run.
        HTML survives the include and renders nowhere.
    """
    return f"<!-- Généré par {command}. Ne pas éditer à la main. -->"


def fragment(body: str, command: str) -> str:
    """Wrap a rendered body into the text of a fragment file.

    Args:
        body: The Markdown to write.
        command: What rewrites the file, for the banner.

    Returns:
        The banner, the body, and the trailing newline every file in the
        repository ends with.
    """
    return f"{banner(command)}\n\n{body}\n"


def number(value: float | int | str | None) -> str:
    """Render an integer with French thousands separators.

    Args:
        value: The quantity, possibly absent or carried as a string, which is
            how a run record stores its parameter count.

    Returns:
        The digits grouped by three with a space, or :data:`MISSING` when there
        is nothing to render.
    """
    if value is None or value == "":
        return MISSING
    return f"{int(float(value)):,}".replace(",", " ")


def decimal(value: float | int | None, places: int) -> str:
    """Render a measured value with a decimal comma.

    Halves are rounded up, on the decimal digits rather than on the binary float
    behind them. The records hold values already rounded once when they were
    written, so an exact half like ``525.25`` is not a measurement that landed
    there, it is an artefact of that first rounding, and rounding it a second
    time is a display decision rather than a numerical one. Left to
    :func:`format`, that decision would fall out of the binary representation:
    ``525.25`` is exact and rounds to the even digit, ``1380.35`` is stored a
    hair below and rounds down, so two neighbouring cells of one table would
    follow two different rules for a reason no reader could see. Rounding the
    digits up is the convention the pages were typed with and it is the same
    every time.

    Args:
        value: The quantity, or ``None`` when the record holds none.
        places: Decimals to keep.

    Returns:
        The value with French thousands separators and a decimal comma, or
        :data:`MISSING`.
    """
    if value is None:
        return MISSING
    quantum = Decimal(1).scaleb(-places)
    rounded = Decimal(str(float(value))).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:,.{places}f}".replace(",", " ").replace(".", ",")


def share(part: int, total: int) -> str:
    """Render a proportion as a percentage.

    Args:
        part: Count of the subset.
        total: Count of the whole.

    Returns:
        The percentage to one decimal, or :data:`MISSING` when the whole is
        empty, which is not a share of zero.
    """
    if total <= 0:
        return MISSING
    return f"{100.0 * part / total:.1f}".replace(".", ",") + " %"


def table(columns: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    """Render one Markdown table.

    Args:
        columns: Header, in order.
        rows: Cells, already rendered, one sequence per row.

    Returns:
        The table, without a trailing newline.
    """
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(cells) + " |" for cells in rows)
    return "\n".join(lines)


def write(rendered: dict[Path, str]) -> list[Path]:
    """Write the rendered fragments to disk.

    Args:
        rendered: Mapping from destination path to the text that belongs in it.

    Returns:
        The paths written, in the order they were rendered.
    """
    written: list[Path] = []
    for path, text in rendered.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        written.append(path)
    return written


def stale(rendered: dict[Path, str]) -> list[Path]:
    """Return the fragments on disk that are not what the records produce.

    Args:
        rendered: Mapping from destination path to the text that belongs in it.

    Returns:
        The paths that are missing or hold something else, in render order.
    """
    return [
        path
        for path, text in rendered.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != text
    ]
