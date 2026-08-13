"""Unit tests of the Markdown primitives shared by the documentation generators.

Two modules render tables into ``reports/_generated`` and both reach a reader through
these functions, so a defect here is a defect on every generated table at once.
Five properties carry that: a fragment must name the command that rewrites it
and not the other one, a missing measurement must not render as a value, an
exact half must round the same way whichever binary float carries it, a
fragment must land on disk with the line endings the repository uses whatever
platform wrote it, and a table injected into a hand written page must land in
its own region without disturbing the prose or the region beside it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.markdown import (
    MISSING,
    REGION_BEGIN,
    REGION_END,
    banner,
    decimal,
    fragment,
    inject_regions,
    number,
    share,
    stale,
    table,
    write,
)

pytestmark = pytest.mark.unit


def page(directory: Path, *names: str) -> Path:
    """Write a page carrying one empty region per name and return it."""
    markers = "\n\n".join(
        f"{REGION_BEGIN.format(name=name)}\n{REGION_END.format(name=name)}" for name in names
    )
    path = directory / "page.md"
    path.write_text(f"# Titre\n\nAvant.\n\n{markers}\n\nApres.\n", encoding="utf-8")
    return path


def test_banner_names_the_command_it_is_given() -> None:
    """The banner points at the command that rewrites the file, not at a fixed one."""
    assert "python -m src.data.fragments" in banner("python -m src.data.fragments")
    assert "python -m src.experiments.fragments" in banner("python -m src.experiments.fragments")


def test_two_generators_do_not_share_one_banner() -> None:
    """A reader sent to the wrong command would run it and see nothing change."""
    assert banner("python -m src.data.fragments") != banner("python -m src.experiments.fragments")


def test_fragment_carries_its_banner_and_one_trailing_newline() -> None:
    """A generated file says it is generated and ends the way every file here does."""
    rendered = fragment("| a |", "cmd")

    assert rendered.startswith(banner("cmd"))
    assert rendered.endswith("| a |\n")
    assert not rendered.endswith("\n\n")


def test_number_groups_thousands_with_a_space() -> None:
    """French thousands are separated by a space, not by a comma."""
    assert number(20000) == "20 000"
    assert number("1234567") == "1 234 567"


@pytest.mark.parametrize("absent", [None, ""])
def test_number_renders_nothing_for_an_absent_quantity(absent: object) -> None:
    """A quantity the record does not hold is empty, never a zero."""
    assert number(absent) == MISSING  # type: ignore[arg-type]


def test_decimal_rounds_an_exact_half_up() -> None:
    """The convention the pages were typed with, applied to the decimal digits."""
    assert decimal(525.25, 1) == "525,3"


def test_decimal_rounds_the_same_way_whatever_the_float_carries() -> None:
    """525.25 is exactly representable and 1380.35 is not.

    Left to ``format`` the first would round to the even digit and the second
    would follow the hair of error in its binary representation, so two cells of
    one table would round in two directions for a reason no reader could see.
    """
    assert decimal(525.25, 1) == "525,3"
    assert decimal(1380.35, 1) == "1 380,4"


def test_decimal_writes_a_decimal_comma_and_grouped_thousands() -> None:
    """The pages are French, and so is the punctuation of their numbers."""
    assert decimal(1305.0, 1) == "1 305,0"
    assert decimal(0.0966, 3) == "0,097"


def test_decimal_renders_nothing_for_an_absent_measurement() -> None:
    """An unmeasured quantity is empty, so it cannot be read as a zero."""
    assert decimal(None, 1) == MISSING


def test_share_renders_nothing_for_an_empty_whole() -> None:
    """No examples is not a share of zero percent, it is no share at all."""
    assert share(0, 0) == MISSING
    assert share(3, -1) == MISSING


def test_share_renders_one_decimal() -> None:
    """The precision the truncation table is read to."""
    assert share(7686, 20000) == "38,4 %"


def test_table_renders_a_header_a_rule_and_its_rows() -> None:
    """The shape MkDocs needs, without a trailing newline of its own."""
    rendered = table(["A", "B"], [["1", "2"], ["3", "4"]])

    assert rendered.splitlines() == ["| A | B |", "| --- | --- |", "| 1 | 2 |", "| 3 | 4 |"]
    assert not rendered.endswith("\n")


def test_table_renders_a_header_alone_when_there_is_no_row() -> None:
    """An empty table still says what it would have held."""
    assert table(["A"], []).splitlines() == ["| A |", "| --- |"]


def test_write_creates_the_directory_and_returns_what_it_wrote(tmp_path: Path) -> None:
    """The fragments directory may not exist yet on a fresh clone."""
    target = tmp_path / "nested" / "one.md"

    written = write({target: "body\n"})

    assert written == [target]
    assert target.read_text(encoding="utf-8") == "body\n"


def test_write_uses_the_line_endings_of_the_repository(tmp_path: Path) -> None:
    """Without this the fragments come out CRLF on Windows and every one reads as stale."""
    target = tmp_path / "one.md"

    write({target: "first\nsecond\n"})

    assert target.read_bytes() == b"first\nsecond\n"


def test_stale_reports_a_fragment_that_does_not_exist(tmp_path: Path) -> None:
    """A missing fragment is out of date, not absent from the comparison."""
    missing = tmp_path / "gone.md"

    assert stale({missing: "body\n"}) == [missing]


def test_stale_reports_a_fragment_edited_by_hand(tmp_path: Path) -> None:
    """The property the whole mechanism rests on: an edit does not survive unnoticed."""
    target = tmp_path / "one.md"
    write({target: "generated\n"})
    target.write_text("retyped by hand\n", encoding="utf-8")

    assert stale({target: "generated\n"}) == [target]


def test_stale_reports_nothing_when_the_disk_matches(tmp_path: Path) -> None:
    """A fresh fragment is not reported, which is what makes the check usable."""
    target = tmp_path / "one.md"
    write({target: "generated\n"})

    assert stale({target: "generated\n"}) == []


# ---------------------------------------------------------------------------
# Injection, for the pages a generated table lands in the middle of
# ---------------------------------------------------------------------------


def test_each_region_of_a_page_is_rewritten_independently(tmp_path: Path) -> None:
    """A page carries several regions, and one pass fills them all."""
    path = page(tmp_path, "un", "deux")

    text = inject_regions(path, {"un": "| Un |", "deux": "| Deux |"}, "cmd")

    assert text.startswith("# Titre\n\nAvant.\n")
    assert text.endswith("Apres.\n")
    assert f"{REGION_BEGIN.format(name='un')}\n{banner('cmd')}\n\n| Un |\n" in text
    assert f"{REGION_BEGIN.format(name='deux')}\n{banner('cmd')}\n\n| Deux |\n" in text


def test_a_region_the_call_does_not_name_is_left_alone(tmp_path: Path) -> None:
    """Two generators share the report, and neither may blank the other's table."""
    path = page(tmp_path, "mien", "tien")
    path.write_text(inject_regions(path, {"tien": "| Tien |"}, "autre"), encoding="utf-8")

    text = inject_regions(path, {"mien": "| Mien |"}, "cmd")

    assert "| Tien |" in text
    assert "| Mien |" in text


def test_injecting_twice_writes_the_same_page(tmp_path: Path) -> None:
    """The region is found by its markers, so a second pass replaces rather than nests."""
    path = page(tmp_path, "un", "deux")
    regions = {"un": "| Un |", "deux": "| Deux |"}

    once = inject_regions(path, regions, "cmd")
    path.write_text(once, encoding="utf-8")
    twice = inject_regions(path, regions, "cmd")

    assert twice == once
    assert twice.count(REGION_BEGIN.format(name="un")) == 1


def test_a_page_missing_one_of_its_regions_is_refused(tmp_path: Path) -> None:
    """A table with nowhere to go must stop the run, not vanish silently."""
    path = page(tmp_path, "un")

    with pytest.raises(ValueError, match="carries no `deux` region"):
        inject_regions(path, {"un": "| Un |", "deux": "| Deux |"}, "cmd")


def test_markers_in_the_wrong_order_are_refused(tmp_path: Path) -> None:
    """A closing marker before its opening one is a parse error, not an empty region."""
    path = tmp_path / "page.md"
    path.write_text(
        f"{REGION_END.format(name='un')}\n{REGION_BEGIN.format(name='un')}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="carries no `un` region"):
        inject_regions(path, {"un": "| Un |"}, "cmd")


def test_a_missing_page_is_refused_rather_than_created(tmp_path: Path) -> None:
    """Writing the file would produce a page holding a table and nothing else."""
    with pytest.raises(ValueError, match="has nowhere to go"):
        inject_regions(tmp_path / "nowhere.md", {"un": "| Un |"}, "cmd")
