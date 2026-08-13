"""Unit tests for the comparison with the archived campaign tree.

This script is what stands in for the lost commit of the campaign: it is the
evidence a reader re-runs instead of believing a hash. Its notion of "the same
code" is therefore checked rather than trusted, and the cases that matter are
the ones where two files differ textually but compute the same thing.

The git side is not exercised here. What is checked is the comparison itself,
which is where a wrong answer would be silent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.compare_archive import compare, format_report, logic_of, main

pytestmark = pytest.mark.unit


def test_a_comment_is_not_a_difference() -> None:
    before = "def f(x):\n    return x + 1\n"
    after = "def f(x):\n    # The increment nobody had explained.\n    return x + 1\n"

    assert logic_of(before) == logic_of(after)


def test_a_docstring_is_not_a_difference_either() -> None:
    # Docstrings are AST nodes, unlike comments: without removing them, the
    # nine files that only gained prose would all read as changed.
    before = "def f(x):\n    return x\n"
    after = 'def f(x):\n    """Return x."""\n    return x\n'

    assert logic_of(before) == logic_of(after)


def test_a_module_docstring_is_removed_too() -> None:
    assert logic_of('"""A module."""\nY = 1\n') == logic_of("Y = 1\n")


def test_a_body_reduced_to_its_docstring_stays_parsable() -> None:
    # Removing the only statement of a function would leave an empty body, and
    # ast.dump would raise rather than compare.
    assert logic_of('def f():\n    """Nothing yet."""\n') == logic_of("def f():\n    pass\n")


def test_reformatting_is_not_a_difference() -> None:
    before = "def f(a, b):\n    return a+b\n"
    after = "def f(\n    a,\n    b,\n):\n    return a + b\n"

    assert logic_of(before) == logic_of(after)


def test_a_changed_constant_is_a_difference() -> None:
    # The case the whole script exists for: a number that moved changes what a
    # run measures, and no amount of identical prose hides it.
    assert logic_of("LIMIT = 512\n") != logic_of("LIMIT = 256\n")


def test_a_changed_operator_is_a_difference() -> None:
    assert logic_of("def f(a, b):\n    return a + b\n") != logic_of(
        "def f(a, b):\n    return a - b\n"
    )


def test_the_comparison_sorts_files_into_the_four_answers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = {
        "src/kept.py": "X = 1\n",
        "src/moved.py": "X = 1\n",
        "src/gone.py": "X = 1\n",
    }
    monkeypatch.setattr("scripts.compare_archive.REPO_ROOT", tmp_path)
    monkeypatch.setattr("scripts.compare_archive.list_files", lambda ref, subtree: sorted(archive))
    monkeypatch.setattr("scripts.compare_archive.read_at", lambda ref, path: archive.get(path))

    source = tmp_path / "src"
    source.mkdir()
    (source / "kept.py").write_text("# same logic, new comment\nX = 1\n", encoding="utf-8")
    (source / "moved.py").write_text("X = 2\n", encoding="utf-8")
    (source / "added.py").write_text("X = 3\n", encoding="utf-8")

    result = compare("some-ref", "src")

    assert result["identical"] == ["src/kept.py"]
    assert result["changed"] == ["src/moved.py"]
    assert result["only_in_archive"] == ["src/gone.py"]
    assert result["only_in_worktree"] == ["src/added.py"]


def test_a_file_that_no_longer_parses_is_reported_as_changed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A comparison that cannot be made is not a comparison that succeeded.
    monkeypatch.setattr("scripts.compare_archive.REPO_ROOT", tmp_path)
    monkeypatch.setattr("scripts.compare_archive.list_files", lambda ref, subtree: ["src/broken.py"])
    monkeypatch.setattr("scripts.compare_archive.read_at", lambda ref, path: "X = 1\n")

    source = tmp_path / "src"
    source.mkdir()
    (source / "broken.py").write_text("def (:\n", encoding="utf-8")

    assert compare("some-ref", "src")["changed"] == ["src/broken.py"]


def test_the_report_names_what_moved() -> None:
    result = {
        "identical": ["src/a.py"],
        "changed": ["src/b.py"],
        "only_in_archive": [],
        "only_in_worktree": [],
    }

    report = format_report("some-ref", result)

    assert "src/b.py" in report
    assert "src/a.py" not in report


def test_an_unmoved_tree_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.compare_archive.compare",
        lambda ref, subtree: {
            "identical": ["src/a.py"],
            "changed": [],
            "only_in_archive": [],
            "only_in_worktree": [],
        },
    )

    assert main([]) == 0


def test_a_moved_tree_exits_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.compare_archive.compare",
        lambda ref, subtree: {
            "identical": [],
            "changed": ["src/b.py"],
            "only_in_archive": [],
            "only_in_worktree": [],
        },
    )

    assert main(["--json"]) == 1
