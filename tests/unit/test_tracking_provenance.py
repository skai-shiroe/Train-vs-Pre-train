"""Unit tests of the commit recorded beside every run.

The property under test is that an unanswerable question produces ``unknown``
and never a plausible looking default. A clean working tree and a machine
without git must not report the same thing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.tracking import provenance
from src.tracking.provenance import UNKNOWN, Provenance, describe_provenance, run_git

pytestmark = pytest.mark.unit


def git(*arguments: str, cwd: Path) -> None:
    """Run a git command in a test repository."""
    subprocess.run(["git", *arguments], cwd=cwd, check=True, capture_output=True)  # nosec B603 B607


def test_an_unknown_working_tree_is_not_reported_as_clean() -> None:
    # False would state that the tree was inspected and found unchanged.
    rendered = Provenance(commit="abc", branch="main", dirty=None).to_dict()

    assert rendered["git_dirty"] == UNKNOWN


def test_a_clean_tree_and_a_dirty_one_are_distinguished() -> None:
    assert Provenance("abc", "main", False).to_dict()["git_dirty"] == "false"
    assert Provenance("abc", "main", True).to_dict()["git_dirty"] == "true"


def test_a_directory_outside_a_repository_answers_unknown(tmp_path: Path) -> None:
    result = describe_provenance(tmp_path)

    assert result.commit == UNKNOWN
    assert result.branch == UNKNOWN
    assert result.dirty is None


def test_a_repository_without_a_commit_answers_unknown(tmp_path: Path) -> None:
    # The state of this repository while the lots are being staged: a working
    # tree full of files and no commit to point at.
    git("init", cwd=tmp_path)

    result = describe_provenance(tmp_path)

    assert result.commit == UNKNOWN
    assert result.dirty is False


def test_a_commit_is_read_back(tmp_path: Path) -> None:
    git("init", cwd=tmp_path)
    git("config", "user.email", "test@example.com", cwd=tmp_path)
    git("config", "user.name", "Test", cwd=tmp_path)
    (tmp_path / "file.txt").write_text("content\n", encoding="utf-8")
    git("add", "file.txt", cwd=tmp_path)
    git("commit", "-m", "chore: initial", cwd=tmp_path)

    result = describe_provenance(tmp_path)

    assert len(result.commit) == 40
    assert result.dirty is False

    (tmp_path / "file.txt").write_text("changed\n", encoding="utf-8")

    assert describe_provenance(tmp_path).dirty is True


def test_a_failing_command_answers_nothing(tmp_path: Path) -> None:
    assert run_git(("rev-parse", "HEAD"), tmp_path) is None


def test_a_machine_without_git_answers_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(provenance.shutil, "which", lambda name: None)

    assert run_git(("--version",), tmp_path) is None


def test_a_hung_command_answers_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A run that finished must not be held hostage by a git that does not.
    def timeout(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd="git", timeout=provenance.GIT_TIMEOUT_SECONDS)

    monkeypatch.setattr(provenance.subprocess, "run", timeout)

    assert run_git(("--version",), tmp_path) is None


def test_the_repository_of_the_project_answers_the_three_fields() -> None:
    rendered = describe_provenance().to_dict()

    assert set(rendered) == {"git_commit", "git_branch", "git_dirty"}
