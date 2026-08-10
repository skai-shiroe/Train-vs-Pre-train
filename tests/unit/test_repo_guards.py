"""Unit tests for the repository guards enforced by pre-commit and the CI.

These two guards are blocking rules of the specification, so they are tested
like production code: a guard that silently stops detecting is worse than no
guard at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_commit_msg import first_meaningful_line
from scripts.check_commit_msg import main as commit_msg_main
from scripts.check_dashes import FORBIDDEN, scan_file
from scripts.check_dashes import main as dashes_main

EM_DASH = chr(0x2014)
EN_DASH = chr(0x2013)


def write(tmp_path: Path, name: str, content: str) -> Path:
    """Write a UTF-8 file and return its path.

    Args:
        tmp_path: Directory provided by the pytest fixture.
        name: File name to create.
        content: Text written to the file.

    Returns:
        The path of the created file.
    """
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# check_dashes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_forbidden_set_covers_the_four_long_dashes() -> None:
    assert {ord(char) for char in FORBIDDEN} == {0x2012, 0x2013, 0x2014, 0x2015}


@pytest.mark.unit
def test_clean_file_produces_no_hit(tmp_path: Path) -> None:
    path = write(tmp_path, "clean.md", "Une phrase simple, sans tiret long.\n")

    assert scan_file(path) == []


@pytest.mark.unit
@pytest.mark.parametrize("dash", [EM_DASH, EN_DASH], ids=["em", "en"])
def test_long_dash_is_reported_with_its_position(tmp_path: Path, dash: str) -> None:
    path = write(tmp_path, "dirty.md", f"premiere ligne\nsuite {dash} fin\n")

    hits = scan_file(path)

    assert len(hits) == 1
    line, column, char = hits[0]
    assert (line, char) == (2, dash)
    assert column == 7


@pytest.mark.unit
def test_ordinary_hyphen_stays_legal(tmp_path: Path) -> None:
    path = write(tmp_path, "hyphen.md", "pre-entraine, sous-ensemble, --cov-fail-under\n")

    assert scan_file(path) == []


@pytest.mark.unit
def test_binary_file_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"\xff\xfe\x00\x01")

    assert scan_file(path) == []


@pytest.mark.unit
def test_main_fails_on_a_dirty_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write(tmp_path, "dirty.md", f"texte {EM_DASH} texte\n")

    assert dashes_main([str(path)]) == 1
    assert "U+2014" in capsys.readouterr().out


@pytest.mark.unit
def test_main_passes_on_clean_input(tmp_path: Path) -> None:
    path = write(tmp_path, "clean.md", "texte : texte\n")

    assert dashes_main([str(path)]) == 0


@pytest.mark.unit
def test_missing_path_is_ignored() -> None:
    assert dashes_main(["does/not/exist.md"]) == 0


@pytest.mark.unit
def test_the_guard_does_not_flag_itself() -> None:
    guard = Path(__file__).resolve().parents[2] / "scripts" / "check_dashes.py"

    assert scan_file(guard) == []


# ---------------------------------------------------------------------------
# check_commit_msg
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "subject",
    [
        "feat(api): add the compare endpoint",
        "fix(scratch): correct the causal mask shape",
        "docs: document the ablation protocol",
        "refactor(training)!: change the checkpoint format",
        "ci(github): split the pipeline into modules",
    ],
)
def test_valid_subjects_are_accepted(tmp_path: Path, subject: str) -> None:
    path = write(tmp_path, "COMMIT_EDITMSG", f"{subject}\n")

    assert commit_msg_main([str(path)]) == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "subject",
    [
        "added stuff",
        "wip",
        "Feat(api): capitalised type",
        "feat api: missing parenthesis",
        "feat: subject ending with a period.",
        "chore:missing space after the colon",
    ],
)
def test_invalid_subjects_are_rejected(tmp_path: Path, subject: str) -> None:
    path = write(tmp_path, "COMMIT_EDITMSG", f"{subject}\n")

    assert commit_msg_main([str(path)]) == 1


@pytest.mark.unit
def test_merge_and_fixup_commits_are_exempt(tmp_path: Path) -> None:
    for subject in ("Merge branch 'main'", 'Revert "feat: x"', "fixup! feat: x"):
        path = write(tmp_path, "COMMIT_EDITMSG", f"{subject}\n")
        assert commit_msg_main([str(path)]) == 0


@pytest.mark.unit
def test_comment_lines_are_skipped() -> None:
    message = "# Please enter the commit message\n\nfeat(data): add the XSum loader\n"

    assert first_meaningful_line(message) == "feat(data): add the XSum loader"


@pytest.mark.unit
def test_empty_message_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "COMMIT_EDITMSG", "\n# comment only\n")

    assert commit_msg_main([str(path)]) == 1


@pytest.mark.unit
def test_overlong_subject_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "COMMIT_EDITMSG", "feat: " + "x" * 80 + "\n")

    assert commit_msg_main([str(path)]) == 1


@pytest.mark.unit
def test_missing_argument_is_reported() -> None:
    assert commit_msg_main([]) == 1
