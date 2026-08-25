"""Unit tests for the resolution of the tracking store.

The value being resolved is a connection URI carrying a password, so two
properties matter more than the parsing itself: an exported variable must win
over the file, and nothing that reaches a console may contain the password.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tracking.store import (
    ARTIFACT_ROOT_VARIABLE,
    DEFAULT_ARTIFACT_ROOT,
    TRACKING_URI_VARIABLE,
    describe_store,
    load_env_file,
    redact,
    resolve_artifact_root,
    resolve_tracking_uri,
)

pytestmark = pytest.mark.unit

URI = "postgresql://syntra_mlflow:s3cret@192.168.32.128:5432/syntra_mlflow"


@pytest.fixture(autouse=True)
def _no_inherited_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test against an environment that declares nothing.

    Without this the suite passes or fails depending on whether the machine
    running it has a ``.env`` exported, which is the opposite of a unit test.
    """
    monkeypatch.delenv(TRACKING_URI_VARIABLE, raising=False)
    monkeypatch.delenv(ARTIFACT_ROOT_VARIABLE, raising=False)


def write_env(directory: Path, body: str) -> Path:
    """Write a dotenv file and return its path.

    Args:
        directory: Directory the file is written in.
        body: Content of the file.

    Returns:
        The path of the written file.
    """
    path = directory / ".env"
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_a_missing_file_is_not_an_error(tmp_path: Path) -> None:
    assert load_env_file(tmp_path / "absent") == {}


def test_comments_blank_lines_and_malformed_lines_are_skipped(tmp_path: Path) -> None:
    path = write_env(tmp_path, "# commentaire\n\nSANS_EGAL\nA=1\n")

    assert load_env_file(path) == {"A": "1"}


def test_surrounding_quotes_and_whitespace_are_stripped(tmp_path: Path) -> None:
    path = write_env(tmp_path, '  A = "un"  \nB = \'deux\'\nC=trois\n')

    assert load_env_file(path) == {"A": "un", "B": "deux", "C": "trois"}


def test_a_value_containing_an_equal_sign_survives(tmp_path: Path) -> None:
    # A base64 password ends on '=' often enough for this to matter.
    path = write_env(tmp_path, f"{TRACKING_URI_VARIABLE}=postgresql://u:p=q@h:5432/d\n")

    assert load_env_file(path)[TRACKING_URI_VARIABLE] == "postgresql://u:p=q@h:5432/d"


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_the_file_is_read_when_nothing_is_exported(tmp_path: Path) -> None:
    path = write_env(tmp_path, f"{TRACKING_URI_VARIABLE}={URI}\n")

    assert resolve_tracking_uri(path) == URI


def test_an_exported_variable_wins_over_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_env(tmp_path, f"{TRACKING_URI_VARIABLE}={URI}\n")
    monkeypatch.setenv(TRACKING_URI_VARIABLE, "postgresql://autre@ailleurs:5432/d")

    assert resolve_tracking_uri(path) == "postgresql://autre@ailleurs:5432/d"


def test_nothing_configured_resolves_to_none(tmp_path: Path) -> None:
    assert resolve_tracking_uri(tmp_path / "absent") is None


def test_an_empty_value_counts_as_absent(tmp_path: Path) -> None:
    path = write_env(tmp_path, f"{TRACKING_URI_VARIABLE}=\n")

    assert resolve_tracking_uri(path) is None


def test_the_artifact_root_always_has_a_value(tmp_path: Path) -> None:
    # Unlike the URI: an experiment has to be created somewhere, and a
    # directory beside the checkout is an answer where a database is not.
    assert resolve_artifact_root(tmp_path / "absent") == DEFAULT_ARTIFACT_ROOT


def test_the_artifact_root_is_read_from_the_file(tmp_path: Path) -> None:
    path = write_env(tmp_path, f"{ARTIFACT_ROOT_VARIABLE}=D:/mlartifacts\n")

    assert resolve_artifact_root(path) == "D:/mlartifacts"


# ---------------------------------------------------------------------------
# What reaches a console
# ---------------------------------------------------------------------------


def test_the_password_never_survives_redaction() -> None:
    rendered = redact(URI)

    assert "s3cret" not in rendered
    assert "syntra_mlflow:***@192.168.32.128:5432/syntra_mlflow" in rendered


def test_a_uri_without_a_password_is_left_alone() -> None:
    assert redact("postgresql://hote:5432/base") == "postgresql://hote:5432/base"


def test_a_uri_with_a_user_and_no_password_is_left_alone() -> None:
    assert redact("postgresql://utilisateur@hote:5432/base") == "postgresql://utilisateur@hote:5432/base"


def test_no_uri_renders_as_a_readable_stand_in() -> None:
    rendered = redact(None)

    assert "SQLite" in rendered


def test_the_description_names_the_store_and_the_root_without_the_password(
    tmp_path: Path,
) -> None:
    path = write_env(tmp_path, f"{TRACKING_URI_VARIABLE}={URI}\n{ARTIFACT_ROOT_VARIABLE}=arte\n")

    described = describe_store(env_file=path)

    assert "s3cret" not in described
    assert "192.168.32.128" in described
    assert "arte" in described
