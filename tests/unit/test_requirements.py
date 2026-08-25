"""Unit tests keeping the requirements files and pyproject.toml in step.

The repository declares its dependencies once, in ``pyproject.toml``. The
``requirements*.txt`` files exist because a reader expects them and because
``pip install -r`` is what most graders type, not because the project has two
opinions about what it needs. These tests are what makes the second copy safe:
a dependency added on one side and forgotten on the other fails the suite
instead of surfacing as a missing module three commands later.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

RUNTIME = REPO_ROOT / "requirements.txt"
DEV = REPO_ROOT / "requirements-dev.txt"
EDA = REPO_ROOT / "requirements-eda.txt"


def read_requirements(path: Path) -> list[str]:
    """Return the requirement specifiers a file declares.

    Comments, blank lines and ``-r`` includes are dropped: the includes are
    tested on their own, and comparing them to a dependency group would fail
    for the wrong reason.

    Args:
        path: Requirements file to read.

    Returns:
        The specifiers, in the order the file lists them.
    """
    specifiers = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        specifiers.append(stripped)
    return specifiers


def declared_dependencies() -> dict[str, list[str]]:
    """Return the dependency groups pyproject.toml declares.

    Returns:
        The runtime list under ``runtime``, then one entry per optional group.
    """
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    groups = {"runtime": list(project["dependencies"])}
    groups.update(
        {name: list(specifiers) for name, specifiers in project["optional-dependencies"].items()}
    )
    return groups


@pytest.mark.unit
def test_the_requirements_files_are_shipped() -> None:
    for path in (RUNTIME, DEV, EDA):
        assert path.is_file(), f"{path.name} is missing from the repository root."


@pytest.mark.unit
def test_the_runtime_requirements_match_pyproject() -> None:
    assert read_requirements(RUNTIME) == declared_dependencies()["runtime"]


@pytest.mark.unit
def test_the_dev_requirements_match_pyproject() -> None:
    assert read_requirements(DEV) == declared_dependencies()["dev"]


@pytest.mark.unit
def test_the_eda_requirements_match_pyproject() -> None:
    assert read_requirements(EDA) == declared_dependencies()["eda"]


@pytest.mark.unit
def test_the_optional_groups_include_the_runtime_file() -> None:
    """The two optional files must pull the runtime one rather than restate it.

    Without the include, ``pip install -r requirements-dev.txt`` on a clean
    environment installs pytest and nothing to test with, which looks like a
    broken repository rather than an incomplete command.
    """
    for path in (DEV, EDA):
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        assert "-r requirements.txt" in lines, f"{path.name} does not include requirements.txt."


@pytest.mark.unit
def test_every_optional_group_of_pyproject_has_a_file() -> None:
    """A group added to pyproject.toml must not stay without its file.

    This is the direction the other tests cannot catch: they compare the files
    that exist, so a brand new group would simply never be looked at.
    """
    files = {"dev": DEV, "eda": EDA}
    optional = set(declared_dependencies()) - {"runtime"}

    assert optional == set(files), (
        f"pyproject.toml declares {sorted(optional)}, "
        f"the repository ships files for {sorted(files)}."
    )
