"""Where the tracking store lives, resolved without putting a secret in the repository.

Section 19 asks for MLflow. The store behind it is now PostgreSQL rather than a
file beside the checkout, and that moves one thing into the environment: a
connection URI carries a password, and a password does not belong in a tracked
file.

**The repository carries the shape, the environment carries the secret.**
``.env.example`` is committed and shows the URI a reader has to fill;
``.env`` holds the filled one and is ignored. Nothing here has a default host,
user or password compiled into it, so a checkout without a ``.env`` cannot
silently reach somebody else's database.

**A missing configuration is reported, never guessed.** With no URI resolved,
:func:`resolve_tracking_uri` returns ``None`` and the caller leaves MLflow to
its own resolution, which lands on a SQLite file in the working directory. That
fallback is what keeps ``make reproduce`` working on a fresh clone with no
database, and :func:`describe_store` is what stops it from being silent: the
runner prints the store it is about to write to, so a campaign that traced into
a local file instead of PostgreSQL says so on its first line.

**The environment wins over the file.** ``MLFLOW_TRACKING_URI`` exported in a
shell overrides ``.env``, which is the order every tool that reads a dotenv
uses, and the order a one-off run against another store needs.

The parser is eight lines rather than a dependency. ``python-dotenv`` would add
a package to the training image to read ``KEY=value``, and the file this project
writes has no quoting, no interpolation and no multi-line values.
"""

from __future__ import annotations

import os
from pathlib import Path

#: File holding the values of this machine. Ignored by git.
DEFAULT_ENV_FILE = Path(".env")

#: Where the run metadata goes. Read by MLflow itself when it is exported, so
#: the name is MLflow's rather than one of ours.
TRACKING_URI_VARIABLE = "MLFLOW_TRACKING_URI"

#: Where the artefacts go, weights included. PostgreSQL holds the metadata of a
#: run; a 240 MB model is a file, and a database is the wrong place for it.
ARTIFACT_ROOT_VARIABLE = "MLFLOW_ARTIFACT_ROOT"

#: Used when nothing configures the artefact root. Relative to the working
#: directory, ignored by git, and the same directory ``make mlflow-ui`` serves.
DEFAULT_ARTIFACT_ROOT = "mlartifacts"


def load_env_file(path: Path | str = DEFAULT_ENV_FILE) -> dict[str, str]:
    """Read a dotenv file into a mapping.

    Blank lines and lines opening with ``#`` are skipped. A line without ``=``
    is skipped rather than raising: a malformed comment must not stop a
    training run that does not depend on the file.

    Args:
        path: File to read. A missing file yields an empty mapping.

    Returns:
        The declared values, with surrounding whitespace and matching quotes
        stripped from each one.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return {}

    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        cleaned = value.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'":
            cleaned = cleaned[1:-1]
        values[key.strip()] = cleaned
    return values


def _resolve(variable: str, env_file: Path | str) -> str | None:
    """Return one setting, the exported value winning over the file.

    Args:
        variable: Name of the variable to resolve.
        env_file: Dotenv file consulted when the variable is not exported.

    Returns:
        The value, or ``None`` when neither source declares a non empty one.
    """
    exported = os.environ.get(variable)
    if exported:
        return exported
    return load_env_file(env_file).get(variable) or None


def resolve_tracking_uri(env_file: Path | str = DEFAULT_ENV_FILE) -> str | None:
    """Return the tracking URI this machine is configured for.

    Args:
        env_file: Dotenv file consulted when the variable is not exported.

    Returns:
        The URI, or ``None`` when nothing configures one. ``None`` means the
        caller leaves MLflow to resolve its own store, which is a local SQLite
        file, and :func:`describe_store` is what makes that visible.
    """
    return _resolve(TRACKING_URI_VARIABLE, env_file)


def resolve_artifact_root(env_file: Path | str = DEFAULT_ENV_FILE) -> str:
    """Return the root the artefacts of a run are written under.

    Args:
        env_file: Dotenv file consulted when the variable is not exported.

    Returns:
        The configured root, or :data:`DEFAULT_ARTIFACT_ROOT`. Unlike the URI
        this one always has a value: an experiment has to be created somewhere,
        and a directory beside the checkout is a working answer where a
        database is not.
    """
    return _resolve(ARTIFACT_ROOT_VARIABLE, env_file) or DEFAULT_ARTIFACT_ROOT


def redact(uri: str | None) -> str:
    """Return a URI safe to print, with the password replaced.

    A tracking URI reaches the console, the run log and the CI output. The
    host and the database name are what a reader needs to check they are
    writing where they think; the password is not.

    Args:
        uri: The URI to render, or ``None``.

    Returns:
        The URI with everything between ``:`` and ``@`` of the credentials
        replaced by ``***``, or a readable stand-in when there is no URI.
    """
    if not uri:
        return "resolution par defaut de MLflow (SQLite local)"

    scheme, separator, rest = uri.partition("://")
    if not separator or "@" not in rest:
        return uri

    credentials, _, location = rest.partition("@")
    user, has_password, _ = credentials.partition(":")
    if not has_password:
        return uri
    return f"{scheme}://{user}:***@{location}"


def describe_store(
    uri: str | None = None,
    *,
    artifact_root: str | None = None,
    env_file: Path | str = DEFAULT_ENV_FILE,
) -> str:
    """Return the one line a run prints about where it traces.

    Args:
        uri: URI the caller was given, ``None`` to resolve one.
        artifact_root: Root the caller was given, ``None`` to resolve one.
        env_file: Dotenv file consulted when a variable is not exported.

    Returns:
        A line naming the store and the artefact root, password redacted.
    """
    resolved_uri = uri if uri is not None else resolve_tracking_uri(env_file)
    resolved_root = artifact_root or resolve_artifact_root(env_file)
    return f"tracking {redact(resolved_uri)} | artefacts {resolved_root}"
