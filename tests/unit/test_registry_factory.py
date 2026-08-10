"""Unit tests of the registry backend selection.

The property under test is that a backend which is declared but not written
refuses instead of falling back. Serving weights from another store than the one
the deployment asked for is worse than not starting.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.core.config import Settings
from backend.app.registry.factory import BACKENDS, LOCAL, MLFLOW, build_registry
from backend.app.registry.local import LocalModelRegistry

pytestmark = pytest.mark.unit


def settings(**overrides: object) -> Settings:
    """Return settings isolated from any local .env file."""
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_the_local_backend_is_the_default() -> None:
    registry = build_registry(settings())

    assert isinstance(registry, LocalModelRegistry)


def test_the_local_backend_reads_the_configured_root(tmp_path: Path) -> None:
    registry = build_registry(settings(registry_backend=LOCAL, registry_root=tmp_path))

    assert isinstance(registry, LocalModelRegistry)
    assert registry.root == tmp_path


def test_the_mlflow_backend_refuses_rather_than_falling_back() -> None:
    with pytest.raises(NotImplementedError, match="not implemented"):
        build_registry(settings(registry_backend=MLFLOW))


def test_an_unknown_backend_names_the_ones_that_exist() -> None:
    with pytest.raises(ValueError, match="Unknown registry backend"):
        build_registry(settings(registry_backend="s3"))


def test_the_setting_is_read_case_insensitively() -> None:
    assert isinstance(build_registry(settings(registry_backend="  Local ")), LocalModelRegistry)


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_declared_backend_is_handled(backend: str) -> None:
    # A value the settings accept and the factory does not know would fail at
    # the first lookup rather than at startup.
    try:
        build_registry(settings(registry_backend=backend))
    except NotImplementedError:
        pass
    except ValueError:
        pytest.fail(f"{backend} is a declared backend the factory does not recognise.")
