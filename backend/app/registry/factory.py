"""Selecting the registry backend from the settings.

``SYNTRA_REGISTRY_BACKEND`` chooses between the directory backed registry the
project ships with and an MLflow model registry. This module is the only place
that reads it, so the API depends on :class:`ModelRegistry` and never on a
concrete backend, which is what section 20 asks for.

**An unwritten backend refuses, it does not fall back.** Selecting ``mlflow``
today raises. Falling back to the local registry would answer the request with
weights from another store, and the API would report a model whose provenance is
not the one the deployment asked for. The failure is at startup, with the name
of the setting that produced it.
"""

from __future__ import annotations

from backend.app.core.config import Settings
from backend.app.registry.base import ModelRegistry
from backend.app.registry.local import LocalModelRegistry

#: Backend writing to a directory, the default and the only one implemented.
LOCAL = "local"

#: Backend reading the MLflow model registry. Declared here because the setting
#: accepts it and a reader of this module must find out what it does.
MLFLOW = "mlflow"

#: Every value ``SYNTRA_REGISTRY_BACKEND`` may take.
BACKENDS: tuple[str, ...] = (LOCAL, MLFLOW)


def build_registry(settings: Settings) -> ModelRegistry:
    """Return the registry the running instance must use.

    Args:
        settings: The application settings.

    Returns:
        The registry.

    Raises:
        NotImplementedError: If the MLflow backend is selected. It is declared
            by the settings and not written yet.
        ValueError: If the setting names a backend that does not exist.
    """
    backend = settings.registry_backend.strip().lower()

    if backend == LOCAL:
        return LocalModelRegistry(settings.registry_root)

    if backend == MLFLOW:
        raise NotImplementedError(
            "The MLflow model registry backend is not implemented. Set "
            f"SYNTRA_REGISTRY_BACKEND={LOCAL} and publish with "
            "python -m src.experiments.publish."
        )

    raise ValueError(
        f"Unknown registry backend {settings.registry_backend!r}. Available: {', '.join(BACKENDS)}."
    )
