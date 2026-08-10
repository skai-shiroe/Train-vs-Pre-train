"""The contract the API depends on.

Section 20 asks for an abstraction so the backend never carries a path to a
checkpoint. Everything downstream, the loader of section 27 and the endpoints of
sections 23 and 24, is written against this class and against
:class:`backend.app.registry.models.ModelVersion`, so replacing the local
backend with the MLflow one changes a factory and nothing else.

**A lookup resolves a version, it does not load one.** The registry answers with
an identity and a location. Building a torch module out of that is the loader's
job, and keeping the two apart is what lets the API resolve ``champion`` at
startup, compare it with what it already holds, and skip the load when they
match.

**An alias points at one exact version, never at the newest one.** If
``champion`` meant the latest version of the champion model, publishing would
change what production serves without anyone promoting anything. Promotion is a
separate act, and it names a version.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from backend.app.registry.models import ModelVersion


class ModelRegistry(ABC):
    """Resolve a model name, and optionally a version, to a published version."""

    @abstractmethod
    def names(self) -> list[str]:
        """List the registered model names.

        Returns:
            The names, sorted. Aliases are not names and do not appear.
        """

    @abstractmethod
    def versions(self, model_name: str) -> list[ModelVersion]:
        """List every published version of one model.

        Args:
            model_name: Registered model name.

        Returns:
            The versions, oldest first.

        Raises:
            ModelNotFoundError: If no model is registered under that name.
        """

    @abstractmethod
    def get_model(self, model_name: str, version: str | None = None) -> ModelVersion:
        """Resolve one version.

        Args:
            model_name: A registered model name, or one of the deployment
                aliases of section 20.
            version: An exact version. When absent, an alias resolves to the
                version it points at, and a model name resolves to the version
                it currently serves.

        Returns:
            The resolved version.

        Raises:
            ModelNotFoundError: If the name, the alias or the version is
                unknown, or if a name has no version to serve yet.
        """

    @abstractmethod
    def resolve_artifact(self, model: ModelVersion, *, verify: bool = True) -> Path | None:
        """Return the local file holding the weights of a version.

        Args:
            model: The resolved version.
            verify: Whether the checksum recorded at publication is checked
                against the file. On by default: the guarantee the registry
                offers is that the weights served are the weights measured, and
                a check that is off by default is not a guarantee.

        Returns:
            The path of the weights, or ``None`` for a version whose weights are
            pulled from Hugging Face at load time.

        Raises:
            ModelNotFoundError: If the artifact the version points at is
                missing.
            ModelNotReadyError: If the artifact does not match its checksum.
        """
