"""What the running instance serves, and what it has in memory.

Section 27 asks for the models to be loaded at startup and kept in memory. The
store is the object that holds them. It sits between the registry, which knows
what exists, and the endpoints, which know what was asked for.

**A lookup and a load are two steps, and the store keeps them apart.** Resolving
``champion`` costs a file read; building the network costs seconds and a few
hundred megabytes. Because the registry answers first, the store can notice that
the resolved version is already in memory and skip the load entirely, which is
what makes a promotion cheap and a repeated request free.

**Loading is keyed by version, never by name.** ``scratch`` served ``v1``
yesterday and serves ``v2`` today; both may sit in memory at once, and a request
naming either gets the weights it asked for. A cache keyed by name would return
the old network under the new name after a promotion.

**A model that fails to load does not stop the instance.** The failure is
recorded with its reason and reported by ``/ready``, which is what a health
check reads. Refusing to start would take down an API that can still answer for
its other models, and would hide the reason behind a container restart loop.

**Nothing here is async.** Building a torch model blocks; the services run the
store in a worker thread rather than pretend otherwise.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import torch

from backend.app.core.errors import ModelNotFoundError, SyntraError
from backend.app.inference.base import LoadedModel
from backend.app.inference.loader import load_version
from backend.app.registry.base import ModelRegistry
from backend.app.registry.models import ALIASES, CHAMPION, ModelVersion

LOGGER = logging.getLogger(__name__)

#: Signature of the loader, injected so the store can be tested without torch.
Loader = Callable[[ModelVersion, Path | None], LoadedModel]


class ModelStore:
    """The models the instance can serve, and those it currently holds."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        device: torch.device,
        loader: Loader | None = None,
    ) -> None:
        """Build the store.

        Args:
            registry: The registry the instance resolves names through.
            device: Where loaded weights are placed.
            loader: How a resolved version becomes a network. Defaults to
                :func:`backend.app.inference.loader.load_version`.
        """
        self.registry = registry
        self.device = device
        self._load: Loader = loader or (
            lambda version, weights: load_version(version, weights, device=device)
        )
        self._models: dict[str, LoadedModel] = {}
        self._failures: dict[str, str] = {}

    # -- State ------------------------------------------------------------

    @property
    def loaded(self) -> list[str]:
        """Return the versions currently in memory.

        Returns:
            The ``name:version`` labels, sorted.
        """
        return sorted(self._models)

    @property
    def failures(self) -> dict[str, str]:
        """Return the versions that failed to load, with their reason.

        Returns:
            A mapping from label to reason, safe to expose: the reasons are
            written by the loader and carry no path and no stack trace.
        """
        return dict(self._failures)

    @property
    def ready(self) -> bool:
        """Return whether the instance can answer an inference request.

        Returns:
            ``True`` as soon as one version is in memory. An instance holding
            one working model out of two is degraded, not down.
        """
        return bool(self._models)

    # -- Reading the registry ---------------------------------------------

    def catalog(self) -> list[ModelVersion]:
        """Return the version each registered name currently serves.

        Returns:
            One version per name, sorted by name. A name whose default pointer
            is missing is skipped and logged: the registry knows it exists but
            cannot say which version answers for it, and inventing one would
            put a version behind a name that never served it.
        """
        versions: list[ModelVersion] = []
        for name in self.registry.names():
            try:
                versions.append(self.registry.get_model(name))
            except ModelNotFoundError:
                LOGGER.warning("model %s has versions but no default pointer", name)
        return versions

    def alias_targets(self) -> dict[str, str]:
        """Return where each deployment alias points.

        Returns:
            A mapping from alias to ``name:version``, holding only the aliases
            that are actually promoted. An alias pointing nowhere is absent
            rather than empty: section 20 gives an alias one exact target, and
            a null target would read as a version named null.
        """
        targets: dict[str, str] = {}
        for alias in ALIASES:
            try:
                targets[alias] = self.registry.get_model(alias).label
            except ModelNotFoundError:
                continue
        return targets

    def resolve(self, name: str | None = None, version: str | None = None) -> ModelVersion:
        """Resolve what a request named into one exact version.

        Args:
            name: A registered name or a deployment alias. ``None`` means the
                champion, which is the version the deployment promoted.
            version: An exact version, when the caller pinned one.

        Returns:
            The resolved version.

        Raises:
            ModelNotFoundError: If the name, the alias or the version is
                unknown, or if nothing has been promoted to the alias that was
                asked for.
        """
        return self.registry.get_model(name or CHAMPION, version)

    # -- Loading ----------------------------------------------------------

    def get(self, name: str | None = None, version: str | None = None) -> LoadedModel:
        """Return the loaded model a request named, loading it if needed.

        Args:
            name: A registered name or a deployment alias.
            version: An exact version.

        Returns:
            The loaded model.

        Raises:
            ModelNotFoundError: If nothing is registered under that name.
            ModelNotReadyError: If the version exists but cannot be loaded.
        """
        resolved = self.resolve(name, version)
        held = self._models.get(resolved.label)
        if held is not None:
            return held
        return self.load(resolved)

    def load(self, version: ModelVersion) -> LoadedModel:
        """Load one resolved version into memory.

        Args:
            version: The version to load.

        Returns:
            The loaded model.

        Raises:
            ModelNotFoundError: If the artifact the version points at is gone.
            ModelNotReadyError: If it does not match its checksum, or cannot be
                turned into a network.
        """
        try:
            weights = self.registry.resolve_artifact(version)
            loaded = self._load(version, weights)
        except SyntraError as error:
            self._failures[version.label] = error.message
            LOGGER.error("model %s failed to load: %s", version.label, error.message)
            raise

        self._models[version.label] = loaded
        self._failures.pop(version.label, None)
        LOGGER.info(
            "model %s loaded on %s",
            version.label,
            loaded.device.type,
            extra={"model": version.name, "version": version.version},
        )
        return loaded

    def warm(self) -> list[str]:
        """Load the version every registered name currently serves.

        Called at startup when ``SYNTRA_EAGER_LOAD_MODELS`` is on, so that the
        first request pays for generation and not for a load. Failures are
        recorded rather than raised: an instance that can still serve one model
        is worth starting, and ``/ready`` reports what is missing.

        Returns:
            The labels loaded, sorted.
        """
        for version in self.catalog():
            try:
                self.load(version)
            except SyntraError:
                continue
        return self.loaded
