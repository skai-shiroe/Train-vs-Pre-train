"""Unit tests for the model store of section 27."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from backend.app.core.errors import ModelNotFoundError, ModelNotReadyError
from backend.app.inference.base import LoadedModel
from backend.app.registry.base import ModelRegistry
from backend.app.registry.models import CHAMPION, ModelVersion
from backend.app.services.store import ModelStore
from backend.tests.conftest import build_version

CPU = torch.device("cpu")


class FakeSummarizer:
    """A summariser that returns a constant, so a test never builds a network."""

    def __init__(self, label: str) -> None:
        self.label = label

    def summarize(self, documents: Any, config: Any = None, *, batch_size: int = 8) -> list[str]:
        return [f"{self.label} resume" for _ in documents]

    def describe(self) -> dict[str, str]:
        return {"model": self.label}


class FakeRegistry(ModelRegistry):
    """A registry held in memory, with the behaviour of the local one."""

    def __init__(
        self,
        catalog: dict[str, list[ModelVersion]] | None = None,
        *,
        current: dict[str, str] | None = None,
        aliases: dict[str, tuple[str, str]] | None = None,
        artifacts: dict[str, Path | None] | None = None,
    ) -> None:
        self.catalog = catalog or {}
        self.current = current if current is not None else {n: "v1" for n in self.catalog}
        self.aliases = aliases or {}
        self.artifacts = artifacts or {}
        self.resolved: list[str] = []

    def names(self) -> list[str]:
        return sorted(self.catalog)

    def versions(self, model_name: str) -> list[ModelVersion]:
        if model_name not in self.catalog:
            raise ModelNotFoundError(details={"model": model_name})
        return self.catalog[model_name]

    def get_model(self, model_name: str, version: str | None = None) -> ModelVersion:
        if model_name in self.aliases:
            name, target = self.aliases[model_name]
            return self._find(name, target)
        resolved = version or self.current.get(model_name)
        if resolved is None:
            raise ModelNotFoundError(details={"model": model_name})
        return self._find(model_name, resolved)

    def _find(self, name: str, version: str) -> ModelVersion:
        for candidate in self.catalog.get(name, []):
            if candidate.version == version:
                return candidate
        raise ModelNotFoundError(details={"model": name, "version": version})

    def resolve_artifact(self, model: ModelVersion, *, verify: bool = True) -> Path | None:
        self.resolved.append(model.label)
        return self.artifacts.get(model.label)


def loader(version: ModelVersion, weights: Path | None) -> LoadedModel:
    """Build a loaded model without torch."""
    return LoadedModel(
        version=version, summarizer=FakeSummarizer(version.label), device=CPU, token_budget=64
    )


def build_store(registry: FakeRegistry, **kwargs: Any) -> ModelStore:
    """Build a store over a fake registry."""
    return ModelStore(registry, device=CPU, loader=kwargs.pop("loader", loader), **kwargs)


def two_versions() -> FakeRegistry:
    """Return a registry holding two versions of one name and one of another."""
    return FakeRegistry(
        {
            "scratch": [build_version(version="v1"), build_version(version="v2")],
            "pretrained": [build_version(name="pretrained", version="v1")],
        },
        current={"scratch": "v2", "pretrained": "v1"},
        aliases={CHAMPION: ("scratch", "v1")},
    )


# -- Reading ----------------------------------------------------------------


@pytest.mark.unit
def test_the_catalogue_holds_the_version_each_name_serves() -> None:
    store = build_store(two_versions())

    assert [version.label for version in store.catalog()] == ["pretrained:v1", "scratch:v2"]


@pytest.mark.unit
def test_a_name_without_a_default_pointer_is_skipped() -> None:
    registry = two_versions()
    del registry.current["scratch"]

    assert [version.label for version in build_store(registry).catalog()] == ["pretrained:v1"]


@pytest.mark.unit
def test_only_promoted_aliases_are_reported() -> None:
    assert build_store(two_versions()).alias_targets() == {CHAMPION: "scratch:v1"}


@pytest.mark.unit
def test_an_empty_registry_answers_an_empty_catalogue() -> None:
    store = build_store(FakeRegistry())

    assert store.catalog() == []
    assert store.alias_targets() == {}
    assert store.ready is False


@pytest.mark.unit
def test_a_request_without_a_name_resolves_the_champion() -> None:
    assert build_store(two_versions()).resolve().label == "scratch:v1"


@pytest.mark.unit
def test_an_unpromoted_champion_is_not_found() -> None:
    registry = two_versions()
    registry.aliases = {}

    with pytest.raises(ModelNotFoundError):
        build_store(registry).resolve()


# -- Loading ----------------------------------------------------------------


@pytest.mark.unit
def test_a_model_is_loaded_once_and_kept() -> None:
    calls: list[str] = []

    def counting(version: ModelVersion, weights: Path | None) -> LoadedModel:
        calls.append(version.label)
        return loader(version, weights)

    store = build_store(two_versions(), loader=counting)
    first = store.get("scratch")
    second = store.get("scratch")

    assert first is second
    assert calls == ["scratch:v2"]
    assert store.loaded == ["scratch:v2"]


@pytest.mark.unit
def test_the_cache_is_keyed_by_version_not_by_name() -> None:
    store = build_store(two_versions())

    store.get("scratch")
    store.get("scratch", "v1")

    assert store.loaded == ["scratch:v1", "scratch:v2"]


@pytest.mark.unit
def test_an_alias_and_its_target_share_one_load() -> None:
    store = build_store(two_versions())

    through_alias = store.get(CHAMPION)
    through_name = store.get("scratch", "v1")

    assert through_alias is through_name


@pytest.mark.unit
def test_the_checksum_of_the_registry_is_verified_before_a_load() -> None:
    registry = two_versions()

    build_store(registry).get("scratch")

    assert registry.resolved == ["scratch:v2"]


@pytest.mark.unit
def test_a_failed_load_is_recorded_and_raised() -> None:
    def failing(version: ModelVersion, weights: Path | None) -> LoadedModel:
        raise ModelNotReadyError("poids corrompus", details={"model": version.name})

    store = build_store(two_versions(), loader=failing)

    with pytest.raises(ModelNotReadyError):
        store.get("scratch")

    assert store.failures == {"scratch:v2": "poids corrompus"}
    assert store.ready is False


@pytest.mark.unit
def test_a_load_that_finally_succeeds_clears_its_failure() -> None:
    attempts: list[str] = []

    def flaky(version: ModelVersion, weights: Path | None) -> LoadedModel:
        attempts.append(version.label)
        if len(attempts) == 1:
            raise ModelNotReadyError("disque plein")
        return loader(version, weights)

    store = build_store(two_versions(), loader=flaky)
    with pytest.raises(ModelNotReadyError):
        store.get("scratch")
    store.get("scratch")

    assert store.failures == {}
    assert store.ready is True


@pytest.mark.unit
def test_warming_loads_every_name() -> None:
    store = build_store(two_versions())

    assert store.warm() == ["pretrained:v1", "scratch:v2"]
    assert store.ready is True


@pytest.mark.unit
def test_one_broken_model_does_not_stop_the_others() -> None:
    def selective(version: ModelVersion, weights: Path | None) -> LoadedModel:
        if version.name == "scratch":
            raise ModelNotReadyError("checksum mismatch")
        return loader(version, weights)

    store = build_store(two_versions(), loader=selective)

    assert store.warm() == ["pretrained:v1"]
    assert store.failures == {"scratch:v2": "checksum mismatch"}
    assert store.ready is True


@pytest.mark.unit
def test_warming_an_empty_registry_loads_nothing() -> None:
    store = build_store(FakeRegistry())

    assert store.warm() == []
    assert store.failures == {}


@pytest.mark.unit
def test_an_unknown_name_is_not_found_before_anything_is_loaded() -> None:
    store = build_store(two_versions())

    with pytest.raises(ModelNotFoundError):
        store.get("absent")

    assert store.loaded == []
    assert store.failures == {}
