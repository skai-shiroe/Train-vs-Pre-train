"""Unit tests of the directory backed registry.

Three properties matter more than the rest. Publishing must not change what is
served, a version must never be rewritten, and the weights handed to the API
must be the ones whose checksum was recorded at publication.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.core.errors import ModelNotFoundError, ModelNotReadyError
from backend.app.registry.local import LocalModelRegistry, checksum_file
from backend.app.registry.models import ModelDraft, ModelSource

pytestmark = pytest.mark.unit


def weights(tmp_path: Path, content: bytes = b"weights") -> Path:
    """Write a stand in for a checkpoint and return its path."""
    path = tmp_path / "source.pt"
    path.write_bytes(content)
    return path


def scratch_draft(name: str = "scratch") -> ModelDraft:
    """Return a draft of a from scratch model."""
    return ModelDraft(
        name=name,
        source=ModelSource.CHECKPOINT,
        architecture={"type": "scratch", "d_model": 256},
        experiment="scratch_100",
        dataset_version="259d8397ce78",  # pragma: allowlist secret
        git_commit="deadbeef",
        metrics={"rougeL_f": 0.15},
    )


def baseline_draft() -> ModelDraft:
    """Return a draft of the zero shot baseline."""
    return ModelDraft(
        name="pretrained",
        source=ModelSource.HUGGING_FACE,
        hf_id="t5-small",
        revision="df1b051c",
        experiment="pretrained_zero_shot",
    )


# ---------------------------------------------------------------------------
# Reading an empty registry
# ---------------------------------------------------------------------------


def test_reading_an_empty_registry_creates_nothing(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "registry")

    assert registry.names() == []
    assert registry.read_index().current == {}
    assert not (tmp_path / "registry").exists()


def test_an_unknown_name_is_not_found(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path)

    with pytest.raises(ModelNotFoundError):
        registry.get_model("scratch")


def test_an_alias_pointing_nowhere_is_not_found(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path)

    with pytest.raises(ModelNotFoundError):
        registry.get_model("champion")


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------


def test_the_first_version_becomes_the_one_the_name_serves(tmp_path: Path) -> None:
    # Nothing to displace, and a name that serves nothing is a name the API
    # cannot answer for.
    registry = LocalModelRegistry(tmp_path / "registry")

    published = registry.publish(scratch_draft(), weights=weights(tmp_path))

    assert published.version == "v1"
    assert registry.get_model("scratch").version == "v1"
    assert registry.names() == ["scratch"]


def test_publishing_a_second_version_does_not_change_what_is_served(tmp_path: Path) -> None:
    # Copying a checkpoint into the registry must not swap the model behind a
    # running API.
    registry = LocalModelRegistry(tmp_path / "registry")
    registry.publish(scratch_draft(), weights=weights(tmp_path))

    second = registry.publish(scratch_draft(), weights=weights(tmp_path, b"other"))

    assert second.version == "v2"
    assert registry.get_model("scratch").version == "v1"
    assert [model.version for model in registry.versions("scratch")] == ["v1", "v2"]


def test_activating_is_an_explicit_act(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "registry")
    registry.publish(scratch_draft(), weights=weights(tmp_path))
    registry.publish(scratch_draft(), weights=weights(tmp_path, b"other"), activate=True)

    assert registry.get_model("scratch").version == "v2"


def test_a_published_version_is_never_rewritten(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "registry")
    first = registry.publish(scratch_draft(), weights=weights(tmp_path))
    second = registry.publish(scratch_draft(), weights=weights(tmp_path, b"other"))

    assert first.checksum != second.checksum
    assert registry.read_version("scratch", "v1").checksum == first.checksum


def test_a_removed_version_does_not_free_its_identifier(tmp_path: Path) -> None:
    # A document already refers to v1. Handing the number to another set of
    # weights would silently change what that document points at.
    registry = LocalModelRegistry(tmp_path / "registry")
    registry.publish(scratch_draft(), weights=weights(tmp_path))
    registry.publish(scratch_draft(), weights=weights(tmp_path, b"other"))

    assert registry.next_version("scratch") == "v3"


def test_versions_are_ordered_by_number_not_by_name(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "registry")
    for index in range(11):
        registry.publish(scratch_draft(), weights=weights(tmp_path, f"w{index}".encode()))

    assert [model.version for model in registry.versions("scratch")][-2:] == ["v10", "v11"]


def test_the_baseline_is_published_without_a_file(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "registry")

    published = registry.publish(baseline_draft())

    assert published.artifact is None
    assert registry.resolve_artifact(published) is None


def test_a_checkpoint_without_its_weights_is_refused(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "registry")

    with pytest.raises(ValueError, match="requires a weights file"):
        registry.publish(scratch_draft())


def test_a_hub_version_carrying_weights_is_refused(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "registry")

    with pytest.raises(ValueError, match="carries no weights file"):
        registry.publish(baseline_draft(), weights=weights(tmp_path))


def test_missing_weights_are_reported_before_anything_is_written(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "registry")

    with pytest.raises(FileNotFoundError):
        registry.publish(scratch_draft(), weights=tmp_path / "absent.pt")

    assert registry.names() == []


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------


def test_the_weights_are_copied_into_the_registry(tmp_path: Path) -> None:
    # runs/ is scratch space the next execution overwrites.
    registry = LocalModelRegistry(tmp_path / "registry")
    source = weights(tmp_path)

    published = registry.publish(scratch_draft(), weights=source)
    source.unlink()

    path = registry.resolve_artifact(published)
    assert path is not None
    assert path.read_bytes() == b"weights"
    assert published.checksum == checksum_file(path)


def test_tampered_weights_are_refused(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "registry")
    published = registry.publish(scratch_draft(), weights=weights(tmp_path))

    path = registry.root / str(published.artifact)
    path.write_bytes(b"something else")

    with pytest.raises(ModelNotReadyError):
        registry.resolve_artifact(published)


def test_the_check_can_be_skipped_deliberately(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "registry")
    published = registry.publish(scratch_draft(), weights=weights(tmp_path))
    (registry.root / str(published.artifact)).write_bytes(b"something else")

    assert registry.resolve_artifact(published, verify=False) is not None


def test_a_version_pointing_at_a_missing_file_is_not_found(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "registry")
    published = registry.publish(scratch_draft(), weights=weights(tmp_path))
    (registry.root / str(published.artifact)).unlink()

    with pytest.raises(ModelNotFoundError):
        registry.resolve_artifact(published)


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------


def test_an_alias_resolves_to_the_exact_version_it_names(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "registry")
    registry.publish(scratch_draft(), weights=weights(tmp_path))
    registry.publish(scratch_draft(), weights=weights(tmp_path, b"other"))
    registry.promote("champion", "scratch", "v1")

    assert registry.get_model("champion").version == "v1"


def test_promoting_does_not_move_when_a_newer_version_appears(tmp_path: Path) -> None:
    # If champion meant "the newest", publishing would change production.
    registry = LocalModelRegistry(tmp_path / "registry")
    registry.publish(scratch_draft(), weights=weights(tmp_path))
    registry.promote("champion", "scratch", "v1")
    registry.publish(scratch_draft(), weights=weights(tmp_path, b"other"), activate=True)

    assert registry.get_model("champion").version == "v1"
    assert registry.get_model("scratch").version == "v2"


def test_an_alias_asked_for_with_a_version_is_refused(tmp_path: Path) -> None:
    # The pair states two answers to one question.
    registry = LocalModelRegistry(tmp_path / "registry")
    registry.publish(scratch_draft(), weights=weights(tmp_path))
    registry.promote("champion", "scratch", "v1")

    with pytest.raises(ModelNotFoundError):
        registry.get_model("champion", "v1")


def test_promoting_an_unknown_alias_is_refused(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "registry")
    registry.publish(scratch_draft(), weights=weights(tmp_path))

    with pytest.raises(ValueError, match="Unknown alias"):
        registry.promote("staging", "scratch", "v1")


def test_promoting_an_unpublished_version_is_refused(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "registry")
    registry.publish(scratch_draft(), weights=weights(tmp_path))

    with pytest.raises(ModelNotFoundError):
        registry.promote("champion", "scratch", "v9")


def test_activating_an_unpublished_version_is_refused(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "registry")
    registry.publish(scratch_draft(), weights=weights(tmp_path))

    with pytest.raises(ModelNotFoundError):
        registry.activate("scratch", "v9")


def test_activating_an_older_version_is_a_rollback(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "registry")
    registry.publish(scratch_draft(), weights=weights(tmp_path))
    registry.publish(scratch_draft(), weights=weights(tmp_path, b"other"), activate=True)

    registry.activate("scratch", "v1")

    assert registry.get_model("scratch").version == "v1"


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def test_listing_the_versions_of_an_unknown_name_is_not_found(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "registry")

    with pytest.raises(ModelNotFoundError):
        registry.versions("scratch")


def test_two_publications_racing_for_one_identifier_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two processes publishing at once would otherwise have the second write
    # into the directory of the first, replacing weights a document names.
    registry = LocalModelRegistry(tmp_path / "registry")
    registry.publish(scratch_draft(), weights=weights(tmp_path))
    monkeypatch.setattr(LocalModelRegistry, "next_version", lambda self, name: "v1")

    with pytest.raises(ValueError, match="already exists"):
        registry.publish(scratch_draft(), weights=weights(tmp_path, b"other"))


def test_a_corrupted_index_is_a_defect_not_an_empty_registry(tmp_path: Path) -> None:
    # Reading it as empty would silently unpublish every alias.
    registry = LocalModelRegistry(tmp_path / "registry")
    registry.publish(scratch_draft(), weights=weights(tmp_path))
    registry.index_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError):
        registry.read_index()


def test_the_two_models_of_section_20_live_side_by_side(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "registry")
    registry.publish(scratch_draft(), weights=weights(tmp_path))
    registry.publish(baseline_draft())
    registry.promote("champion", "pretrained", "v1")
    registry.promote("challenger", "scratch", "v1")

    assert registry.names() == ["pretrained", "scratch"]
    assert registry.get_model("champion").name == "pretrained"
    assert registry.get_model("challenger").name == "scratch"
