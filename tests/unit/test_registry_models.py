"""Unit tests of what a registered version must state about its weights.

The property under test is that a version identifies exactly one set of weights.
Every rule below exists because breaking it produces a registry entry that looks
complete and points at something that can change.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.registry.models import (
    ALIASES,
    ModelDraft,
    ModelRef,
    ModelSource,
    ModelVersion,
    RegistryIndex,
)

pytestmark = pytest.mark.unit


def version(**overrides: object) -> ModelVersion:
    """Return a valid checkpoint version, with fields replaced."""
    fields: dict[str, object] = {
        "name": "scratch",
        "version": "v1",
        "source": ModelSource.CHECKPOINT,
        "artifact": "scratch/v1/weights.pt",
        "checksum": "a" * 64,
        "created_at": "2026-08-08T12:00:00+00:00",
    }
    fields.update(overrides)
    return ModelVersion(**fields)  # type: ignore[arg-type]


def test_a_checkpoint_version_states_its_file_and_its_checksum() -> None:
    stored = version()

    assert stored.label == "scratch:v1"
    assert stored.ref == ModelRef(name="scratch", version="v1")


def test_a_checkpoint_without_a_checksum_is_refused() -> None:
    # Without it nothing can state that the weights served are the weights
    # that were measured.
    with pytest.raises(ValidationError, match="checksum"):
        version(checksum=None)


def test_a_checkpoint_without_a_file_is_refused() -> None:
    with pytest.raises(ValidationError, match="artifact path"):
        version(artifact=None)


def test_a_hub_version_without_an_identifier_is_refused() -> None:
    with pytest.raises(ValidationError, match="identifier"):
        version(source=ModelSource.HUGGING_FACE, artifact=None, checksum=None)


def test_an_unpinned_revision_is_refused() -> None:
    # Section 14. An unpinned identifier resolves to whatever the hub serves
    # that day, so the weights would change without the version changing.
    with pytest.raises(ValidationError, match="without pinning a revision"):
        version(
            name="pretrained",
            source=ModelSource.HUGGING_FACE,
            artifact=None,
            checksum=None,
            hf_id="t5-small",
            revision=None,
        )


def test_a_pinned_revision_is_accepted() -> None:
    stored = version(
        name="pretrained",
        source=ModelSource.HUGGING_FACE,
        artifact=None,
        checksum=None,
        hf_id="t5-small",
        revision="df1b051c49625cf57a3d0d8d3863ed4d13564fe4",  # pragma: allowlist secret
    )

    assert stored.revision is not None


def test_a_fine_tuned_checkpoint_also_needs_its_revision() -> None:
    # The checkpoint holds our weights, but the architecture they are loaded
    # into is downloaded from the same unpinned identifier.
    with pytest.raises(ValidationError, match="without pinning a revision"):
        version(name="pretrained", hf_id="t5-small")


@pytest.mark.parametrize("alias", ALIASES)
def test_a_model_may_not_be_named_after_an_alias(alias: str) -> None:
    with pytest.raises(ValidationError, match="deployment alias"):
        version(name=alias)


@pytest.mark.parametrize("bad", ["v0", "1", "latest", "v01"])
def test_a_version_identifier_is_a_number(bad: str) -> None:
    with pytest.raises(ValidationError):
        version(version=bad)


# ---------------------------------------------------------------------------
# Drafts
# ---------------------------------------------------------------------------


def test_a_draft_is_refused_before_anything_is_written() -> None:
    # The same rule as on the stored version, checked earlier so a rejected
    # publication does not leave half a version directory behind.
    with pytest.raises(ValidationError, match="without pinning a revision"):
        ModelDraft(name="pretrained", source=ModelSource.HUGGING_FACE, hf_id="t5-small")


def test_a_draft_becomes_a_version_when_the_registry_stamps_it() -> None:
    draft = ModelDraft(
        name="scratch",
        source=ModelSource.CHECKPOINT,
        architecture={"type": "scratch", "d_model": 256},
        metrics={"rougeL_f": 0.15},
        experiment="scratch_100",
    )

    stored = draft.stamp(
        version="v3",
        created_at="2026-08-08T12:00:00+00:00",
        artifact="scratch/v3/weights.pt",
        checksum="b" * 64,
    )

    assert stored.version == "v3"
    assert stored.architecture["d_model"] == 256
    assert stored.metrics["rougeL_f"] == pytest.approx(0.15)
    assert stored.experiment == "scratch_100"


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def test_an_empty_index_is_valid() -> None:
    assert RegistryIndex().current == {}


def test_an_unknown_alias_is_refused() -> None:
    with pytest.raises(ValidationError, match="Unknown aliases"):
        RegistryIndex(aliases={"staging": ModelRef(name="scratch", version="v1")})


def test_the_two_aliases_of_section_20_are_accepted() -> None:
    index = RegistryIndex(
        current={"scratch": "v2"},
        aliases={
            "champion": ModelRef(name="scratch", version="v2"),
            "challenger": ModelRef(name="pretrained", version="v1"),
        },
    )

    assert set(index.aliases) == set(ALIASES)
