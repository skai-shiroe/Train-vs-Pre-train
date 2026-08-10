"""What a registered model version is, and what makes one valid.

Section 20 forbids the backend from depending on hard coded file paths. A model
version is therefore described by a document, not by a location: the API asks
the registry for ``champion`` and receives an identity, a place to read the
weights from and the measurement that justified publishing them.

**A version identifies weights exactly, or it is not a version.** A local
checkpoint is identified by its path and by the checksum computed when it was
published. A model pulled from Hugging Face is identified by its identifier and
its revision. Section 14 tolerates an unpinned revision outside a reported
experiment, and a registered version is the opposite of that: it is what the API
serves. An identifier without a revision resolves to whatever the hub publishes
that day, so it is refused here rather than discovered later.

**The measurement travels with the weights.** ``experiment``,
``dataset_version``, ``git_commit`` and the ROUGE scores are copied from the run
record at publication. Without them a registry entry is an anonymous file, and
the question the API will be asked, which model produced this summary, has no
answer that can be traced back to a run.

**The registry stores identity, never a loaded model.** Nothing here imports
torch. Resolving ``champion`` to a version costs a file read, so the API can
decide what to load before paying for it, per section 27.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Root of the local registry. Mirrored by ``SYNTRA_REGISTRY_ROOT`` so the
#: training side and the API resolve the same directory without sharing a
#: settings object.
DEFAULT_REGISTRY_ROOT = Path("artifacts") / "registry"

#: File describing one version, inside its version directory.
VERSION_FILE = "version.json"

#: File holding the pointers: what each name serves, and where the aliases
#: point. Sits at the root of the registry.
INDEX_FILE = "index.json"

#: Deployment aliases of section 20. They are names a client may ask for, so no
#: model may be registered under one of them.
CHAMPION = "champion"
CHALLENGER = "challenger"
ALIASES: tuple[str, ...] = (CHAMPION, CHALLENGER)

#: Names the specification expects to find in the registry, kept as constants so
#: the publication command and the API agree on their spelling.
SCRATCH = "scratch"
PRETRAINED = "pretrained"

#: Shape of a version identifier. Ordering is numeric, so a lexicographic sort
#: is never used to find the newest one.
VERSION_PATTERN = r"^v[1-9][0-9]*$"

#: Shape of a model name. Same restriction as an experiment name: it becomes a
#: directory and a value in a query string.
NAME_PATTERN = r"^[a-z][a-z0-9_]*$"


def require_registrable_name(name: str) -> None:
    """Refuse a model name that collides with a deployment alias.

    Args:
        name: The name a model would be registered under.

    Raises:
        ValueError: If the name is one of :data:`ALIASES`.
    """
    if name in ALIASES:
        raise ValueError(
            f"{name!r} is a deployment alias, not a model name. A model registered under it "
            f"would make a lookup ambiguous. Aliases: {', '.join(ALIASES)}."
        )


def require_pinned_revision(label: str, hf_id: str | None, revision: str | None) -> None:
    """Refuse a Hugging Face identifier that is not pinned to a revision.

    Args:
        label: How the offending version is named in the message.
        hf_id: The identifier, when the version names one.
        revision: The commit or tag pinned for it.

    Raises:
        ValueError: If an identifier is present without a revision.
    """
    if hf_id and not revision:
        raise ValueError(
            f"Version {label} names {hf_id!r} without pinning a revision. An unpinned "
            "identifier resolves to whatever the hub serves that day, so the weights behind "
            "this version would change without the version changing. Set model.revision in "
            "the experiment configuration, per section 14."
        )


class ModelSource(StrEnum):
    """Where the weights of a version come from."""

    #: A file published into the registry, produced by a training run.
    CHECKPOINT = "checkpoint"

    #: Weights downloaded from the Hugging Face hub at load time, which is what
    #: the zero shot baseline is.
    HUGGING_FACE = "huggingface"


class ModelRef(BaseModel):
    """A pointer to one exact version.

    Attributes:
        name: Registered model name.
        version: Version identifier.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=NAME_PATTERN, description="Registered model name.")
    version: str = Field(pattern=VERSION_PATTERN, description="Version identifier.")


class ModelVersion(BaseModel):
    """One published version of one model.

    Attributes:
        name: Registered model name, for example ``scratch``.
        version: Version identifier, assigned at publication.
        source: Whether the weights are a published file or a hub download.
        artifact: Path of the weights, relative to the registry root. Present
            for a checkpoint source only.
        checksum: SHA-256 of that file, computed at publication and checked
            before the weights are served.
        hf_id: Hugging Face identifier, for the baseline and for the
            architecture a fine tuned checkpoint is loaded into.
        revision: Commit or tag pinned for that identifier.
        architecture: The model block of the experiment configuration, which is
            what a loader needs to rebuild the network the weights fit.
        tokenizer: The tokeniser block of the data pipeline configuration.
            Storing it here is what lets the API tokenise exactly as the
            training did without reading a configuration file of the ML side.
        experiment: Name of the run that produced the weights.
        dataset_version: Checksum of the corpus that run trained on.
        git_commit: Commit the run was started from.
        metrics: Scores copied from the run record.
        created_at: Publication timestamp, in UTC.
        description: One line stating what this version is.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=NAME_PATTERN, description="Registered model name.")
    version: str = Field(pattern=VERSION_PATTERN, description="Version identifier.")
    source: ModelSource = Field(description="Where the weights come from.")

    artifact: str | None = Field(default=None, description="Weights, relative to the root.")
    checksum: str | None = Field(default=None, description="SHA-256 of the artifact.")
    hf_id: str | None = Field(default=None, description="Hugging Face identifier.")
    revision: str | None = Field(default=None, description="Pinned commit or tag.")

    architecture: dict[str, Any] = Field(default_factory=dict, description="Model block.")
    tokenizer: dict[str, Any] = Field(default_factory=dict, description="Tokeniser block.")

    experiment: str | None = Field(default=None, description="Run that produced the weights.")
    dataset_version: str | None = Field(default=None, description="Corpus checksum.")
    git_commit: str | None = Field(default=None, description="Commit of the run.")
    metrics: dict[str, float] = Field(default_factory=dict, description="Scores of the run.")
    created_at: str = Field(description="Publication timestamp, in UTC.")
    description: str = Field(default="", description="One line describing the version.")

    @model_validator(mode="after")
    def _check_identification(self) -> ModelVersion:
        """Refuse a version that does not identify its weights exactly.

        Returns:
            The validated version.

        Raises:
            ValueError: If a published file carries no path or no checksum, if
                a hub model carries no identifier, if a Hugging Face identifier
                is not pinned to a revision, or if the name is one of the
                deployment aliases.
        """
        require_registrable_name(self.name)

        if self.source is ModelSource.CHECKPOINT and not (self.artifact and self.checksum):
            raise ValueError(
                "A checkpoint version must carry both its artifact path and its checksum. "
                "Without the checksum nothing can state that the weights served are the "
                "weights that were measured."
            )

        if self.source is ModelSource.HUGGING_FACE and not self.hf_id:
            raise ValueError("A Hugging Face version must carry the identifier it is pulled from.")

        require_pinned_revision(self.label, self.hf_id, self.revision)
        return self

    @property
    def ref(self) -> ModelRef:
        """Return the pointer to this version.

        Returns:
            The reference an alias or a default pointer stores.
        """
        return ModelRef(name=self.name, version=self.version)

    @property
    def label(self) -> str:
        """Return the human readable identifier of this version.

        Returns:
            The ``name:version`` pair used in messages and logs.
        """
        return f"{self.name}:{self.version}"


class ModelDraft(BaseModel):
    """A version about to be published, before the registry stamps it.

    The registry assigns the version identifier, computes the checksum and sets
    the publication timestamp. Everything else is what the caller read out of a
    run record, which is why it is gathered here rather than passed as a dozen
    arguments.

    Attributes:
        name: Registered model name.
        source: Where the weights come from.
        hf_id: Hugging Face identifier, when one is involved.
        revision: Commit or tag pinned for that identifier.
        architecture: The model block of the experiment configuration.
        tokenizer: The tokeniser block of the data pipeline configuration.
        experiment: Name of the run that produced the weights.
        dataset_version: Checksum of the corpus that run trained on.
        git_commit: Commit the run was started from.
        metrics: Scores copied from the run record.
        description: One line stating what this version is.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=NAME_PATTERN, description="Registered model name.")
    source: ModelSource = Field(description="Where the weights come from.")
    hf_id: str | None = Field(default=None, description="Hugging Face identifier.")
    revision: str | None = Field(default=None, description="Pinned commit or tag.")
    architecture: dict[str, Any] = Field(default_factory=dict, description="Model block.")
    tokenizer: dict[str, Any] = Field(default_factory=dict, description="Tokeniser block.")
    experiment: str | None = Field(default=None, description="Run that produced the weights.")
    dataset_version: str | None = Field(default=None, description="Corpus checksum.")
    git_commit: str | None = Field(default=None, description="Commit of the run.")
    metrics: dict[str, float] = Field(default_factory=dict, description="Scores of the run.")
    description: str = Field(default="", description="One line describing the version.")

    @model_validator(mode="after")
    def _check_identification(self) -> ModelDraft:
        """Refuse a draft the registry could not store.

        The two rules are checked here as well as on :class:`ModelVersion`, and
        for a reason worth the repetition: a draft is validated before any file
        is copied, so a publication that would be rejected fails without having
        written half a version directory first.

        Returns:
            The validated draft.

        Raises:
            ValueError: If the name is a deployment alias, or if a Hugging Face
                identifier is not pinned to a revision.
        """
        require_registrable_name(self.name)
        require_pinned_revision(self.name, self.hf_id, self.revision)
        return self

    def stamp(
        self,
        *,
        version: str,
        created_at: str,
        artifact: str | None,
        checksum: str | None,
    ) -> ModelVersion:
        """Turn the draft into the version the registry stores.

        Args:
            version: Identifier assigned by the registry.
            created_at: Publication timestamp, in UTC.
            artifact: Path of the weights, relative to the registry root.
            checksum: SHA-256 of that file.

        Returns:
            The validated version.

        Raises:
            ValueError: If the result does not identify its weights exactly.
        """
        return ModelVersion(
            name=self.name,
            version=version,
            source=self.source,
            artifact=artifact,
            checksum=checksum,
            hf_id=self.hf_id,
            revision=self.revision,
            architecture=dict(self.architecture),
            tokenizer=dict(self.tokenizer),
            experiment=self.experiment,
            dataset_version=self.dataset_version,
            git_commit=self.git_commit,
            metrics=dict(self.metrics),
            created_at=created_at,
            description=self.description,
        )


class RegistryIndex(BaseModel):
    """The pointers of a registry: what is served, and under which alias.

    The published versions themselves are not listed here. They are directories
    on disk, and a listing that could disagree with them would be a second
    source of truth. What cannot be read from disk is which version a name
    serves and where an alias points, which is exactly what this holds.

    Attributes:
        current: Version served for each model name when no version is asked
            for.
        aliases: Deployment aliases, each pointing at one exact version.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    current: dict[str, str] = Field(default_factory=dict, description="Default version per name.")
    aliases: dict[str, ModelRef] = Field(default_factory=dict, description="Alias targets.")

    @model_validator(mode="after")
    def _check_aliases(self) -> RegistryIndex:
        """Refuse an alias outside the set of section 20.

        Returns:
            The validated index.

        Raises:
            ValueError: If an unknown alias is declared.
        """
        unknown = sorted(set(self.aliases) - set(ALIASES))
        if unknown:
            raise ValueError(f"Unknown aliases: {unknown}. Available: {', '.join(ALIASES)}.")
        return self
