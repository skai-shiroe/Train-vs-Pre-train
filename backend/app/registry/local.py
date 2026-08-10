"""A registry backed by a directory, which is the one the project ships with.

The layout is deliberately readable without any tool::

    artifacts/registry/
    ├── index.json                  what each name serves, where each alias points
    ├── scratch/
    │   └── v1/
    │       ├── version.json        identity, provenance and scores
    │       └── weights.pt          the state dictionary, nothing else
    └── pretrained/
        └── v1/
            └── version.json        no file: the weights come from the hub

**A published version is never modified.** Publishing allocates the next
identifier and refuses to write into a directory that already exists. A version
is what a measurement points at, so changing its content would silently change
what a reported score refers to.

**Publishing does not change what is served.** A new version sits beside the
others until something promotes it. The exception is the first version of a
name, which becomes the default because there was nothing to displace. Without
that rule, copying a checkpoint into the registry would swap the model behind a
running API.

**The pointers live in one file, the versions on disk.** ``index.json`` holds
only what cannot be derived: the default version of each name and the target of
each alias. Listing the versions there too would create a second source of
truth, and a listing that disagrees with the directories is the kind of drift
nothing detects until a lookup fails.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from backend.app.core.errors import ModelNotFoundError, ModelNotReadyError
from backend.app.registry.base import ModelRegistry
from backend.app.registry.models import (
    ALIASES,
    DEFAULT_REGISTRY_ROOT,
    INDEX_FILE,
    VERSION_FILE,
    ModelDraft,
    ModelRef,
    ModelSource,
    ModelVersion,
    RegistryIndex,
)

#: Name of the weights file inside a version directory.
WEIGHTS_FILE = "weights.pt"

#: Layout of that file: a mapping holding a version marker and the state
#: dictionary under ``model``, and nothing else. The optimiser moments, the
#: scheduler position and the random generator states belong to a run that may
#: be resumed; a served model never resumes anything, and carrying them would
#: triple the size of every published version.
WEIGHTS_PAYLOAD_VERSION = 1
WEIGHTS_PAYLOAD_KEY = "model"

#: Bytes read at a time when checksumming. Large enough to be fast on a model
#: of a few hundred megabytes, small enough not to hold one in memory twice.
CHUNK_BYTES = 1024 * 1024

#: Matches a version directory and captures its number.
VERSION_DIRECTORY = re.compile(r"^v([1-9][0-9]*)$")


def checksum_file(path: Path) -> str:
    """Return the SHA-256 of a file.

    Args:
        path: File to read.

    Returns:
        The digest, in lowercase hexadecimal.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    """Return the current UTC time as a timestamp.

    Returns:
        An ISO 8601 string with second precision.
    """
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


class LocalModelRegistry(ModelRegistry):
    """Model registry stored under a directory of the filesystem."""

    def __init__(self, root: Path | str = DEFAULT_REGISTRY_ROOT) -> None:
        """Build the registry.

        Args:
            root: Directory holding the index and the model directories. It is
                created on the first publication, not here: reading an empty
                registry must not have the side effect of creating one.
        """
        self.root = Path(root)

    # -- Layout -----------------------------------------------------------

    @property
    def index_path(self) -> Path:
        """Return the path of the index.

        Returns:
            The path, whether or not the file exists yet.
        """
        return self.root / INDEX_FILE

    def version_directory(self, name: str, version: str) -> Path:
        """Return the directory of one version.

        Args:
            name: Registered model name.
            version: Version identifier.

        Returns:
            The directory, whether or not it exists yet.
        """
        return self.root / name / version

    # -- Index ------------------------------------------------------------

    def read_index(self) -> RegistryIndex:
        """Read the pointers of the registry.

        Returns:
            The index, empty when the registry holds none yet.

        Raises:
            ValueError: If the file exists but does not hold a valid index. A
                corrupted index is a defect, not an empty registry: treating it
                as empty would silently unpublish every alias.
        """
        if not self.index_path.is_file():
            return RegistryIndex()
        return RegistryIndex.model_validate_json(self.index_path.read_text(encoding="utf-8"))

    def write_index(self, index: RegistryIndex) -> Path:
        """Write the pointers of the registry.

        Args:
            index: The index to store.

        Returns:
            The path written.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(index.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return self.index_path

    # -- Reading ----------------------------------------------------------

    def names(self) -> list[str]:
        """List the registered model names.

        Returns:
            The names holding at least one version, sorted.
        """
        if not self.root.is_dir():
            return []
        found = [
            entry.name
            for entry in self.root.iterdir()
            if entry.is_dir()
            and any(VERSION_DIRECTORY.match(child.name) for child in entry.iterdir())
        ]
        return sorted(found)

    def version_numbers(self, name: str) -> list[int]:
        """List the version numbers published under one name.

        Args:
            name: Registered model name.

        Returns:
            The numbers, ascending. Empty when the name holds no version.
        """
        directory = self.root / name
        if not directory.is_dir():
            return []
        numbers = [
            int(match.group(1))
            for child in directory.iterdir()
            if child.is_dir() and (match := VERSION_DIRECTORY.match(child.name))
        ]
        return sorted(numbers)

    def read_version(self, name: str, version: str) -> ModelVersion:
        """Read one version document.

        Args:
            name: Registered model name.
            version: Version identifier.

        Returns:
            The version.

        Raises:
            ModelNotFoundError: If the version was never published.
        """
        path = self.version_directory(name, version) / VERSION_FILE
        if not path.is_file():
            raise ModelNotFoundError(details={"model": name, "version": version})
        return ModelVersion.model_validate_json(path.read_text(encoding="utf-8"))

    def versions(self, model_name: str) -> list[ModelVersion]:
        """List every published version of one model.

        Args:
            model_name: Registered model name.

        Returns:
            The versions, oldest first.

        Raises:
            ModelNotFoundError: If no model is registered under that name.
        """
        numbers = self.version_numbers(model_name)
        if not numbers:
            raise ModelNotFoundError(details={"model": model_name})
        return [self.read_version(model_name, f"v{number}") for number in numbers]

    def get_model(self, model_name: str, version: str | None = None) -> ModelVersion:
        """Resolve one version.

        Args:
            model_name: A registered model name, or one of the aliases of
                section 20.
            version: An exact version. Refused next to an alias: an alias
                already names a version, so the pair would state two answers to
                one question.

        Returns:
            The resolved version.

        Raises:
            ModelNotFoundError: If the name, the alias or the version is
                unknown, if an alias is asked for with a version, or if a name
                has no version to serve yet.
        """
        index = self.read_index()

        if model_name in ALIASES:
            if version is not None:
                raise ModelNotFoundError(details={"model": model_name, "version": version})
            target = index.aliases.get(model_name)
            if target is None:
                raise ModelNotFoundError(details={"alias": model_name})
            return self.read_version(target.name, target.version)

        resolved = version or index.current.get(model_name)
        if resolved is None:
            raise ModelNotFoundError(details={"model": model_name})
        return self.read_version(model_name, resolved)

    def resolve_artifact(self, model: ModelVersion, *, verify: bool = True) -> Path | None:
        """Return the local file holding the weights of a version.

        Args:
            model: The resolved version.
            verify: Whether the checksum recorded at publication is checked.

        Returns:
            The path of the weights, or ``None`` when they are pulled from
            Hugging Face at load time.

        Raises:
            ModelNotFoundError: If the artifact the version points at is
                missing.
            ModelNotReadyError: If the artifact does not match its checksum.
        """
        if model.source is not ModelSource.CHECKPOINT or model.artifact is None:
            return None

        path = self.root / model.artifact
        if not path.is_file():
            raise ModelNotFoundError(details={"model": model.name, "version": model.version})

        if verify and model.checksum is not None and checksum_file(path) != model.checksum:
            raise ModelNotReadyError(
                details={
                    "model": model.name,
                    "version": model.version,
                    "reason": "checksum mismatch",
                }
            )
        return path

    # -- Writing ----------------------------------------------------------

    def next_version(self, name: str) -> str:
        """Return the identifier the next publication under a name receives.

        Args:
            name: Registered model name.

        Returns:
            ``v1`` for a name that holds nothing, and the successor of the
            highest number otherwise. The highest, not the count: a version
            removed by hand must not free an identifier a document already
            refers to.
        """
        numbers = self.version_numbers(name)
        return f"v{max(numbers) + 1}" if numbers else "v1"

    def publish(
        self,
        draft: ModelDraft,
        *,
        weights: Path | None = None,
        activate: bool = False,
        created_at: str | None = None,
    ) -> ModelVersion:
        """Store a new version of a model.

        Args:
            draft: What the caller read out of the run record.
            weights: File holding the state dictionary, copied into the
                registry. Required for a checkpoint source, refused otherwise.
            activate: Whether this version becomes the one the name serves. The
                first version of a name activates whatever this says, because
                a name that serves nothing is a name the API cannot answer for.
            created_at: Publication timestamp. Defaults to now.

        Returns:
            The stored version.

        Raises:
            FileNotFoundError: If the weights file does not exist.
            ValueError: If the source and the weights disagree, or if the
                version directory already exists.
        """
        expects_file = draft.source is ModelSource.CHECKPOINT
        if expects_file and weights is None:
            raise ValueError(f"Publishing {draft.name!r} as a checkpoint requires a weights file.")
        if not expects_file and weights is not None:
            raise ValueError(
                f"Publishing {draft.name!r} from {draft.source.value} carries no weights file: "
                "the version records where they are pulled from, it does not copy them."
            )
        if weights is not None and not weights.is_file():
            raise FileNotFoundError(f"Weights not found: {weights}")

        version = self.next_version(draft.name)
        directory = self.version_directory(draft.name, version)
        if directory.exists():
            raise ValueError(f"Version directory already exists: {directory}")
        directory.mkdir(parents=True)

        artifact: str | None = None
        checksum: str | None = None
        if weights is not None:
            destination = directory / WEIGHTS_FILE
            shutil.copyfile(weights, destination)
            artifact = f"{draft.name}/{version}/{WEIGHTS_FILE}"
            checksum = checksum_file(destination)

        stored = draft.stamp(
            version=version,
            created_at=created_at or utc_now(),
            artifact=artifact,
            checksum=checksum,
        )
        (directory / VERSION_FILE).write_text(
            json.dumps(stored.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        index = self.read_index()
        if activate or draft.name not in index.current:
            self.write_index(
                index.model_copy(update={"current": {**index.current, draft.name: version}})
            )
        return stored

    def activate(self, name: str, version: str) -> ModelVersion:
        """Make one version the one its name serves by default.

        Args:
            name: Registered model name.
            version: Version identifier.

        Returns:
            The activated version.

        Raises:
            ModelNotFoundError: If that version was never published.
        """
        model = self.read_version(name, version)
        index = self.read_index()
        self.write_index(index.model_copy(update={"current": {**index.current, name: version}}))
        return model

    def promote(self, alias: str, name: str, version: str) -> ModelVersion:
        """Point a deployment alias at one exact version.

        Args:
            alias: One of the aliases of section 20.
            name: Registered model name.
            version: Version identifier.

        Returns:
            The promoted version.

        Raises:
            ValueError: If the alias is not one of :data:`ALIASES`.
            ModelNotFoundError: If that version was never published.
        """
        if alias not in ALIASES:
            raise ValueError(f"Unknown alias {alias!r}. Available: {', '.join(ALIASES)}.")

        model = self.read_version(name, version)
        index = self.read_index()
        aliases = {**index.aliases, alias: ModelRef(name=name, version=version)}
        self.write_index(index.model_copy(update={"aliases": aliases}))
        return model
