"""Turning a registered version into a network that answers requests.

This is the loader of section 27. It reads a :class:`ModelVersion`, rebuilds the
architecture the run used, puts the published weights into it and hands back a
:class:`LoadedModel`. It is the only module of the backend that builds a torch
model, and it never chooses which version to build: that decision belongs to the
registry.

**The architecture is revalidated, not reinterpreted.** ``version.architecture``
holds the ``model`` block of the experiment file, copied at publication. Parsing
it back with :class:`src.experiments.config.ScratchModelConfig` and
:class:`src.experiments.config.PretrainedModelConfig` means the API rebuilds the
network from the very class the training built it from. A second reading of the
same mapping, written by hand here, would drift the day a field is added, and
the symptom would be a model that loads and answers badly.

**A version that cannot be loaded is not ready, it is not a failed request.**
Weights that do not fit their architecture, a baseline that no longer exists, a
hub that cannot be reached: none of them is caused by the caller, so each one
raises :class:`ModelNotReadyError` and reaches the client as 503 rather than
500.

**Nothing here downloads what a version did not pin.** A fine tuned checkpoint
carries its own weights but its architecture still comes from the hub, which is
why section 20 refuses to publish an unpinned revision. The rule is enforced at
publication; this module simply passes the pinned revision on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from pydantic import ValidationError

from backend.app.core.errors import ModelNotReadyError
from backend.app.inference.base import LoadedModel, Summarizer
from backend.app.registry.local import WEIGHTS_PAYLOAD_KEY, WEIGHTS_PAYLOAD_VERSION
from backend.app.registry.models import ModelVersion
from src.data.config import TokenizerConfig
from src.data.tokenize import build_tokenizer
from src.experiments.config import PretrainedModelConfig, ScratchModelConfig
from src.models.pretrained.base import BaselineConfig, PretrainedSummarizer
from src.models.pretrained.factory import get_baseline_class
from src.models.scratch.summarizer import ScratchSummarizer
from src.models.scratch.transformer import ScratchTransformer

#: Value of the ``type`` discriminator of the ``model`` block, per section 13.
SCRATCH_TYPE = "scratch"
PRETRAINED_TYPE = "pretrained"


def not_ready(version: ModelVersion, reason: str) -> ModelNotReadyError:
    """Build the error raised when a published version cannot be loaded.

    Args:
        version: The version being loaded.
        reason: What stopped the load, in one sentence, without any path or
            stack trace: it reaches the client.

    Returns:
        The error to raise.
    """
    return ModelNotReadyError(
        f"Le modele {version.label} n'a pas pu etre charge : {reason}",
        details={"model": version.name, "version": version.version, "reason": reason},
    )


def read_weights(path: Path) -> dict[str, Any]:
    """Read the state dictionary published under a version.

    The payload is the one :mod:`backend.app.registry.local` describes: a
    version marker and the weights, nothing else. Reading stays on
    ``weights_only=True``, which is what makes a published artifact safe to load
    without trusting whoever wrote it.

    Args:
        path: The weights file inside the registry.

    Returns:
        The state dictionary.

    Raises:
        ValueError: If the file was written by another payload version, or does
            not hold a state dictionary at all.
    """
    # nosec B614 - the finding asks for weights_only, which is set. The rule
    # flags every torch.load call, so the exception is on the false positive.
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)  # nosec B614

    stored = payload.get("version")
    if stored != WEIGHTS_PAYLOAD_VERSION:
        raise ValueError(
            f"les poids ont ete ecrits par la version de payload {stored!r}, "
            f"cette instance lit la version {WEIGHTS_PAYLOAD_VERSION}"
        )
    weights = payload.get(WEIGHTS_PAYLOAD_KEY)
    if not isinstance(weights, dict):
        raise ValueError("le fichier de poids ne porte pas de dictionnaire d'etat")
    return weights


def parse_architecture(version: ModelVersion) -> ScratchModelConfig | PretrainedModelConfig:
    """Rebuild the model block the run was configured with.

    Args:
        version: The resolved version.

    Returns:
        The validated block, discriminated on its ``type`` field.

    Raises:
        ModelNotReadyError: If the block is absent, carries an unknown type or
            no longer validates against the configuration model.
    """
    block = version.architecture
    declared = block.get("type")

    try:
        if declared == SCRATCH_TYPE:
            return ScratchModelConfig.model_validate(block)
        if declared == PRETRAINED_TYPE:
            return PretrainedModelConfig.model_validate(block)
    except ValidationError as error:
        raise not_ready(
            version, f"son bloc model est invalide ({error.error_count()} champs)"
        ) from error

    raise not_ready(version, f"son bloc model declare le type {declared!r}, inconnu")


def parse_tokenizer(version: ModelVersion) -> TokenizerConfig:
    """Rebuild the tokeniser settings the corpus was encoded with.

    Args:
        version: The resolved version.

    Returns:
        The validated tokeniser block.

    Raises:
        ModelNotReadyError: If the block is absent or invalid. It is what makes
            the API truncate documents exactly where the training truncated
            them, so a version without it cannot be served at all.
    """
    try:
        return TokenizerConfig.model_validate(version.tokenizer)
    except ValidationError as error:
        raise not_ready(
            version, f"son bloc tokenizer est invalide ({error.error_count()} champs)"
        ) from error


def load_scratch(
    version: ModelVersion,
    block: ScratchModelConfig,
    tokenizer_config: TokenizerConfig,
    weights: Path | None,
) -> ScratchSummarizer:
    """Rebuild the from scratch Transformer and put the published weights in it.

    The vocabulary size is read from the tokeniser rather than from the version
    document, exactly as the experiment runner sized the embedding table. A
    number stored at publication could disagree with the tokeniser the API
    loads, and the mismatch would surface as a shape error at best.

    Args:
        version: The resolved version, used for the error messages.
        block: The validated architecture block.
        tokenizer_config: The tokeniser block of the corpus.
        weights: The published state dictionary.

    Returns:
        The summariser, marked as trained.

    Raises:
        ModelNotReadyError: If the version carries no weights, if they were
            written by another payload version, or if they do not fit the
            architecture the block describes.
    """
    if weights is None:
        raise not_ready(version, "un modele from scratch n'existe pas hors de ses poids publies")

    tokenizer = build_tokenizer(tokenizer_config.hf_id)
    architecture = block.architecture(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = ScratchTransformer(architecture)
    try:
        model.load_state_dict(read_weights(weights))
    except (ValueError, RuntimeError) as error:
        raise not_ready(
            version, f"ses poids ne correspondent pas a son architecture ({error})"
        ) from error

    return ScratchSummarizer(model, tokenizer, tokenizer_config, trained=True)


def load_pretrained(
    version: ModelVersion,
    block: PretrainedModelConfig,
    tokenizer_config: TokenizerConfig,
    weights: Path | None,
) -> PretrainedSummarizer:
    """Rebuild the pretrained baseline, fine tuned or zero shot.

    Args:
        version: The resolved version.
        block: The validated architecture block.
        tokenizer_config: The tokeniser block of the corpus.
        weights: The published state dictionary, or ``None`` for a version
            whose weights are pulled from the hub.

    Returns:
        The summariser, marked as fine tuned when weights were published for it.

    Raises:
        ModelNotReadyError: If the baseline is no longer registered, if the
            weights do not fit it, or if the hub cannot be reached.
    """
    config = BaselineConfig.from_tokenizer_config(
        tokenizer_config,
        hf_id=version.hf_id or block.hf_id,
        revision=version.revision or block.revision,
    )

    try:
        baseline = get_baseline_class(block.baseline)
    except KeyError as error:
        raise not_ready(version, f"sa baseline n'est plus enregistree ({error})") from error

    try:
        if weights is None:
            return baseline.from_pretrained(config, fine_tuned=False)
        return baseline.from_checkpoint(config, read_weights(weights))
    except (ValueError, RuntimeError) as error:
        raise not_ready(
            version, f"ses poids ne correspondent pas a son architecture ({error})"
        ) from error
    except OSError as error:
        raise not_ready(version, f"l'architecture n'a pas pu etre recuperee ({error})") from error


def load_version(
    version: ModelVersion, weights: Path | None, *, device: torch.device
) -> LoadedModel:
    """Load one published version onto a device.

    Args:
        version: The version the registry resolved.
        weights: The file the registry resolved for it, checksum already
            verified, or ``None`` when the weights come from the hub.
        device: Where the weights are moved once built.

    Returns:
        The loaded model, bound to the document that describes it.

    Raises:
        ModelNotReadyError: If the version cannot be turned into a network. The
            cause is on the deployment side, never on the caller side.
    """
    tokenizer_config = parse_tokenizer(version)
    block = parse_architecture(version)

    summarizer: Summarizer
    if isinstance(block, ScratchModelConfig):
        summarizer = load_scratch(version, block, tokenizer_config, weights).to(device)
    else:
        summarizer = load_pretrained(version, block, tokenizer_config, weights).to(device)

    return LoadedModel(
        version=version,
        summarizer=summarizer,
        device=device,
        token_budget=tokenizer_config.max_target_tokens,
    )
