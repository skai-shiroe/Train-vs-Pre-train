"""Unit tests for the model loader of section 27."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch

from backend.app.core.errors import ErrorCode, ModelNotReadyError
from backend.app.inference import loader as loader_module
from backend.app.inference.loader import (
    load_pretrained,
    load_scratch,
    load_version,
    parse_architecture,
    parse_tokenizer,
    read_weights,
)
from backend.app.registry.models import ModelSource, ModelVersion
from backend.tests.conftest import FakeTokenizer, build_version, write_weights
from src.data.config import TokenizerConfig
from src.experiments.config import PretrainedModelConfig, ScratchModelConfig
from src.models.scratch.summarizer import ScratchSummarizer
from src.models.scratch.transformer import ScratchTransformer

CPU = torch.device("cpu")


@pytest.fixture
def offline_tokenizer(monkeypatch: pytest.MonkeyPatch, fake_tokenizer: FakeTokenizer) -> Any:
    """Make the loader build models without downloading a vocabulary."""
    monkeypatch.setattr(loader_module, "build_tokenizer", lambda hf_id: fake_tokenizer)
    return fake_tokenizer


def build_tiny_transformer(tokenizer: FakeTokenizer, **overrides: Any) -> ScratchTransformer:
    """Build the Transformer a tiny version describes."""
    block = ScratchModelConfig.model_validate({**build_version().architecture, **overrides})
    return ScratchTransformer(
        block.architecture(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    )


def publish_weights(tmp_path: Path, tokenizer: FakeTokenizer, **overrides: Any) -> Path:
    """Write the weights of a tiny Transformer where a version points."""
    model = build_tiny_transformer(tokenizer, **overrides)
    return write_weights(tmp_path / "weights.pt", dict(model.state_dict()))


# -- Reading the payload ----------------------------------------------------


@pytest.mark.unit
def test_weights_are_read_back(tmp_path: Path, fake_tokenizer: FakeTokenizer) -> None:
    path = publish_weights(tmp_path, fake_tokenizer)

    state = read_weights(path)

    assert "encoder.embedding.embedding.weight" in " ".join(state)


@pytest.mark.unit
def test_a_payload_of_another_version_is_refused(tmp_path: Path) -> None:
    path = write_weights(tmp_path / "weights.pt", {"a": torch.zeros(1)}, version=99)

    with pytest.raises(ValueError, match="version de payload"):
        read_weights(path)


@pytest.mark.unit
def test_a_payload_without_a_state_dictionary_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "weights.pt"
    torch.save({"version": 1, "model": torch.zeros(1)}, path)

    with pytest.raises(ValueError, match="dictionnaire d'etat"):
        read_weights(path)


# -- Reading the document ---------------------------------------------------


@pytest.mark.unit
def test_the_architecture_block_is_revalidated() -> None:
    block = parse_architecture(build_version())

    assert isinstance(block, ScratchModelConfig)
    assert block.d_model == 16


@pytest.mark.unit
def test_a_pretrained_block_selects_the_other_branch() -> None:
    version = build_version(
        name="pretrained",
        source=ModelSource.HUGGING_FACE,
        artifact=None,
        checksum=None,
        hf_id="t5-small",
        revision="d769bba",
        architecture={"type": "pretrained", "baseline": "t5", "mode": "zero_shot"},
    )

    assert isinstance(parse_architecture(version), PretrainedModelConfig)


@pytest.mark.unit
@pytest.mark.parametrize("block", [{}, {"type": "unknown"}])
def test_a_version_without_a_known_type_is_not_ready(block: dict[str, Any]) -> None:
    with pytest.raises(ModelNotReadyError) as raised:
        parse_architecture(build_version(architecture=block))

    assert raised.value.code is ErrorCode.MODEL_NOT_READY
    assert raised.value.status_code == 503


@pytest.mark.unit
def test_an_invalid_architecture_block_is_not_ready() -> None:
    with pytest.raises(ModelNotReadyError, match="bloc model est invalide"):
        parse_architecture(build_version(architecture={"type": "scratch", "d_model": -1}))


@pytest.mark.unit
def test_the_tokenizer_block_is_revalidated() -> None:
    assert parse_tokenizer(build_version()).source_prefix == "summarize: "


@pytest.mark.unit
def test_an_invalid_tokenizer_block_is_not_ready() -> None:
    with pytest.raises(ModelNotReadyError, match="bloc tokenizer est invalide"):
        parse_tokenizer(build_version(tokenizer={"hf_id": "t5-small"}))


# -- Building the from scratch model ----------------------------------------


@pytest.mark.unit
def test_a_from_scratch_version_is_rebuilt_and_marked_trained(
    tmp_path: Path, offline_tokenizer: FakeTokenizer
) -> None:
    version = build_version()
    weights = publish_weights(tmp_path, offline_tokenizer)

    loaded = load_version(version, weights, device=CPU)

    assert isinstance(loaded.summarizer, ScratchSummarizer)
    assert loaded.summarizer.trained is True
    assert loaded.label == "scratch:v1"
    assert loaded.device == CPU


@pytest.mark.unit
def test_a_from_scratch_version_without_weights_is_not_ready(
    offline_tokenizer: FakeTokenizer,
) -> None:
    with pytest.raises(ModelNotReadyError, match="hors de ses poids publies"):
        load_version(build_version(), None, device=CPU)


@pytest.mark.unit
def test_weights_that_do_not_fit_the_architecture_are_not_ready(
    tmp_path: Path, offline_tokenizer: FakeTokenizer
) -> None:
    # Published under one width, described by the document as another one.
    weights = publish_weights(tmp_path, offline_tokenizer, d_model=32, d_ff=64)

    with pytest.raises(ModelNotReadyError, match="ne correspondent pas"):
        load_version(build_version(), weights, device=CPU)


@pytest.mark.unit
def test_the_vocabulary_comes_from_the_tokeniser(
    tmp_path: Path, offline_tokenizer: FakeTokenizer
) -> None:
    loaded = load_version(build_version(), publish_weights(tmp_path, offline_tokenizer), device=CPU)

    assert isinstance(loaded.summarizer, ScratchSummarizer)
    assert loaded.summarizer.model.config.vocab_size == len(offline_tokenizer)


@pytest.mark.unit
def test_a_loaded_model_answers(tmp_path: Path, offline_tokenizer: FakeTokenizer) -> None:
    loaded = load_version(build_version(), publish_weights(tmp_path, offline_tokenizer), device=CPU)

    summary = loaded.summarize("un document de test", None)

    assert isinstance(summary, str)
    assert loaded.describe()["device"] == "cpu"


# -- Building the pretrained baseline ---------------------------------------


class FakeBaseline:
    """A baseline that records how it was built, without touching the hub."""

    built: list[tuple[str, bool]] = []

    def __init__(self, *, fine_tuned: bool) -> None:
        self.fine_tuned = fine_tuned
        self.moved_to: torch.device | None = None

    @classmethod
    def from_pretrained(cls, config: Any, *, fine_tuned: bool = False) -> FakeBaseline:
        cls.built.append((config.hf_id, fine_tuned))
        return cls(fine_tuned=fine_tuned)

    @classmethod
    def from_checkpoint(cls, config: Any, state: dict[str, Any]) -> FakeBaseline:
        cls.built.append((config.hf_id, True))
        return cls(fine_tuned=True)

    def to(self, device: torch.device) -> FakeBaseline:
        self.moved_to = device
        return self

    def summarize(
        self, documents: Sequence[str], config: Any = None, *, batch_size: int = 8
    ) -> list[str]:
        return ["resume" for _ in documents]

    def describe(self) -> dict[str, str]:
        return {"baseline": "fake"}


@pytest.fixture
def fake_baseline(monkeypatch: pytest.MonkeyPatch) -> Callable[[], list[tuple[str, bool]]]:
    """Replace the baseline registry with a class that needs no network."""
    FakeBaseline.built = []
    monkeypatch.setattr(loader_module, "get_baseline_class", lambda name: FakeBaseline)
    return lambda: FakeBaseline.built


def pretrained_version(**overrides: Any) -> ModelVersion:
    """Build a published version of the pretrained side."""
    fields: dict[str, Any] = {
        "name": "pretrained",
        "source": ModelSource.HUGGING_FACE,
        "artifact": None,
        "checksum": None,
        "hf_id": "t5-small",
        "revision": "d769bba",
        "architecture": {"type": "pretrained", "baseline": "t5", "mode": "zero_shot"},
    }
    fields.update(overrides)
    return build_version(**fields)


@pytest.mark.unit
def test_a_hub_version_is_loaded_zero_shot(
    fake_baseline: Callable[[], list[tuple[str, bool]]],
) -> None:
    loaded = load_version(pretrained_version(), None, device=CPU)

    assert fake_baseline() == [("t5-small", False)]
    assert isinstance(loaded.summarizer, FakeBaseline)
    assert loaded.summarizer.moved_to == CPU


@pytest.mark.unit
def test_a_fine_tuned_version_is_loaded_from_its_weights(
    tmp_path: Path, fake_baseline: Callable[[], list[tuple[str, bool]]]
) -> None:
    weights = write_weights(tmp_path / "weights.pt", {"a": torch.zeros(1)})
    version = pretrained_version(
        source=ModelSource.CHECKPOINT,
        artifact="pretrained/v1/weights.pt",
        checksum="0" * 64,
        architecture={"type": "pretrained", "baseline": "t5", "mode": "fine_tuned"},
    )

    load_version(version, weights, device=CPU)

    assert fake_baseline() == [("t5-small", True)]


@pytest.mark.unit
def test_the_pinned_revision_travels_to_the_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class Recording(FakeBaseline):
        @classmethod
        def from_pretrained(cls, config: Any, *, fine_tuned: bool = False) -> FakeBaseline:
            seen.update({"hf_id": config.hf_id, "revision": config.revision})
            return cls(fine_tuned=fine_tuned)

    monkeypatch.setattr(loader_module, "get_baseline_class", lambda name: Recording)
    load_version(pretrained_version(), None, device=CPU)

    assert seen == {"hf_id": "t5-small", "revision": "d769bba"}


@pytest.mark.unit
def test_an_unknown_baseline_is_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> type[FakeBaseline]:
        raise KeyError(f"Unknown baseline {name!r}.")

    monkeypatch.setattr(loader_module, "get_baseline_class", missing)

    with pytest.raises(ModelNotReadyError, match="baseline n'est plus enregistree"):
        load_version(pretrained_version(), None, device=CPU)


@pytest.mark.unit
def test_fine_tuned_weights_that_do_not_fit_the_baseline_are_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Mismatched(FakeBaseline):
        @classmethod
        def from_checkpoint(cls, config: Any, state: dict[str, Any]) -> FakeBaseline:
            raise RuntimeError("size mismatch for shared.weight")

    monkeypatch.setattr(loader_module, "get_baseline_class", lambda name: Mismatched)
    weights = write_weights(tmp_path / "weights.pt", {"a": torch.zeros(1)})

    fine_tuned = pretrained_version(
        source=ModelSource.CHECKPOINT,
        artifact="pretrained/v1/weights.pt",
        checksum="0" * 64,
    )

    with pytest.raises(ModelNotReadyError, match="ne correspondent pas"):
        load_version(fine_tuned, weights, device=CPU)


@pytest.mark.unit
def test_a_hub_that_cannot_be_reached_is_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    class Offline(FakeBaseline):
        @classmethod
        def from_pretrained(cls, config: Any, *, fine_tuned: bool = False) -> FakeBaseline:
            raise OSError("connection refused")

    monkeypatch.setattr(loader_module, "get_baseline_class", lambda name: Offline)

    with pytest.raises(ModelNotReadyError, match="n'a pas pu etre recuperee"):
        load_version(pretrained_version(), None, device=CPU)


@pytest.mark.unit
def test_the_error_names_the_version_without_leaking_a_path() -> None:
    with pytest.raises(ModelNotReadyError) as raised:
        load_scratch(
            build_version(),
            ScratchModelConfig.model_validate(build_version().architecture),
            TokenizerConfig.model_validate(build_version().tokenizer),
            None,
        )

    assert raised.value.details == {
        "model": "scratch",
        "version": "v1",
        "reason": "un modele from scratch n'existe pas hors de ses poids publies",
    }


@pytest.mark.unit
def test_a_hub_version_carrying_weights_is_fine_tuned(
    tmp_path: Path, fake_baseline: Callable[[], list[tuple[str, bool]]]
) -> None:
    weights = write_weights(tmp_path / "weights.pt", {"a": torch.zeros(1)})

    summarizer = load_pretrained(
        pretrained_version(),
        PretrainedModelConfig.model_validate(pretrained_version().architecture),
        TokenizerConfig.model_validate(build_version().tokenizer),
        weights,
    )

    assert isinstance(summarizer, FakeBaseline)
    assert summarizer.fine_tuned is True
