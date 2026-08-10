"""Unit tests for the T5 baseline.

A T5 of a few thousand parameters and a tokeniser that needs no vocabulary keep
these tests offline and fast. What they check is the wiring: the prefix, the
truncation, the shape of what comes back from generation, the persistence, and
the fact that the loss adapter really drives the shared training loop. Whether
``t5-small`` actually summarises is the integration suite's business.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration

from src.data.example import Example
from src.models.generation import GenerationConfig
from src.models.pretrained.base import BaselineConfig
from src.models.pretrained.t5 import T5Summarizer
from src.training.config import TrainingConfig
from src.training.trainer import Trainer
from src.utils.seed import set_seed
from tests.conftest import FakeTokenizer

T5Factory = Callable[..., T5ForConditionalGeneration]
LoaderFactory = Callable[..., DataLoader[Example]]


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    set_seed(1234)


@pytest.fixture
def config() -> BaselineConfig:
    return BaselineConfig(
        hf_id="t5-small",
        source_prefix="summarize: ",
        max_source_tokens=16,
        max_target_tokens=6,
    )


@pytest.fixture
def summarizer(
    config: BaselineConfig, tiny_t5: T5Factory, fake_tokenizer: FakeTokenizer
) -> T5Summarizer:
    return T5Summarizer(config, tiny_t5(), fake_tokenizer)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_a_blank_prefix_is_refused_before_anything_is_downloaded() -> None:
    # T5 selects its task from the prefix. Loading the weights first and only
    # then discovering the mistake would cost a download for nothing, and the
    # run would report a score that measures the wrong task.
    with pytest.raises(ValueError, match="selects its task from a textual prefix"):
        T5Summarizer.from_pretrained(BaselineConfig(hf_id="t5-small", source_prefix="   "))


def test_the_prefix_requirement_is_stated_on_the_class() -> None:
    with pytest.raises(ValueError, match="summarize: "):
        T5Summarizer.check_config(BaselineConfig(hf_id="t5-small"))


def test_a_prefixed_configuration_passes_the_check() -> None:
    T5Summarizer.check_config(BaselineConfig(hf_id="t5-small", source_prefix="summarize: "))


def test_the_baseline_is_registered_under_a_name() -> None:
    assert T5Summarizer.name == "t5"
    assert T5Summarizer.default_hf_id == "t5-small"


# ---------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------


def test_a_baseline_starts_zero_shot(summarizer: T5Summarizer) -> None:
    assert summarizer.fine_tuned is False
    assert summarizer.describe()["mode"] == "zero_shot"


def test_the_description_carries_what_produced_a_score(summarizer: T5Summarizer) -> None:
    described = summarizer.describe()

    assert described["baseline"] == "t5"
    assert described["hf_id"] == "t5-small"
    assert described["revision"] == "default"
    assert int(described["parameters"]) == summarizer.num_parameters


def test_a_pinned_revision_is_reported(tiny_t5: T5Factory, fake_tokenizer: FakeTokenizer) -> None:
    config = BaselineConfig(hf_id="t5-small", source_prefix="summarize: ", revision="abc123")

    summarizer = T5Summarizer(config, tiny_t5(), fake_tokenizer)  # type: ignore[arg-type]

    assert summarizer.describe()["revision"] == "abc123"


def test_the_parameters_are_counted(summarizer: T5Summarizer) -> None:
    assert summarizer.num_parameters > 0


def test_the_model_and_the_tokeniser_are_reachable(
    summarizer: T5Summarizer, fake_tokenizer: FakeTokenizer
) -> None:
    # The trainer needs the module itself, and the evaluation needs the
    # tokeniser the corpus was encoded with.
    assert isinstance(summarizer.model, T5ForConditionalGeneration)
    assert summarizer.tokenizer is fake_tokenizer


def test_moving_the_baseline_returns_it(summarizer: T5Summarizer) -> None:
    moved = summarizer.to(torch.device("cpu"))

    assert moved is summarizer
    assert summarizer.device.type == "cpu"


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def test_the_prefix_is_prepended_to_every_document(
    summarizer: T5Summarizer, fake_tokenizer: FakeTokenizer
) -> None:
    summarizer.encode(["a first document", "a second one"])

    assert fake_tokenizer.calls[-1] == [
        "summarize: a first document",
        "summarize: a second one",
    ]


def test_the_encoded_batch_respects_the_truncation_budget(summarizer: T5Summarizer) -> None:
    input_ids, attention_mask = summarizer.encode(["word " * 200])

    assert input_ids.shape[1] <= summarizer.config.max_source_tokens
    assert attention_mask.shape == input_ids.shape


def test_the_encoded_batch_lands_on_the_model_device(summarizer: T5Summarizer) -> None:
    input_ids, _ = summarizer.encode(["a document"])

    assert input_ids.device == summarizer.device


def test_encoding_nothing_is_refused(summarizer: T5Summarizer) -> None:
    with pytest.raises(ValueError, match="Cannot encode an empty batch"):
        summarizer.encode([])


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_generation_drops_the_decoder_start_token(summarizer: T5Summarizer) -> None:
    # The from scratch generator returns the generated tokens only. Keeping the
    # start token here would make the two outputs incomparable.
    input_ids, attention_mask = summarizer.encode(["a document"])
    settings = GenerationConfig(max_new_tokens=4)

    generated = summarizer.generate_ids(input_ids, attention_mask, settings)
    raw = summarizer.model.generate(
        input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=4, do_sample=False
    )

    assert generated.shape[1] == raw.shape[1] - 1
    assert torch.equal(generated, raw[:, 1:])


def test_generation_stays_within_the_budget(summarizer: T5Summarizer) -> None:
    input_ids, attention_mask = summarizer.encode(["a document"])

    generated = summarizer.generate_ids(
        input_ids, attention_mask, GenerationConfig(max_new_tokens=3)
    )

    assert generated.shape[1] <= 3


def test_the_default_budget_is_the_target_truncation(summarizer: T5Summarizer) -> None:
    input_ids, attention_mask = summarizer.encode(["a document"])

    generated = summarizer.generate_ids(input_ids, attention_mask)

    assert generated.shape[1] <= summarizer.config.max_target_tokens


def test_beam_search_still_returns_one_sequence_per_input(summarizer: T5Summarizer) -> None:
    input_ids, attention_mask = summarizer.encode(["first", "second"])

    generated = summarizer.generate_ids(
        input_ids, attention_mask, GenerationConfig(max_new_tokens=4, num_beams=3)
    )

    assert generated.shape[0] == 2


def test_generation_leaves_the_model_in_evaluation_mode(summarizer: T5Summarizer) -> None:
    summarizer.model.train()
    input_ids, attention_mask = summarizer.encode(["a document"])

    summarizer.generate_ids(input_ids, attention_mask, GenerationConfig(max_new_tokens=2))

    assert summarizer.model.training is False


def test_generation_builds_no_gradient(summarizer: T5Summarizer) -> None:
    input_ids, attention_mask = summarizer.encode(["a document"])

    generated = summarizer.generate_ids(
        input_ids, attention_mask, GenerationConfig(max_new_tokens=2)
    )

    assert generated.requires_grad is False


# ---------------------------------------------------------------------------
# Summarising
# ---------------------------------------------------------------------------


def test_one_summary_comes_back_per_document(summarizer: T5Summarizer) -> None:
    summaries = summarizer.summarize(["first", "second", "third"])

    assert len(summaries) == 3
    assert all(isinstance(summary, str) for summary in summaries)


def test_summarising_nothing_returns_nothing(summarizer: T5Summarizer) -> None:
    assert summarizer.summarize([]) == []


def test_the_documents_are_summarised_in_batches(
    summarizer: T5Summarizer, fake_tokenizer: FakeTokenizer
) -> None:
    # A long test split must not be encoded in one tensor, or the memory cost
    # would grow with the corpus rather than with the batch.
    fake_tokenizer.calls.clear()

    summarizer.summarize(["first", "second", "third"], batch_size=2)

    assert [len(call) for call in fake_tokenizer.calls] == [2, 1]


def test_an_impossible_batch_size_is_refused(summarizer: T5Summarizer) -> None:
    with pytest.raises(ValueError, match="batch_size must be strictly positive"):
        summarizer.summarize(["first"], batch_size=0)


def test_the_special_tokens_are_stripped_from_the_summaries(summarizer: T5Summarizer) -> None:
    summaries = summarizer.summarize(["a document"], GenerationConfig(max_new_tokens=4))

    assert "t0" not in summaries[0]
    assert "t1" not in summaries[0]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_saving_writes_the_weights_and_the_tokeniser(
    summarizer: T5Summarizer, fake_tokenizer: FakeTokenizer, tmp_path: Path
) -> None:
    # The directory must reload on a machine that never saw the hub, which it
    # cannot do without the tokeniser next to the weights.
    directory = summarizer.save_pretrained(tmp_path / "baseline")

    assert (directory / "config.json").is_file()
    assert any(directory.glob("*.safetensors")) or (directory / "pytorch_model.bin").is_file()
    assert fake_tokenizer.saved_to == [str(directory)]


def test_a_checkpoint_rebuilds_a_fine_tuned_baseline(
    config: BaselineConfig,
    summarizer: T5Summarizer,
    tiny_t5: T5Factory,
    fake_tokenizer: FakeTokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(T5Summarizer, "load_model", classmethod(lambda cls, settings: tiny_t5()))
    monkeypatch.setattr("src.models.pretrained.base.build_tokenizer", lambda hf_id: fake_tokenizer)
    trained = summarizer.model.state_dict()

    reloaded = T5Summarizer.from_checkpoint(config, trained)

    assert reloaded.fine_tuned is True
    assert reloaded.describe()["mode"] == "fine_tuned"
    for name, parameter in reloaded.model.state_dict().items():
        assert torch.equal(parameter, trained[name])


# ---------------------------------------------------------------------------
# Fine tuning through the shared loop
# ---------------------------------------------------------------------------


def test_the_baseline_fine_tunes_through_the_shared_trainer(
    summarizer: T5Summarizer, make_loader: LoaderFactory, tmp_path: Path
) -> None:
    # The whole point of the loss adapter: no second training loop. The same
    # Trainer, the same token weighted averaging and the same checkpoints as
    # the from scratch model.
    trainer = Trainer(
        summarizer.model,
        TrainingConfig(
            epochs=2,
            batch_size=4,
            learning_rate=1e-2,
            warmup_ratio=0.0,
            device="cpu",
            mixed_precision=False,
            early_stopping_patience=0,
            log_every_steps=1000,
        ),
        train_loader=make_loader(16, batch_size=4),
        validation_loader=make_loader(8, batch_size=4),
        output_dir=tmp_path / "runs",
        batch_loss=summarizer.batch_loss(),
        callbacks=[],
        metadata=summarizer.describe(),
    )

    result = trainer.train()

    assert len(result.epochs) == 2
    assert result.epochs[-1].train_loss < result.epochs[0].train_loss
    assert result.best_checkpoint is not None
    assert result.best_checkpoint.is_file()


def test_the_smoothed_loss_is_also_wired_to_the_trainer(
    summarizer: T5Summarizer, make_loader: LoaderFactory
) -> None:
    batch = next(iter(make_loader(4, batch_size=4)))

    plain = summarizer.batch_loss()(summarizer.model, batch).detach()
    smoothed = summarizer.batch_loss(0.1)(summarizer.model, batch).detach()

    assert torch.isfinite(plain)
    assert torch.isfinite(smoothed)
    assert float(smoothed) != float(plain)
