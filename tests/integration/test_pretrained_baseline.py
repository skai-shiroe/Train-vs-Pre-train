"""Integration tests of the pretrained baseline.

These tests load the real ``t5-small`` weights, so they are marked ``slow`` and
stay out of the pre-push hook. They cover what a stub cannot: that the
downloaded weights carry real knowledge, that the shared tokeniser is literally
shared, and that fine tuning the baseline runs through the same training loop as
the from scratch model.

Nothing here reports a score. The corpus is a handful of hand written
paragraphs, and any number measured on it would say nothing about XSum.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from src.data.config import TokenizerConfig
from src.data.dataset import build_dataloader
from src.data.example import Example
from src.data.tokenize import build_tokenizer, encode_batch
from src.models.generation import GenerationConfig
from src.models.pretrained.base import BaselineConfig
from src.models.pretrained.factory import build_summarizer
from src.models.pretrained.t5 import T5Summarizer
from src.training.config import TrainingConfig
from src.training.trainer import Trainer
from src.utils.seed import set_seed

pytestmark = [pytest.mark.integration, pytest.mark.slow]

DOCUMENT = (
    "The city council has approved a plan to rebuild the old railway bridge "
    "over the river. Work is expected to start in the spring and to last for "
    "about eighteen months. Local traders have warned that the closure of the "
    "main road will cut them off from their customers during the works."
)


@pytest.fixture(scope="module")
def tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig(
        hf_id="t5-small",
        source_prefix="summarize: ",
        max_source_tokens=128,
        max_target_tokens=24,
    )


@pytest.fixture(scope="module")
def config(tokenizer_config: TokenizerConfig) -> BaselineConfig:
    return BaselineConfig.from_tokenizer_config(tokenizer_config)


@pytest.fixture(scope="module")
def summarizer(config: BaselineConfig) -> T5Summarizer:
    baseline = build_summarizer("t5", config)
    assert isinstance(baseline, T5Summarizer)
    return baseline


@pytest.fixture
def examples() -> list[Example]:
    return [
        Example(
            example_id=f"id-{index}",
            source=f"{DOCUMENT} The report number is {index}.",
            target=f"The council approved the bridge plan, report {index}.",
        )
        for index in range(8)
    ]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_the_baseline_and_the_corpus_share_one_tokeniser(summarizer: T5Summarizer) -> None:
    # Section 2.1 requires the two models to share the vocabulary. Sharing the
    # instance is stronger than repeating an identifier in two files.
    assert summarizer.tokenizer is build_tokenizer("t5-small")


def test_the_real_checkpoint_is_the_size_it_should_be(summarizer: T5Summarizer) -> None:
    # t5-small is a 60 million parameter model. A wrong identifier silently
    # loading something else would show up here.
    assert 50_000_000 < summarizer.num_parameters < 80_000_000


def test_a_freshly_loaded_baseline_is_zero_shot(summarizer: T5Summarizer) -> None:
    assert summarizer.fine_tuned is False
    assert summarizer.describe()["mode"] == "zero_shot"


# ---------------------------------------------------------------------------
# Zero shot
# ---------------------------------------------------------------------------


def test_the_zero_shot_baseline_produces_a_summary(summarizer: T5Summarizer) -> None:
    summaries = summarizer.summarize([DOCUMENT], GenerationConfig(max_new_tokens=24))

    assert len(summaries) == 1
    assert summaries[0].strip()
    assert len(summaries[0]) < len(DOCUMENT)


def test_the_prefix_steers_the_model(summarizer: T5Summarizer, config: BaselineConfig) -> None:
    # This is why a blank prefix is refused: T5 does a different job depending
    # on what it is asked, and the prefix is the whole of the question.
    translation = T5Summarizer(
        BaselineConfig(
            hf_id=config.hf_id,
            source_prefix="translate English to German: ",
            max_source_tokens=config.max_source_tokens,
            max_target_tokens=config.max_target_tokens,
        ),
        summarizer.model,
        summarizer.tokenizer,
    )
    settings = GenerationConfig(max_new_tokens=24)

    assert summarizer.summarize([DOCUMENT], settings) != translation.summarize([DOCUMENT], settings)


def test_beam_search_and_greedy_search_both_run(summarizer: T5Summarizer) -> None:
    greedy = summarizer.summarize([DOCUMENT], GenerationConfig(max_new_tokens=24))
    beams = summarizer.summarize([DOCUMENT], GenerationConfig(max_new_tokens=24, num_beams=4))

    assert greedy[0].strip()
    assert beams[0].strip()


def test_decoding_is_reproducible(summarizer: T5Summarizer) -> None:
    settings = GenerationConfig(max_new_tokens=24, num_beams=2)

    assert summarizer.summarize([DOCUMENT], settings) == summarizer.summarize([DOCUMENT], settings)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def test_the_pretrained_weights_beat_an_untrained_model(
    summarizer: T5Summarizer, examples: list[Example], tokenizer_config: TokenizerConfig
) -> None:
    # A model that has learned nothing spreads its probability mass evenly over
    # the vocabulary, which costs log(vocab_size) per token. Anything above that
    # would mean the weights never arrived.
    batch = encode_batch(examples[:2], summarizer.tokenizer, tokenizer_config)

    loss = summarizer.batch_loss()(summarizer.model, batch).detach()

    assert torch.isfinite(loss)
    assert float(loss) < math.log(int(summarizer.tokenizer.vocab_size))


# ---------------------------------------------------------------------------
# Fine tuning
# ---------------------------------------------------------------------------


def test_fine_tuning_runs_through_the_shared_training_loop(
    summarizer: T5Summarizer,
    examples: list[Example],
    tokenizer_config: TokenizerConfig,
    tmp_path: Path,
) -> None:
    set_seed(42)
    train_loader = build_dataloader(
        examples, summarizer.tokenizer, tokenizer_config, batch_size=2, shuffle=False
    )
    validation_loader = build_dataloader(
        examples[:4], summarizer.tokenizer, tokenizer_config, batch_size=2, shuffle=False
    )

    trainer = Trainer(
        summarizer.model,
        TrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            warmup_ratio=0.0,
            device="cpu",
            mixed_precision=False,
            early_stopping_patience=0,
            log_every_steps=1000,
        ),
        train_loader=train_loader,
        validation_loader=validation_loader,
        output_dir=tmp_path / "runs",
        batch_loss=summarizer.batch_loss(),
        callbacks=[],
        metadata=summarizer.describe(),
    )

    before = float(trainer.evaluate(validation_loader))
    result = trainer.train()

    assert result.epochs[0].validation_loss < before
    assert result.best_checkpoint is not None
    assert result.best_checkpoint.is_file()


def test_a_saved_baseline_reloads_without_the_hub(
    summarizer: T5Summarizer, config: BaselineConfig, tmp_path: Path
) -> None:
    # This is the path the model registry of section 20 and the API of section
    # 27 take: a directory, no network, and the tokeniser that goes with it.
    directory = summarizer.save_pretrained(tmp_path / "baseline")

    reloaded = T5Summarizer.from_pretrained(
        BaselineConfig(
            hf_id=str(directory),
            source_prefix=config.source_prefix,
            max_source_tokens=config.max_source_tokens,
            max_target_tokens=config.max_target_tokens,
        ),
        fine_tuned=True,
    )

    settings = GenerationConfig(max_new_tokens=16)
    assert reloaded.describe()["mode"] == "fine_tuned"
    assert reloaded.summarize([DOCUMENT], settings) == summarizer.summarize([DOCUMENT], settings)
