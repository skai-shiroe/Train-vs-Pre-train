"""Integration tests of the training chain.

These tests load the real ``t5-small`` tokeniser and run the real Transformer,
so they cover the links the specification asks for: dataset to tokeniser,
tokeniser to model, model to loss, loss to checkpoint.

The model is deliberately tiny and the corpus is synthetic. What is checked is
that the chain holds together and that the length grouping actually pays off on
documents with a realistic length spread, not that the architecture learns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.config import TokenizerConfig
from src.data.dataset import build_dataloader
from src.data.example import Example
from src.data.tokenize import build_tokenizer, source_token_lengths
from src.models.scratch.config import ScratchTransformerConfig
from src.models.scratch.transformer import ScratchTransformer
from src.training.config import TrainingConfig
from src.training.sampler import LengthGroupedSampler, build_training_dataloader, padding_waste
from src.training.trainer import Trainer
from src.utils.seed import set_seed

pytestmark = pytest.mark.integration

BATCH_SIZE = 8
SEED = 42


@pytest.fixture(scope="module")
def tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig(
        hf_id="t5-small",
        source_prefix="summarize: ",
        max_source_tokens=128,
        max_target_tokens=16,
    )


@pytest.fixture(scope="module")
def tokenizer(tokenizer_config: TokenizerConfig) -> object:
    return build_tokenizer(tokenizer_config.hf_id)


@pytest.fixture(scope="module")
def spread_examples() -> list[Example]:
    """Return documents whose lengths span the truncation budget.

    A third of them reach the truncation length, which is the shape that makes
    dynamic padding degenerate on XSum.
    """
    examples = []
    for index in range(160):
        words = 4 + (index * 7) % 90
        if index % 3 == 0:
            words = 200
        examples.append(
            Example(
                example_id=f"id-{index}",
                source=" ".join(f"word{index}{position}" for position in range(words)),
                target=f"Summary of document number {index}.",
            )
        )
    return examples


# ---------------------------------------------------------------------------
# Lengths
# ---------------------------------------------------------------------------


def test_the_measured_lengths_stop_at_the_truncation_budget(
    spread_examples: list[Example], tokenizer: object, tokenizer_config: TokenizerConfig
) -> None:
    lengths = source_token_lengths(spread_examples, tokenizer, tokenizer_config)  # type: ignore[arg-type]

    assert len(lengths) == len(spread_examples)
    assert max(lengths) == tokenizer_config.max_source_tokens


def test_measuring_nothing_needs_no_tokeniser_call(
    tokenizer: object, tokenizer_config: TokenizerConfig
) -> None:
    assert source_token_lengths([], tokenizer, tokenizer_config) == []  # type: ignore[arg-type]


def test_the_measured_length_is_the_width_the_batch_gets(
    spread_examples: list[Example], tokenizer: object, tokenizer_config: TokenizerConfig
) -> None:
    # The sorting key must be the quantity that actually drives the batch
    # width, otherwise the grouping optimises the wrong thing.
    batch = spread_examples[:BATCH_SIZE]
    lengths = source_token_lengths(batch, tokenizer, tokenizer_config)  # type: ignore[arg-type]

    loader = build_dataloader(
        batch,
        tokenizer,  # type: ignore[arg-type]
        tokenizer_config,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    encoded = next(iter(loader))

    assert encoded.input_ids.shape[1] == max(lengths)


# ---------------------------------------------------------------------------
# Length grouping
# ---------------------------------------------------------------------------


def test_grouping_narrows_the_batches_the_model_receives(
    spread_examples: list[Example], tokenizer: object, tokenizer_config: TokenizerConfig
) -> None:
    grouped, sampler = build_training_dataloader(
        spread_examples,
        tokenizer,  # type: ignore[arg-type]
        tokenizer_config,
        batch_size=BATCH_SIZE,
        seed=SEED,
        mega_batch_factor=10,
    )
    assert sampler is not None

    shuffled, no_sampler = build_training_dataloader(
        spread_examples,
        tokenizer,  # type: ignore[arg-type]
        tokenizer_config,
        batch_size=BATCH_SIZE,
        seed=SEED,
        group_by_length=False,
    )
    assert no_sampler is None

    def wasted(loader: object) -> float:
        real = 0
        allocated = 0
        for batch in loader:  # type: ignore[attr-defined]
            real += int(batch.attention_mask.sum())
            allocated += int(batch.attention_mask.numel())
        return 1.0 - real / allocated

    assert wasted(grouped) < wasted(shuffled) / 2


def test_the_loader_covers_the_corpus_exactly_once(
    spread_examples: list[Example], tokenizer: object, tokenizer_config: TokenizerConfig
) -> None:
    loader, _ = build_training_dataloader(
        spread_examples,
        tokenizer,  # type: ignore[arg-type]
        tokenizer_config,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    assert sum(len(batch) for batch in loader) == len(spread_examples)
    assert len(loader) == len(spread_examples) // BATCH_SIZE


def test_the_measured_waste_matches_what_the_batches_carry(
    spread_examples: list[Example], tokenizer: object, tokenizer_config: TokenizerConfig
) -> None:
    # The helper used by the documentation and the trainer must report the same
    # number as the tensors actually produced.
    lengths = source_token_lengths(spread_examples, tokenizer, tokenizer_config)  # type: ignore[arg-type]
    sampler = LengthGroupedSampler(lengths, BATCH_SIZE, seed=SEED, mega_batch_factor=10)
    loader = build_dataloader(
        spread_examples,
        tokenizer,  # type: ignore[arg-type]
        tokenizer_config,
        batch_sampler=sampler,
    )

    real = sum(int(batch.attention_mask.sum()) for batch in loader)
    allocated = sum(int(batch.attention_mask.numel()) for batch in loader)

    assert padding_waste(lengths, sampler) == pytest.approx(1.0 - real / allocated, abs=1e-9)


# ---------------------------------------------------------------------------
# The whole chain
# ---------------------------------------------------------------------------


def test_a_run_goes_from_the_corpus_to_a_checkpoint(
    spread_examples: list[Example],
    tokenizer: object,
    tokenizer_config: TokenizerConfig,
    tmp_path: Path,
) -> None:
    set_seed(SEED)
    vocab_size = int(tokenizer.vocab_size)  # type: ignore[attr-defined]
    model = ScratchTransformer(
        ScratchTransformerConfig(
            vocab_size=vocab_size,
            d_model=32,
            num_heads=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            d_ff=64,
            dropout=0.0,
            max_position=tokenizer_config.max_source_tokens,
        )
    )

    train_loader, _ = build_training_dataloader(
        spread_examples[:64],
        tokenizer,  # type: ignore[arg-type]
        tokenizer_config,
        batch_size=BATCH_SIZE,
        seed=SEED,
        mega_batch_factor=4,
    )
    validation_loader = build_dataloader(
        spread_examples[64:96],
        tokenizer,  # type: ignore[arg-type]
        tokenizer_config,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    trainer = Trainer(
        model,
        TrainingConfig(
            epochs=2,
            batch_size=BATCH_SIZE,
            learning_rate=1e-3,
            warmup_ratio=0.1,
            device="cpu",
            mixed_precision=False,
            early_stopping_patience=0,
            log_every_steps=1000,
        ),
        train_loader=train_loader,
        validation_loader=validation_loader,
        output_dir=tmp_path / "runs",
        callbacks=[],
        metadata={"dataset_version": "integration"},
    )

    result = trainer.train()

    assert len(result.epochs) == 2
    assert result.epochs[-1].train_loss < result.epochs[0].train_loss
    assert result.best_checkpoint is not None
    assert result.best_checkpoint.is_file()
