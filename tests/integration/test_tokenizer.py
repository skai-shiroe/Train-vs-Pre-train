"""Integration tests for the shared T5 tokeniser.

These tests load the real ``t5-small`` tokeniser. The first run downloads it,
later runs read the Hugging Face cache. They cover the chain the specification
asks for: dataset to tokeniser, tokeniser to model input.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from src.data.config import TokenizerConfig
from src.data.dataset import Collator, build_dataloader
from src.data.example import Example
from src.data.tokenize import IGNORE_INDEX, build_tokenizer, encode_batch, token_lengths

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def config() -> TokenizerConfig:
    return TokenizerConfig(
        hf_id="t5-small",
        source_prefix="summarize: ",
        max_source_tokens=64,
        max_target_tokens=16,
    )


@pytest.fixture(scope="module")
def tokenizer(config: TokenizerConfig) -> object:
    return build_tokenizer(config.hf_id)


@pytest.fixture
def examples() -> list[Example]:
    return [
        Example("a", "The council announced a new plan for the city centre today.", "New plan."),
        Example("b", "A much longer document " * 40, "A long summary of the document."),
    ]


def test_the_tokeniser_is_cached_per_identifier(config: TokenizerConfig) -> None:
    assert build_tokenizer(config.hf_id) is build_tokenizer(config.hf_id)


def test_encoding_produces_the_four_expected_tensors(
    examples: list[Example], tokenizer: object, config: TokenizerConfig
) -> None:
    batch = encode_batch(examples, tokenizer, config)  # type: ignore[arg-type]

    assert batch.input_ids.shape[0] == 2
    assert batch.attention_mask.shape == batch.input_ids.shape
    assert batch.labels.shape == batch.target_ids.shape
    assert len(batch) == 2


def test_truncation_respects_the_configured_lengths(
    examples: list[Example], tokenizer: object, config: TokenizerConfig
) -> None:
    batch = encode_batch(examples, tokenizer, config)  # type: ignore[arg-type]

    assert batch.input_ids.shape[1] <= config.max_source_tokens
    assert batch.target_ids.shape[1] <= config.max_target_tokens


def test_padding_positions_are_ignored_by_the_loss(
    examples: list[Example], tokenizer: object, config: TokenizerConfig
) -> None:
    batch = encode_batch(examples, tokenizer, config)  # type: ignore[arg-type]

    padded = batch.target_ids == 0
    assert bool((batch.labels[padded] == IGNORE_INDEX).all())
    assert bool((batch.labels[~padded] == batch.target_ids[~padded]).all())


def test_the_attention_mask_marks_the_real_tokens(
    examples: list[Example], tokenizer: object, config: TokenizerConfig
) -> None:
    batch = encode_batch(examples, tokenizer, config)  # type: ignore[arg-type]

    assert bool(((batch.attention_mask == 0) == (batch.input_ids == 0)).all())


def test_the_source_prefix_is_prepended(tokenizer: object, config: TokenizerConfig) -> None:
    without = encode_batch(
        [Example("a", "hello world", "hi")],
        tokenizer,  # type: ignore[arg-type]
        config.model_copy(update={"source_prefix": ""}),
    )
    with_prefix = encode_batch(
        [Example("a", "hello world", "hi")],
        tokenizer,  # type: ignore[arg-type]
        config,
    )

    assert with_prefix.input_ids.shape[1] > without.input_ids.shape[1]


def test_an_empty_batch_is_refused(tokenizer: object, config: TokenizerConfig) -> None:
    with pytest.raises(ValueError, match="Cannot encode an empty batch"):
        encode_batch([], tokenizer, config)  # type: ignore[arg-type]


def test_the_batch_moves_to_a_device(
    examples: list[Example], tokenizer: object, config: TokenizerConfig
) -> None:
    batch = encode_batch(examples, tokenizer, config)  # type: ignore[arg-type]

    moved = batch.to(torch.device("cpu"))

    assert moved.input_ids.device.type == "cpu"
    assert torch.equal(moved.labels, batch.labels)


def test_token_lengths_are_measured_without_truncation(
    examples: list[Example], tokenizer: object, config: TokenizerConfig
) -> None:
    source_lengths, target_lengths = token_lengths(examples, tokenizer, config)  # type: ignore[arg-type]

    assert len(source_lengths) == len(target_lengths) == 2
    # The second document is far longer than the truncation budget, which is
    # exactly what justifies measuring the untruncated distribution.
    assert source_lengths[1] > config.max_source_tokens


def test_padding_adapts_to_each_batch(tokenizer: object, config: TokenizerConfig) -> None:
    # Dynamic padding: the tokeniser pads to the longest sequence of the batch,
    # not to a fixed max_length. Two batches of different content must
    # therefore produce tensors of different widths.
    short = [Example("s1", "A short one.", "Short."), Example("s2", "Another short.", "Also.")]
    long = [Example("l1", "A far longer document " * 6, "Longer summary here.")]

    narrow = encode_batch(short, tokenizer, config)  # type: ignore[arg-type]
    wide = encode_batch(long, tokenizer, config)  # type: ignore[arg-type]

    assert wide.input_ids.shape[1] > narrow.input_ids.shape[1]


def test_a_batch_is_never_wider_than_its_longest_sequence(
    tokenizer: object, config: TokenizerConfig
) -> None:
    batch = encode_batch(
        [Example("a", "one two three", "x"), Example("b", "a much longer document here", "y")],
        tokenizer,  # type: ignore[arg-type]
        config,
    )

    longest_real = int(batch.attention_mask.sum(dim=1).max())

    assert batch.input_ids.shape[1] == longest_real


def test_a_uniform_batch_carries_no_padding_at_all(
    tokenizer: object, config: TokenizerConfig
) -> None:
    identical = [Example(str(index), "exactly the same text", "same") for index in range(4)]

    batch = encode_batch(identical, tokenizer, config)  # type: ignore[arg-type]

    assert bool((batch.attention_mask == 1).all())


def test_the_collator_encodes_a_list_of_examples(
    examples: list[Example], tokenizer: object, config: TokenizerConfig
) -> None:
    collator = Collator(tokenizer, config)  # type: ignore[arg-type]

    batch = collator(examples)

    assert batch.input_ids.shape[0] == 2


def test_the_dataloader_yields_encoded_batches(
    make_examples: Callable[[int], list[Example]],
    tokenizer: object,
    config: TokenizerConfig,
) -> None:
    loader = build_dataloader(
        make_examples(7),
        tokenizer,  # type: ignore[arg-type]
        config,
        batch_size=3,
        shuffle=False,
    )

    batches = list(loader)

    assert len(batches) == 3
    assert [len(batch) for batch in batches] == [3, 3, 1]


def test_the_evaluation_dataloader_keeps_the_corpus_order(
    make_examples: Callable[[int], list[Example]],
    tokenizer: object,
    config: TokenizerConfig,
) -> None:
    examples = make_examples(6)
    loader = build_dataloader(
        examples,
        tokenizer,  # type: ignore[arg-type]
        config,
        batch_size=2,
        shuffle=False,
    )

    first = next(iter(loader))
    expected = encode_batch(examples[:2], tokenizer, config)  # type: ignore[arg-type]

    assert torch.equal(first.input_ids, expected.input_ids)
