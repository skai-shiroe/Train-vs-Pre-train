"""Unit tests of the from scratch summariser adapter.

Nothing here downloads anything: the tokeniser is the offline fake of
``conftest.py`` and the model is a two layer Transformer with a tiny
vocabulary. What is checked is the wiring, never the quality of what a randomly
initialised model produces.
"""

from __future__ import annotations

import pytest
import torch

from src.data.config import TokenizerConfig
from src.models.generation import GenerationConfig
from src.models.scratch.config import ScratchTransformerConfig
from src.models.scratch.summarizer import ScratchSummarizer
from src.models.scratch.transformer import ScratchTransformer
from tests.conftest import FAKE_VOCAB_SIZE, FakeTokenizer

DOCUMENTS = ["First document to summarise.", "Second document, a little longer than the first."]


def make_model(**overrides: int) -> ScratchTransformer:
    settings = {
        "vocab_size": FAKE_VOCAB_SIZE,
        "d_model": 16,
        "num_heads": 2,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "d_ff": 32,
        "max_position": 64,
    }
    settings.update(overrides)
    return ScratchTransformer(ScratchTransformerConfig(**settings))  # type: ignore[arg-type]


@pytest.fixture
def tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig(
        hf_id="t5-small",
        source_prefix="summarize: ",
        max_source_tokens=32,
        max_target_tokens=8,
    )


@pytest.fixture
def summarizer(
    fake_tokenizer: FakeTokenizer, tokenizer_config: TokenizerConfig
) -> ScratchSummarizer:
    return ScratchSummarizer(make_model(), fake_tokenizer, tokenizer_config)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_a_padding_identifier_that_disagrees_with_the_tokeniser_is_refused(
    fake_tokenizer: FakeTokenizer, tokenizer_config: TokenizerConfig
) -> None:
    """Padding would be attended to instead of masked, silently lowering every score."""
    with pytest.raises(ValueError, match="Padding"):
        ScratchSummarizer(
            make_model(pad_token_id=7), fake_tokenizer, tokenizer_config  # type: ignore[arg-type]
        )


def test_an_end_of_sequence_identifier_that_disagrees_is_refused(
    fake_tokenizer: FakeTokenizer, tokenizer_config: TokenizerConfig
) -> None:
    with pytest.raises(ValueError, match="never stop"):
        ScratchSummarizer(
            make_model(eos_token_id=9), fake_tokenizer, tokenizer_config  # type: ignore[arg-type]
        )


def test_a_truncation_longer_than_the_positional_encoding_is_refused(
    fake_tokenizer: FakeTokenizer,
) -> None:
    config = TokenizerConfig(
        hf_id="t5-small", source_prefix="summarize: ", max_source_tokens=512, max_target_tokens=8
    )

    with pytest.raises(ValueError, match="positional encoding"):
        ScratchSummarizer(make_model(), fake_tokenizer, config)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def test_the_corpus_prefix_is_applied_at_inference(
    summarizer: ScratchSummarizer, fake_tokenizer: FakeTokenizer
) -> None:
    """The model was trained on prefixed documents and must be fed prefixed ones."""
    summarizer.encode(DOCUMENTS)

    assert fake_tokenizer.calls[-1] == [f"summarize: {document}" for document in DOCUMENTS]


def test_encoding_returns_identifiers_only(summarizer: ScratchSummarizer) -> None:
    """The model derives its padding mask from the identifiers themselves."""
    source_ids = summarizer.encode(DOCUMENTS)

    assert source_ids.dtype == torch.long
    assert source_ids.shape[0] == len(DOCUMENTS)


def test_an_empty_batch_cannot_be_encoded(summarizer: ScratchSummarizer) -> None:
    with pytest.raises(ValueError, match="empty batch"):
        summarizer.encode([])


# ---------------------------------------------------------------------------
# Summarising
# ---------------------------------------------------------------------------


def test_one_summary_comes_back_per_document(summarizer: ScratchSummarizer) -> None:
    summaries = summarizer.summarize(DOCUMENTS, GenerationConfig(max_new_tokens=4))

    assert len(summaries) == len(DOCUMENTS)
    assert all(isinstance(summary, str) for summary in summaries)


def test_batching_does_not_change_the_summaries(summarizer: ScratchSummarizer) -> None:
    config = GenerationConfig(max_new_tokens=4)

    whole = summarizer.summarize(DOCUMENTS, config, batch_size=8)
    split = summarizer.summarize(DOCUMENTS, config, batch_size=1)

    assert whole == split


def test_an_empty_document_list_returns_an_empty_summary_list(
    summarizer: ScratchSummarizer,
) -> None:
    assert summarizer.summarize([]) == []


def test_a_non_positive_batch_size_is_refused(summarizer: ScratchSummarizer) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        summarizer.summarize(DOCUMENTS, batch_size=0)


def test_the_default_decoding_budget_is_the_corpus_target_length(
    summarizer: ScratchSummarizer,
) -> None:
    """Both sides default to the same budget, so neither is cut shorter than the other."""
    summaries = summarizer.summarize(DOCUMENTS[:1])

    assert len(summaries) == 1


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def test_an_untrained_model_says_so_in_its_description(summarizer: ScratchSummarizer) -> None:
    """A randomly initialised model produces summaries too. The record must not hide it."""
    description = summarizer.describe()

    assert description["model"] == "scratch"
    assert description["mode"] == "untrained"
    assert description["tokenizer"] == "t5-small"
    assert int(description["parameters"]) > 0


def test_a_trained_model_is_marked_as_trained(
    fake_tokenizer: FakeTokenizer, tokenizer_config: TokenizerConfig
) -> None:
    trained = ScratchSummarizer(
        make_model(), fake_tokenizer, tokenizer_config, trained=True  # type: ignore[arg-type]
    )

    assert trained.trained is True
    assert trained.describe()["mode"] == "trained"


def test_the_model_and_the_tokeniser_stay_reachable(
    summarizer: ScratchSummarizer, fake_tokenizer: FakeTokenizer
) -> None:
    assert isinstance(summarizer.model, ScratchTransformer)
    assert summarizer.tokenizer is fake_tokenizer


def test_moving_to_a_device_returns_the_summariser(summarizer: ScratchSummarizer) -> None:
    moved = summarizer.to(torch.device("cpu"))

    assert moved is summarizer
    assert summarizer.device.type == "cpu"
