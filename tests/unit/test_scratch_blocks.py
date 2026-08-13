"""Unit tests for embeddings, positional encoding and the feed forward block."""

from __future__ import annotations

import math

import pytest
import torch

from src.models.scratch.embeddings import TokenEmbedding
from src.models.scratch.feed_forward import ACTIVATIONS, PositionWiseFeedForward
from src.models.scratch.positional_encoding import SinusoidalPositionalEncoding
from src.utils.seed import set_seed


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    set_seed(1234)


# ---------------------------------------------------------------------------
# TokenEmbedding
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_embedding_produces_the_expected_shape() -> None:
    embedding = TokenEmbedding(vocab_size=100, d_model=32)

    assert embedding(torch.randint(0, 100, (3, 7))).shape == (3, 7, 32)


@pytest.mark.unit
def test_embedding_is_scaled_by_the_square_root_of_d_model() -> None:
    embedding = TokenEmbedding(vocab_size=100, d_model=64)
    token_ids = torch.tensor([[5]])

    scaled = embedding(token_ids)
    raw = embedding.embedding(token_ids)

    assert torch.allclose(scaled, raw * math.sqrt(64), atol=1e-6)


@pytest.mark.unit
def test_the_padding_embedding_starts_at_zero() -> None:
    embedding = TokenEmbedding(vocab_size=100, d_model=32, padding_idx=0)

    assert torch.allclose(embedding.weight[0], torch.zeros(32))


@pytest.mark.unit
def test_the_weight_property_exposes_the_table_for_tying() -> None:
    embedding = TokenEmbedding(vocab_size=100, d_model=32)

    assert embedding.weight is embedding.embedding.weight
    assert embedding.weight.shape == (100, 32)


# ---------------------------------------------------------------------------
# SinusoidalPositionalEncoding
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_positional_encoding_keeps_the_input_shape() -> None:
    encoding = SinusoidalPositionalEncoding(d_model=32, max_position=64, dropout=0.0).eval()

    assert encoding(torch.zeros(2, 10, 32)).shape == (2, 10, 32)


@pytest.mark.unit
def test_the_values_match_the_published_formula() -> None:
    d_model = 16
    encoding = SinusoidalPositionalEncoding(d_model=d_model, max_position=32, dropout=0.0).eval()

    produced = encoding(torch.zeros(1, 5, d_model))[0]

    for position in range(5):
        for pair in range(d_model // 2):
            angle = position / (10000 ** (2 * pair / d_model))
            assert produced[position, 2 * pair] == pytest.approx(math.sin(angle), abs=1e-5)
            assert produced[position, 2 * pair + 1] == pytest.approx(math.cos(angle), abs=1e-5)


@pytest.mark.unit
def test_the_encoding_is_added_not_substituted() -> None:
    encoding = SinusoidalPositionalEncoding(d_model=16, max_position=32, dropout=0.0).eval()
    embeddings = torch.randn(1, 6, 16)

    positioned = encoding(embeddings)
    zero_input = encoding(torch.zeros(1, 6, 16))

    assert torch.allclose(positioned - zero_input, embeddings, atol=1e-6)


@pytest.mark.unit
def test_two_positions_receive_different_signals() -> None:
    encoding = SinusoidalPositionalEncoding(d_model=16, max_position=32, dropout=0.0).eval()

    produced = encoding(torch.zeros(1, 4, 16))[0]

    assert not torch.allclose(produced[0], produced[1])
    assert not torch.allclose(produced[1], produced[2])


@pytest.mark.unit
def test_the_encoding_stays_bounded() -> None:
    encoding = SinusoidalPositionalEncoding(d_model=32, max_position=1024, dropout=0.0).eval()

    produced = encoding(torch.zeros(1, 1024, 32))

    assert float(produced.abs().max()) <= 1.0 + 1e-6


@pytest.mark.unit
def test_the_encoding_is_deterministic() -> None:
    encoding = SinusoidalPositionalEncoding(d_model=16, max_position=32, dropout=0.0).eval()

    assert torch.equal(encoding(torch.zeros(1, 8, 16)), encoding(torch.zeros(1, 8, 16)))


@pytest.mark.unit
def test_the_encoding_carries_no_trainable_parameter() -> None:
    encoding = SinusoidalPositionalEncoding(d_model=16, max_position=32)

    assert list(encoding.parameters()) == []


@pytest.mark.unit
def test_an_odd_width_is_refused() -> None:
    with pytest.raises(ValueError, match="must be even"):
        SinusoidalPositionalEncoding(d_model=15)


@pytest.mark.unit
def test_exceeding_max_position_is_reported() -> None:
    encoding = SinusoidalPositionalEncoding(d_model=16, max_position=8, dropout=0.0).eval()

    with pytest.raises(ValueError, match="exceeds max_position"):
        encoding(torch.zeros(1, 9, 16))


# ---------------------------------------------------------------------------
# PositionWiseFeedForward
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_feed_forward_returns_to_the_model_width() -> None:
    block = PositionWiseFeedForward(d_model=32, d_ff=128, dropout=0.0).eval()

    assert block(torch.randn(2, 5, 32)).shape == (2, 5, 32)


@pytest.mark.unit
def test_feed_forward_treats_positions_independently() -> None:
    block = PositionWiseFeedForward(d_model=16, d_ff=64, dropout=0.0).eval()
    hidden = torch.randn(1, 4, 16)

    whole = block(hidden)
    isolated = block(hidden[:, 2:3])

    assert torch.allclose(whole[:, 2:3], isolated, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("activation", sorted(ACTIVATIONS))
def test_every_declared_activation_works(activation: str) -> None:
    block = PositionWiseFeedForward(32, 64, dropout=0.0, activation=activation).eval()

    assert block(torch.randn(1, 3, 32)).shape == (1, 3, 32)


@pytest.mark.unit
def test_an_unknown_activation_lists_the_available_ones() -> None:
    with pytest.raises(ValueError, match="Available: gelu, relu"):
        PositionWiseFeedForward(32, 64, activation="swish")


@pytest.mark.unit
def test_relu_clips_the_inner_activation() -> None:
    block = PositionWiseFeedForward(d_model=8, d_ff=16, dropout=0.0, activation="relu").eval()

    inner = torch.relu(block.expand(torch.randn(1, 3, 8)))

    assert bool((inner >= 0).all())
