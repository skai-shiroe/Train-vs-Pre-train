"""Unit tests for scaled dot product attention and multi head attention.

These are the tests the specification names first: the attention formula is the
one piece of the architecture whose correctness cannot be checked by reading
the loss curve.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.models.scratch.attention import scaled_dot_product_attention
from src.models.scratch.multi_head_attention import MultiHeadAttention
from src.utils.seed import set_seed


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    set_seed(1234)


# ---------------------------------------------------------------------------
# scaled_dot_product_attention
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_output_keeps_the_query_shape() -> None:
    query = torch.randn(2, 4, 7, 16)
    key = torch.randn(2, 4, 11, 16)
    value = torch.randn(2, 4, 11, 16)

    context, weights = scaled_dot_product_attention(query, key, value)

    assert context.shape == (2, 4, 7, 16)
    assert weights.shape == (2, 4, 7, 11)


@pytest.mark.unit
def test_weights_form_a_probability_distribution() -> None:
    query = torch.randn(2, 3, 5, 8)
    key = torch.randn(2, 3, 6, 8)
    value = torch.randn(2, 3, 6, 8)

    _, weights = scaled_dot_product_attention(query, key, value)

    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 3, 5), atol=1e-5)
    assert bool((weights >= 0).all())


@pytest.mark.unit
def test_the_formula_matches_a_manual_computation() -> None:
    query = torch.randn(1, 1, 3, 4)
    key = torch.randn(1, 1, 5, 4)
    value = torch.randn(1, 1, 5, 4)

    expected_scores = query @ key.transpose(-2, -1) / math.sqrt(4)
    expected_weights = torch.softmax(expected_scores, dim=-1)
    expected_context = expected_weights @ value

    context, weights = scaled_dot_product_attention(query, key, value)

    assert torch.allclose(weights, expected_weights, atol=1e-6)
    assert torch.allclose(context, expected_context, atol=1e-6)


@pytest.mark.unit
def test_scores_are_divided_by_the_square_root_of_d_k() -> None:
    # Without the division the softmax over these two keys would be far
    # sharper. Comparing against the undivided version pins the scaling down.
    d_k = 64
    query = torch.ones(1, 1, 1, d_k)
    key = torch.stack([torch.ones(d_k), torch.zeros(d_k)]).view(1, 1, 2, d_k)
    value = torch.eye(2).view(1, 1, 2, 2).expand(1, 1, 2, 2).reshape(1, 1, 2, 2)

    _, weights = scaled_dot_product_attention(query, key, value)

    scaled_logit = d_k / math.sqrt(d_k)
    expected = torch.softmax(torch.tensor([scaled_logit, 0.0]), dim=-1)

    assert torch.allclose(weights[0, 0, 0], expected, atol=1e-6)


@pytest.mark.unit
def test_a_masked_position_receives_no_weight() -> None:
    query = torch.randn(1, 1, 2, 8)
    key = torch.randn(1, 1, 4, 8)
    value = torch.randn(1, 1, 4, 8)
    mask = torch.tensor([[[[True, True, False, False]]]])

    _, weights = scaled_dot_product_attention(query, key, value, mask=mask)

    assert torch.allclose(weights[..., 2:], torch.zeros(1, 1, 2, 2), atol=1e-7)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 1, 2), atol=1e-5)


@pytest.mark.unit
def test_masking_makes_the_forbidden_values_irrelevant() -> None:
    query = torch.randn(1, 1, 2, 8)
    key = torch.randn(1, 1, 4, 8)
    value = torch.randn(1, 1, 4, 8)
    mask = torch.tensor([[[[True, True, False, False]]]])

    first, _ = scaled_dot_product_attention(query, key, value, mask=mask)

    polluted = value.clone()
    polluted[:, :, 2:] = 1e4
    second, _ = scaled_dot_product_attention(query, key, polluted, mask=mask)

    assert torch.allclose(first, second, atol=1e-4)


@pytest.mark.unit
def test_a_fully_masked_row_does_not_produce_nan() -> None:
    # A batch can hold a fully padded sequence. With negative infinity the
    # softmax would divide zero by zero; the smallest finite value keeps it defined.
    query = torch.randn(1, 1, 1, 8)
    key = torch.randn(1, 1, 3, 8)
    value = torch.randn(1, 1, 3, 8)
    mask = torch.zeros(1, 1, 1, 3, dtype=torch.bool)

    context, weights = scaled_dot_product_attention(query, key, value, mask=mask)

    assert not bool(torch.isnan(context).any())
    assert not bool(torch.isnan(weights).any())


@pytest.mark.unit
def test_mismatched_key_and_value_lengths_are_refused() -> None:
    query = torch.randn(1, 1, 2, 8)
    key = torch.randn(1, 1, 4, 8)
    value = torch.randn(1, 1, 5, 8)

    with pytest.raises(ValueError, match="must share their length"):
        scaled_dot_product_attention(query, key, value)


@pytest.mark.unit
def test_attention_is_permutation_equivariant_over_the_queries() -> None:
    query = torch.randn(1, 1, 4, 8)
    key = torch.randn(1, 1, 6, 8)
    value = torch.randn(1, 1, 6, 8)
    order = torch.tensor([3, 0, 2, 1])

    direct, _ = scaled_dot_product_attention(query, key, value)
    permuted, _ = scaled_dot_product_attention(query[:, :, order], key, value)

    assert torch.allclose(permuted, direct[:, :, order], atol=1e-6)


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_multi_head_output_keeps_the_model_width() -> None:
    attention = MultiHeadAttention(d_model=32, num_heads=4, dropout=0.0).eval()
    hidden = torch.randn(3, 7, 32)

    output, weights = attention(hidden, hidden, hidden)

    assert output.shape == (3, 7, 32)
    assert weights is None


@pytest.mark.unit
def test_weights_are_returned_on_demand() -> None:
    attention = MultiHeadAttention(d_model=32, num_heads=4, dropout=0.0).eval()
    hidden = torch.randn(3, 7, 32)

    _, weights = attention(hidden, hidden, hidden, return_weights=True)

    assert weights is not None
    assert weights.shape == (3, 4, 7, 7)


@pytest.mark.unit
def test_cross_attention_accepts_different_lengths() -> None:
    attention = MultiHeadAttention(d_model=32, num_heads=4, dropout=0.0).eval()
    target = torch.randn(2, 5, 32)
    memory = torch.randn(2, 9, 32)

    output, weights = attention(target, memory, memory, return_weights=True)

    assert output.shape == (2, 5, 32)
    assert weights is not None
    assert weights.shape == (2, 4, 5, 9)


@pytest.mark.unit
def test_head_count_must_divide_the_model_width() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        MultiHeadAttention(d_model=30, num_heads=4)


@pytest.mark.unit
def test_splitting_and_merging_heads_is_a_round_trip() -> None:
    attention = MultiHeadAttention(d_model=32, num_heads=4, dropout=0.0)
    hidden = torch.randn(2, 6, 32)

    restored = attention._merge_heads(attention._split_heads(hidden))

    assert torch.allclose(restored, hidden, atol=1e-7)


@pytest.mark.unit
def test_the_heads_split_the_width_evenly() -> None:
    attention = MultiHeadAttention(d_model=64, num_heads=8, dropout=0.0)

    assert attention.d_head == 8
    assert attention.d_head * attention.num_heads == attention.d_model


@pytest.mark.unit
def test_padding_does_not_change_the_real_positions() -> None:
    attention = MultiHeadAttention(d_model=16, num_heads=2, dropout=0.0).eval()
    hidden = torch.randn(1, 4, 16)
    padded = torch.cat([hidden, torch.randn(1, 3, 16)], dim=1)

    mask = torch.tensor([[[[True] * 4 + [False] * 3]]])

    without, _ = attention(hidden, hidden, hidden)
    with_padding, _ = attention(padded, padded, padded, mask)

    assert torch.allclose(with_padding[:, :4], without, atol=1e-5)
