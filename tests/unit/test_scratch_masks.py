"""Unit tests for the attention masks."""

from __future__ import annotations

import pytest
import torch

from src.models.scratch.masks import (
    build_causal_mask,
    build_decoder_mask,
    build_padding_mask,
    masked_fill_value,
)

PAD = 0


@pytest.mark.unit
def test_padding_mask_has_the_broadcastable_shape() -> None:
    token_ids = torch.tensor([[5, 6, 7, PAD, PAD]])

    mask = build_padding_mask(token_ids, PAD)

    assert mask.shape == (1, 1, 1, 5)


@pytest.mark.unit
def test_padding_mask_keeps_real_tokens_and_forbids_padding() -> None:
    token_ids = torch.tensor([[5, 6, PAD], [PAD, PAD, 9]])

    mask = build_padding_mask(token_ids, PAD)

    assert mask[0, 0, 0].tolist() == [True, True, False]
    assert mask[1, 0, 0].tolist() == [False, False, True]


@pytest.mark.unit
def test_padding_mask_is_boolean() -> None:
    mask = build_padding_mask(torch.tensor([[1, PAD]]), PAD)

    assert mask.dtype is torch.bool


@pytest.mark.unit
def test_causal_mask_is_lower_triangular() -> None:
    mask = build_causal_mask(4)

    assert mask.shape == (1, 1, 4, 4)
    assert mask[0, 0].tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]


@pytest.mark.unit
def test_causal_mask_always_lets_a_position_see_itself() -> None:
    mask = build_causal_mask(6)

    assert bool(mask[0, 0].diagonal().all())


@pytest.mark.unit
def test_causal_mask_forbids_every_future_position() -> None:
    mask = build_causal_mask(6)

    assert not bool(mask[0, 0].triu(diagonal=1).any())


@pytest.mark.unit
@pytest.mark.parametrize("length", [0, -3])
def test_a_non_positive_length_is_refused(length: int) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        build_causal_mask(length)


@pytest.mark.unit
def test_decoder_mask_combines_both_constraints() -> None:
    target_ids = torch.tensor([[7, 8, PAD]])

    mask = build_decoder_mask(target_ids, PAD)

    assert mask.shape == (1, 1, 3, 3)
    # Row 2 may look back at positions 0 and 1, never at the padded position 2.
    assert mask[0, 0, 2].tolist() == [True, True, False]
    # Row 0 sees only itself, and position 0 is a real token.
    assert mask[0, 0, 0].tolist() == [True, False, False]


@pytest.mark.unit
def test_decoder_mask_is_never_more_permissive_than_the_causal_mask() -> None:
    target_ids = torch.tensor([[7, 8, 9, PAD]])

    decoder = build_decoder_mask(target_ids, PAD)
    causal = build_causal_mask(4)

    assert bool((decoder & ~causal).sum() == 0)


@pytest.mark.unit
def test_decoder_mask_follows_the_padding_of_each_row() -> None:
    target_ids = torch.tensor([[7, 8, 9], [7, PAD, PAD]])

    mask = build_decoder_mask(target_ids, PAD)

    assert mask[1, 0, 2].tolist() == [True, False, False]
    assert mask[0, 0, 2].tolist() == [True, True, True]


@pytest.mark.unit
def test_the_mask_lives_on_the_same_device_as_its_input() -> None:
    token_ids = torch.tensor([[1, 2, PAD]])

    assert build_decoder_mask(token_ids, PAD).device == token_ids.device


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fill_value_is_finite_for_every_float_type(dtype: torch.dtype) -> None:
    value = masked_fill_value(dtype)

    assert value < 0
    assert torch.isfinite(torch.tensor(value, dtype=dtype))
