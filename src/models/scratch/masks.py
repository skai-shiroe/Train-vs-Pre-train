"""Attention masks.

Two masks are needed by an encoder decoder Transformer.

**Padding mask.** Batches hold sequences of different lengths, padded to the
longest one. Padding carries no information, so every query must be forbidden
from attending to it. Without this mask the model would learn to read the
padding, and its predictions would depend on the composition of the batch.

**Causal mask.** The decoder is trained with teacher forcing: the whole target
sequence is fed at once. Position ``t`` must therefore only see positions
``<= t``, otherwise the model reads the answer it is asked to predict and the
training loss collapses while generation stays random.

Both masks are boolean and follow the same convention: ``True`` means *keep*,
``False`` means *forbid*. They are broadcast to the attention score shape
``(batch, num_heads, query_len, key_len)``.
"""

from __future__ import annotations

import torch


def build_padding_mask(token_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Build the key padding mask of a batch.

    Args:
        token_ids: Integer tensor of shape ``(batch, seq_len)``.
        pad_token_id: Identifier of the padding token.

    Returns:
        A boolean tensor of shape ``(batch, 1, 1, seq_len)``, ``True`` on real
        tokens. The two singleton dimensions broadcast over the heads and over
        the queries.
    """
    return (token_ids != pad_token_id).unsqueeze(1).unsqueeze(2)


def build_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Build the lower triangular mask forbidding attention to the future.

    Args:
        seq_len: Length of the target sequence.
        device: Device the mask is created on.

    Returns:
        A boolean tensor of shape ``(1, 1, seq_len, seq_len)``, ``True`` on and
        below the diagonal.

    Raises:
        ValueError: If ``seq_len`` is not strictly positive.
    """
    if seq_len <= 0:
        raise ValueError(f"Sequence length must be strictly positive, got {seq_len}.")

    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
    return causal.unsqueeze(0).unsqueeze(1)


def build_decoder_mask(
    target_ids: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """Combine the causal mask with the padding mask of the target sequence.

    A decoder position must satisfy both constraints at once: it may not look
    ahead, and it may not look at padding.

    Args:
        target_ids: Integer tensor of shape ``(batch, target_len)``.
        pad_token_id: Identifier of the padding token.

    Returns:
        A boolean tensor of shape ``(batch, 1, target_len, target_len)``.
    """
    target_len = target_ids.size(1)
    causal = build_causal_mask(target_len, device=target_ids.device)
    padding = build_padding_mask(target_ids, pad_token_id)
    return causal & padding


def masked_fill_value(dtype: torch.dtype) -> float:
    """Return the value written into forbidden attention scores.

    Using the smallest finite value of the dtype rather than negative infinity
    keeps the softmax well defined even when a whole row is masked, which
    happens on fully padded sequences. Negative infinity would produce a row of
    zeros divided by zero, hence NaN.

    Args:
        dtype: Floating point type of the attention scores.

    Returns:
        The fill value for masked positions.
    """
    return float(torch.finfo(dtype).min)
