"""Multi head attention.

A single attention head averages the values it selects, so it can only express
one relation at a time. Section 3.2.2 of the paper runs several heads in
parallel on projected subspaces:

```text
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
head_i             = Attention(Q W_Qi, K W_Ki, V W_Vi)
```

Each head works in ``d_head = d_model / num_heads`` dimensions, so the total
cost matches that of a single head of width ``d_model``. Splitting the width
buys expressiveness for free: one head can track the subject of a sentence
while another tracks its position.

**Implementation.** Rather than ``h`` separate small projections, one wide
projection per role is applied and then reshaped into heads. The two are
mathematically equivalent, and the wide version is a single matrix product.

The concatenation of the heads is a reshape: the head dimension is moved back
next to the feature dimension and the two are merged. ``W_O`` then mixes the
information the heads gathered independently.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.scratch.attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """Project into several subspaces, attend in each, then recombine."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        """Build the four projections of the block.

        Args:
            d_model: Width of the residual stream.
            num_heads: Number of parallel heads.
            dropout: Dropout applied to the attention weights.

        Raises:
            ValueError: If ``num_heads`` does not divide ``d_model``.
        """
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialise the projections with Xavier uniform and zero biases."""
        for projection in (
            self.query_projection,
            self.key_projection,
            self.value_projection,
            self.output_projection,
        ):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def _split_heads(self, projected: torch.Tensor) -> torch.Tensor:
        """Reshape ``(batch, seq, d_model)`` into ``(batch, heads, seq, d_head)``.

        Args:
            projected: Output of one of the linear projections.

        Returns:
            The same values, viewed as separate heads.
        """
        batch, seq_len, _ = projected.shape
        return projected.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, context: torch.Tensor) -> torch.Tensor:
        """Concatenate the heads back into ``(batch, seq, d_model)``.

        Args:
            context: Tensor of shape ``(batch, heads, seq, d_head)``.

        Returns:
            The concatenated context.
        """
        batch, _, seq_len, _ = context.shape
        return context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run multi head attention.

        Self attention passes the same tensor as query, key and value. Cross
        attention passes the decoder states as query and the encoder memory as
        key and value.

        Args:
            query: Tensor of shape ``(batch, query_len, d_model)``.
            key: Tensor of shape ``(batch, key_len, d_model)``.
            value: Tensor of shape ``(batch, key_len, d_model)``.
            mask: Optional boolean mask broadcastable to
                ``(batch, heads, query_len, key_len)``.
            return_weights: Whether to return the attention weights. They are
                useful for the qualitative analysis but cost memory, so they
                are discarded by default.

        Returns:
            A pair ``(output, weights)``. ``weights`` is ``None`` unless
            ``return_weights`` is set.
        """
        heads_query = self._split_heads(self.query_projection(query))
        heads_key = self._split_heads(self.key_projection(key))
        heads_value = self._split_heads(self.value_projection(value))

        context, weights = scaled_dot_product_attention(
            heads_query,
            heads_key,
            heads_value,
            mask=mask,
            dropout=self.dropout,
        )

        output = self.output_projection(self._merge_heads(context))
        return output, weights if return_weights else None
