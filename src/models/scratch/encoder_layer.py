"""One encoder layer.

An encoder layer chains two sublayers: multi head self attention, then the
position wise feed forward network. Each sublayer sits inside a residual
connection followed by layer normalisation.

**Residual connections.** ``x + Sublayer(x)`` gives the gradient a path that
skips the sublayer entirely. Without them a six layer stack trains poorly, and
a deeper one not at all.

**Layer normalisation.** Normalising each position over its features keeps the
scale of the residual stream stable from layer to layer. Two placements exist:

```text
post-norm  x = LayerNorm(x + Sublayer(x))          original paper
pre-norm   x = x + Sublayer(LayerNorm(x))          default here
```

Pre-norm leaves the residual path free of any normalisation, so the gradient
reaches the first layer undistorted. It trains reliably without the long
learning rate warmup that post-norm demands, which matters for a model trained
from scratch on a small corpus. Post-norm stays available through
``norm_first=False`` for a faithful reproduction of the paper.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.scratch.feed_forward import PositionWiseFeedForward
from src.models.scratch.multi_head_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    """Self attention followed by a feed forward network, both residual."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        *,
        norm_first: bool = True,
        activation: str = "relu",
    ) -> None:
        """Build the layer.

        Args:
            d_model: Width of the residual stream.
            num_heads: Number of attention heads.
            d_ff: Inner width of the feed forward network.
            dropout: Dropout applied to the output of each sublayer.
            norm_first: Whether to normalise before each sublayer.
            activation: Activation of the feed forward network.
        """
        super().__init__()
        self.norm_first = norm_first

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout, activation)

        self.attention_norm = nn.LayerNorm(d_model)
        self.feed_forward_norm = nn.LayerNorm(d_model)

        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run one encoder layer.

        Args:
            hidden: Tensor of shape ``(batch, src_len, d_model)``.
            mask: Source padding mask, broadcastable to
                ``(batch, heads, src_len, src_len)``.
            return_weights: Whether to return the self attention weights.

        Returns:
            A pair ``(hidden, weights)``.
        """
        if self.norm_first:
            normalised = self.attention_norm(hidden)
            attended, weights = self.self_attention(
                normalised, normalised, normalised, mask, return_weights=return_weights
            )
            hidden = hidden + self.attention_dropout(attended)

            normalised = self.feed_forward_norm(hidden)
            hidden = hidden + self.feed_forward_dropout(self.feed_forward(normalised))
            return hidden, weights

        attended, weights = self.self_attention(
            hidden, hidden, hidden, mask, return_weights=return_weights
        )
        hidden = self.attention_norm(hidden + self.attention_dropout(attended))
        hidden = self.feed_forward_norm(
            hidden + self.feed_forward_dropout(self.feed_forward(hidden))
        )
        return hidden, weights
