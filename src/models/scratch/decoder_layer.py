"""One decoder layer.

A decoder layer chains three sublayers instead of two:

1. **Masked self attention.** The target attends to itself, under the causal
   mask, so position ``t`` never reads position ``t + 1``.
2. **Cross attention.** This is where the two towers meet. The queries come
   from the decoder, the keys and the values from the encoder memory. The
   decoder asks: given what I have written so far, what part of the source
   document should I read now?
3. **Feed forward network.** Same block as the encoder.

Each sublayer sits inside a residual connection and a layer normalisation,
with the same pre-norm or post-norm choice as the encoder.

The asymmetry of cross attention is the point. Its mask is the *source*
padding mask, not the causal mask: every decoder position may read the whole
source, only the future of the *target* is forbidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.scratch.feed_forward import PositionWiseFeedForward
from src.models.scratch.multi_head_attention import MultiHeadAttention


@dataclass(frozen=True, slots=True)
class DecoderLayerAttentions:
    """Attention weights of one decoder layer, kept for qualitative analysis.

    Attributes:
        self_attention: Weights of the masked self attention, or ``None``.
        cross_attention: Weights of the cross attention, or ``None``.
    """

    self_attention: torch.Tensor | None
    cross_attention: torch.Tensor | None


class DecoderLayer(nn.Module):
    """Masked self attention, then cross attention, then feed forward."""

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
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout, activation)

        self.self_attention_norm = nn.LayerNorm(d_model)
        self.cross_attention_norm = nn.LayerNorm(d_model)
        self.feed_forward_norm = nn.LayerNorm(d_model)

        self.self_attention_dropout = nn.Dropout(dropout)
        self.cross_attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        *,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, DecoderLayerAttentions]:
        """Run one decoder layer.

        Args:
            hidden: Decoder states of shape ``(batch, target_len, d_model)``.
            memory: Encoder output of shape ``(batch, src_len, d_model)``.
            target_mask: Combined causal and target padding mask.
            memory_mask: Source padding mask, applied to cross attention.
            return_weights: Whether to keep the attention weights.

        Returns:
            A pair ``(hidden, attentions)``.
        """
        if self.norm_first:
            normalised = self.self_attention_norm(hidden)
            attended, self_weights = self.self_attention(
                normalised, normalised, normalised, target_mask, return_weights=return_weights
            )
            hidden = hidden + self.self_attention_dropout(attended)

            normalised = self.cross_attention_norm(hidden)
            attended, cross_weights = self.cross_attention(
                normalised, memory, memory, memory_mask, return_weights=return_weights
            )
            hidden = hidden + self.cross_attention_dropout(attended)

            normalised = self.feed_forward_norm(hidden)
            hidden = hidden + self.feed_forward_dropout(self.feed_forward(normalised))
        else:
            attended, self_weights = self.self_attention(
                hidden, hidden, hidden, target_mask, return_weights=return_weights
            )
            hidden = self.self_attention_norm(hidden + self.self_attention_dropout(attended))

            attended, cross_weights = self.cross_attention(
                hidden, memory, memory, memory_mask, return_weights=return_weights
            )
            hidden = self.cross_attention_norm(hidden + self.cross_attention_dropout(attended))

            hidden = self.feed_forward_norm(
                hidden + self.feed_forward_dropout(self.feed_forward(hidden))
            )

        return hidden, DecoderLayerAttentions(
            self_attention=self_weights, cross_attention=cross_weights
        )
