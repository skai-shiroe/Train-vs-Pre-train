"""The encoder stack.

Embedding, positional encoding, then ``N`` identical encoder layers. The
output, called the *memory*, is a contextual representation of the source
document: one vector per source position, each aware of the whole document.

A final layer normalisation is applied when the stack is pre-norm. Without it
the output of the last layer would leave the residual stream unnormalised,
since in pre-norm every normalisation sits *inside* a residual branch.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.scratch.config import ScratchTransformerConfig
from src.models.scratch.embeddings import TokenEmbedding
from src.models.scratch.encoder_layer import EncoderLayer
from src.models.scratch.positional_encoding import SinusoidalPositionalEncoding


class Encoder(nn.Module):
    """Stack of encoder layers producing the source memory."""

    def __init__(self, config: ScratchTransformerConfig, embedding: TokenEmbedding) -> None:
        """Build the stack.

        Args:
            config: Architecture hyperparameters.
            embedding: Token embedding, shared with the decoder so that both
                towers speak the same vocabulary.
        """
        super().__init__()
        self.config = config
        self.embedding = embedding
        self.positional_encoding = SinusoidalPositionalEncoding(
            config.d_model, config.max_position, config.dropout
        )
        self.layers = nn.ModuleList(
            EncoderLayer(
                config.d_model,
                config.num_heads,
                config.d_ff,
                config.dropout,
                norm_first=config.norm_first,
            )
            for _ in range(config.num_encoder_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model) if config.norm_first else nn.Identity()

    def forward(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor | None = None,
        *,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode a batch of source documents.

        Args:
            source_ids: Integer tensor of shape ``(batch, src_len)``.
            source_mask: Source padding mask of shape ``(batch, 1, 1, src_len)``.
            return_weights: Whether to collect the self attention weights.

        Returns:
            A pair ``(memory, weights)`` where ``memory`` has shape
            ``(batch, src_len, d_model)`` and ``weights`` holds one tensor per
            layer, empty unless ``return_weights`` is set.
        """
        hidden = self.positional_encoding(self.embedding(source_ids))

        collected: list[torch.Tensor] = []
        for layer in self.layers:
            hidden, weights = layer(hidden, source_mask, return_weights=return_weights)
            if weights is not None:
                collected.append(weights)

        return self.final_norm(hidden), collected
