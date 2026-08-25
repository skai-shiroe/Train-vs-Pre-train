"""The decoder stack.

Embedding, positional encoding, then ``N`` identical decoder layers, then the
projection back to the vocabulary.

**Output projection.** The last hidden state of each position is mapped to one
score per vocabulary entry. When ``tie_embeddings`` is set, the projection
reuses the transpose of the embedding matrix. Tying saves
``vocab_size * d_model`` parameters, which is 16 million of the 60 million of
the default configuration with the T5 vocabulary, and it regularises a model
trained on a small corpus by forcing the input and output views of a token to
agree.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.scratch.config import ScratchTransformerConfig
from src.models.scratch.decoder_layer import DecoderLayer, DecoderLayerAttentions
from src.models.scratch.embeddings import TokenEmbedding
from src.models.scratch.positional_encoding import SinusoidalPositionalEncoding


class Decoder(nn.Module):
    """Stack of decoder layers followed by the projection to the vocabulary."""

    def __init__(self, config: ScratchTransformerConfig, embedding: TokenEmbedding) -> None:
        """Build the stack.

        Args:
            config: Architecture hyperparameters.
            embedding: Token embedding, shared with the encoder.
        """
        super().__init__()
        self.config = config
        self.embedding = embedding
        self.positional_encoding = SinusoidalPositionalEncoding(
            config.d_model, config.max_position, config.dropout
        )
        self.layers = nn.ModuleList(
            DecoderLayer(
                config.d_model,
                config.num_heads,
                config.d_ff,
                config.dropout,
                norm_first=config.norm_first,
            )
            for _ in range(config.num_decoder_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model) if config.norm_first else nn.Identity()

        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.output_projection.weight = embedding.weight
        else:
            nn.init.normal_(self.output_projection.weight, mean=0.0, std=config.d_model**-0.5)

    def forward(
        self,
        target_ids: torch.Tensor,
        memory: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        *,
        return_weights: bool = False,
        last_position_only: bool = False,
    ) -> tuple[torch.Tensor, list[DecoderLayerAttentions]]:
        """Decode a batch of target sequences against the source memory.

        Args:
            target_ids: Integer tensor of shape ``(batch, target_len)``.
            memory: Encoder output of shape ``(batch, src_len, d_model)``.
            target_mask: Combined causal and target padding mask.
            memory_mask: Source padding mask, applied to cross attention.
            return_weights: Whether to collect the attention weights.
            last_position_only: Whether to project the last position alone.
                Generation reads exactly one row of the result and throws the
                rest away, and that rest is what the projection makes
                expensive: it is the only tensor of the model whose last
                dimension is the vocabulary. Decoding four beams over eight
                documents with a prefix of 128 tokens builds 32 x 128 x 32128
                floats, 526 MB, to use 4 MB of them. The prefix grows by one
                token per step, so the caching allocator ends up holding one
                block of every size between the two, and an evaluation that
                fits on the card in principle spills into system memory
                instead. The values are unchanged: this is the same
                projection, applied to one position.

        Returns:
            A pair ``(logits, attentions)`` where ``logits`` has shape
            ``(batch, target_len, vocab_size)``, or ``(batch, 1, vocab_size)``
            when only the last position was asked for.
        """
        hidden = self.positional_encoding(self.embedding(target_ids))

        collected: list[DecoderLayerAttentions] = []
        for layer in self.layers:
            hidden, attentions = layer(
                hidden,
                memory,
                target_mask,
                memory_mask,
                return_weights=return_weights,
            )
            if return_weights:
                collected.append(attentions)

        if last_position_only:
            hidden = hidden[:, -1:, :]
        return self.output_projection(self.final_norm(hidden)), collected
