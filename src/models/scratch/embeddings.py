"""Token embeddings.

Implements the embedding layer of the original Transformer, including the
scaling factor that the paper introduces in section 3.4:

```text
Embedding(x) = E[x] * sqrt(d_model)
```

The scaling matters. Embeddings are initialised with a standard deviation
around ``d_model ** -0.5``, so without the factor their magnitude would be far
below that of the sinusoidal positional encoding, whose values live in
``[-1, 1]``. The position signal would then drown the token signal.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Map token identifiers to scaled dense vectors."""

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int | None = None) -> None:
        """Build the embedding table.

        Args:
            vocab_size: Number of entries in the vocabulary.
            d_model: Width of the residual stream.
            padding_idx: Identifier whose embedding stays at zero and receives
                no gradient.
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.scale = math.sqrt(d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialise the table with a normal law of standard deviation d_model ** -0.5."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.d_model**-0.5)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0.0)

    @property
    def weight(self) -> torch.Tensor:
        """Return the embedding matrix, shared with the output projection when tied.

        Returns:
            A tensor of shape ``(vocab_size, d_model)``.
        """
        return self.embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed a batch of token identifiers.

        Args:
            token_ids: Integer tensor of shape ``(batch, seq_len)``.

        Returns:
            A tensor of shape ``(batch, seq_len, d_model)``, scaled by
            ``sqrt(d_model)``.
        """
        embedded: torch.Tensor = self.embedding(token_ids)
        return embedded * self.scale
