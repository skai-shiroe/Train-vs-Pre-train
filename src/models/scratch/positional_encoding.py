"""Sinusoidal positional encoding.

Self attention is permutation invariant: without a position signal, the model
would see a bag of tokens. The original paper injects that signal additively,
with fixed sinusoids of geometrically spaced frequencies:

```text
PE(pos, 2i)     = sin(pos / 10000 ** (2i / d_model))
PE(pos, 2i + 1) = cos(pos / 10000 ** (2i / d_model))
```

``pos`` indexes the position in the sequence and ``i`` indexes the pair of
dimensions. Two properties motivate this choice:

1. The encoding is deterministic, so it costs no parameter and generalises to
   positions longer than those seen during training.
2. For any fixed offset ``k``, ``PE(pos + k)`` is a linear function of
   ``PE(pos)``. Relative positions are therefore reachable by a linear map,
   which is exactly what an attention head computes.

The exponent is evaluated in log space, as ``exp(-2i * log(10000) / d_model)``,
because ``10000 ** (2i / d_model)`` underflows in float32 for wide models.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Add fixed sinusoidal position information to a sequence of embeddings."""

    def __init__(self, d_model: int, max_position: int = 1024, dropout: float = 0.1) -> None:
        """Precompute the encoding table.

        Args:
            d_model: Width of the residual stream.
            max_position: Longest sequence supported.
            dropout: Dropout applied to the sum of embedding and position.

        Raises:
            ValueError: If ``d_model`` is not even, since the dimensions are
                filled in sine and cosine pairs.
        """
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even to pair sine and cosine, got {d_model}.")

        self.d_model = d_model
        self.max_position = max_position
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_position, dtype=torch.float32).unsqueeze(1)
        pair_index = torch.arange(0, d_model, 2, dtype=torch.float32)
        frequency = torch.exp(-pair_index * math.log(10000.0) / d_model)

        encoding = torch.zeros(max_position, d_model)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)

        # Registered as a buffer: saved with the checkpoint, moved with the
        # module, but never updated by the optimiser.
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Add the position signal to a batch of embeddings.

        Args:
            embeddings: Tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            A tensor of the same shape, with dropout applied.

        Raises:
            ValueError: If the sequence is longer than ``max_position``.
        """
        seq_len = embeddings.size(1)
        if seq_len > self.max_position:
            raise ValueError(
                f"Sequence of length {seq_len} exceeds max_position={self.max_position}. "
                "Increase max_position or truncate the input."
            )

        encoding = self.get_buffer("encoding")
        positioned: torch.Tensor = self.dropout(embeddings + encoding[:, :seq_len])
        return positioned
