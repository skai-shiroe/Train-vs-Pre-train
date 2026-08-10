"""Position wise feed forward network.

Section 3.3 of the paper. Applied independently and identically to every
position:

```text
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
```

The role of the block is complementary to attention. Attention *moves*
information between positions but combines it linearly. The feed forward
network *transforms* the information at each position, non linearly, without
looking at the neighbours. Stacking the two alternately is what gives depth its
value.

The inner width ``d_ff`` is conventionally four times ``d_model``. Expanding
then projecting back gives the non linearity room to work in.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

#: Activations available to the experiments. ReLU is the choice of the original
#: paper; GELU is the modern default and converges slightly faster on small
#: corpora.
ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": F.relu,
    "gelu": F.gelu,
}


class PositionWiseFeedForward(nn.Module):
    """Two linear layers with a non linearity in between."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        """Build the block.

        Args:
            d_model: Width of the residual stream, in and out.
            d_ff: Inner width.
            dropout: Dropout applied after the activation.
            activation: Name of the activation, ``relu`` or ``gelu``.

        Raises:
            ValueError: If the activation name is unknown.
        """
        super().__init__()
        if activation not in ACTIVATIONS:
            available = ", ".join(sorted(ACTIVATIONS))
            raise ValueError(f"Unknown activation {activation!r}. Available: {available}.")

        self.expand = nn.Linear(d_model, d_ff)
        self.project = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation_name = activation
        self._activation = ACTIVATIONS[activation]
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialise both layers with Xavier uniform and zero biases."""
        for layer in (self.expand, self.project):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Transform every position independently.

        Args:
            hidden: Tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            A tensor of the same shape.
        """
        expanded: torch.Tensor = self.expand(hidden)
        activated = self.dropout(self._activation(expanded))
        projected: torch.Tensor = self.project(activated)
        return projected
