"""Configuration of the from scratch Transformer.

Keeping the hyperparameters in one validated object means an experiment file
fully determines the architecture, and that the same object can be logged to
MLflow without any manual copy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ScratchTransformerConfig:
    """Hyperparameters of the encoder decoder Transformer.

    Attributes:
        vocab_size: Size of the shared vocabulary. Comes from the T5 tokeniser.
        d_model: Width of the residual stream.
        num_heads: Number of attention heads. Must divide ``d_model``.
        num_encoder_layers: Depth of the encoder.
        num_decoder_layers: Depth of the decoder.
        d_ff: Inner width of the position wise feed forward network.
        dropout: Dropout probability applied after every sublayer.
        max_position: Longest sequence the positional encoding supports.
        pad_token_id: Padding identifier, masked out of every attention.
        eos_token_id: End of sequence identifier, stops generation.
        decoder_start_token_id: Token fed first to the decoder. The T5
            tokeniser has no beginning of sequence token, so the padding
            identifier plays that role, exactly as T5 itself does.
        tie_embeddings: Whether the output projection reuses the embedding
            matrix. Tying saves ``vocab_size * d_model`` parameters and helps
            a model trained on a small corpus.
        norm_first: Whether to normalise before each sublayer. Pre-norm is the
            default because it trains far more reliably from scratch without a
            long warmup. Set to ``False`` to reproduce the post-norm layout of
            the original paper.
    """

    vocab_size: int
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_position: int = 1024
    pad_token_id: int = 0
    eos_token_id: int = 1
    decoder_start_token_id: int = 0
    tie_embeddings: bool = True
    norm_first: bool = True

    def __post_init__(self) -> None:
        """Validate the hyperparameters.

        Raises:
            ValueError: If a dimension is not strictly positive, if the number
                of heads does not divide ``d_model``, or if the dropout lies
                outside ``[0, 1)``.
        """
        positive = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "max_position": self.max_position,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be strictly positive, got {value}.")

        if self.num_encoder_layers <= 0 or self.num_decoder_layers <= 0:
            raise ValueError("The encoder and the decoder need at least one layer each.")

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads}); "
                "each head receives d_model / num_heads dimensions."
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must lie in [0, 1), got {self.dropout}.")

    @property
    def d_head(self) -> int:
        """Return the dimension of a single attention head.

        This is the ``d_k`` of the attention formula, whose square root scales
        the dot products.

        Returns:
            ``d_model`` divided by the number of heads.
        """
        return self.d_model // self.num_heads

    def to_dict(self) -> dict[str, Any]:
        """Render the configuration as a flat mapping.

        Returns:
            A mapping suitable for MLflow parameter logging.
        """
        return asdict(self)
