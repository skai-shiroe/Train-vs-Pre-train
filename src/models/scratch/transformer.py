"""The full encoder decoder Transformer.

Assembles the components implemented in this package:

```text
source_ids ---> Encoder ---> memory
                              |
target_ids  ---> Decoder <----+ cross attention
                  |
                logits
```

The model is trained with teacher forcing. The decoder receives the target
sequence shifted right by one position and prefixed with
``decoder_start_token_id``, and it predicts the unshifted target. Position
``t`` of the decoder input holds token ``t - 1`` of the target, so predicting
token ``t`` only requires what the causal mask allows it to see.

The T5 tokeniser has no beginning of sequence token. Following T5 itself, the
padding identifier is used to start the decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.models.scratch.config import ScratchTransformerConfig
from src.models.scratch.decoder import Decoder
from src.models.scratch.decoder_layer import DecoderLayerAttentions
from src.models.scratch.embeddings import TokenEmbedding
from src.models.scratch.encoder import Encoder
from src.models.scratch.masks import build_decoder_mask, build_padding_mask

#: Value ignored by the cross entropy loss, matching the data pipeline.
IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True)
class TransformerOutput:
    """Result of a forward pass.

    Attributes:
        logits: Scores of shape ``(batch, target_len, vocab_size)``.
        loss: Cross entropy loss, present only when labels were supplied.
        encoder_attentions: Self attention weights per encoder layer.
        decoder_attentions: Self and cross attention weights per decoder layer.
    """

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    encoder_attentions: tuple[torch.Tensor, ...] = ()
    decoder_attentions: tuple[DecoderLayerAttentions, ...] = ()


def shift_target_right(target_ids: torch.Tensor, decoder_start_token_id: int) -> torch.Tensor:
    """Build the decoder input from the target sequence.

    The target is shifted one position to the right and the free slot is filled
    with the start token. Without this shift the decoder would be given the
    token it must predict.

    Args:
        target_ids: Integer tensor of shape ``(batch, target_len)``.
        decoder_start_token_id: Token placed at position zero.

    Returns:
        A tensor of the same shape as ``target_ids``.
    """
    shifted = target_ids.new_full(target_ids.shape, decoder_start_token_id)
    shifted[:, 1:] = target_ids[:, :-1]
    return shifted


class ScratchTransformer(nn.Module):
    """Encoder decoder Transformer built component by component."""

    def __init__(self, config: ScratchTransformerConfig) -> None:
        """Build the model.

        Args:
            config: Architecture hyperparameters.
        """
        super().__init__()
        self.config = config

        # One embedding table shared by both towers and, when tied, by the
        # output projection. Source and target live in the same vocabulary.
        self.embedding = TokenEmbedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.encoder = Encoder(config, self.embedding)
        self.decoder = Decoder(config, self.embedding)

    @property
    def num_parameters(self) -> int:
        """Return the number of trainable parameters.

        Returns:
            The parameter count, counting a tied matrix once.
        """
        seen: set[int] = set()
        total = 0
        for parameter in self.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                total += parameter.numel()
        return total

    def encode(
        self, source_ids: torch.Tensor, *, return_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Run the encoder and return the memory with its mask.

        Args:
            source_ids: Integer tensor of shape ``(batch, src_len)``.
            return_weights: Whether to collect the attention weights.

        Returns:
            A triple ``(memory, source_mask, weights)``.
        """
        source_mask = build_padding_mask(source_ids, self.config.pad_token_id)
        memory, weights = self.encoder(source_ids, source_mask, return_weights=return_weights)
        return memory, source_mask, weights

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        *,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, list[DecoderLayerAttentions]]:
        """Run the decoder over an already built decoder input.

        Args:
            decoder_input_ids: Integer tensor of shape ``(batch, target_len)``.
            memory: Encoder output.
            memory_mask: Source padding mask.
            return_weights: Whether to collect the attention weights.

        Returns:
            A pair ``(logits, attentions)``.
        """
        target_mask = build_decoder_mask(decoder_input_ids, self.config.pad_token_id)
        decoded: tuple[torch.Tensor, list[DecoderLayerAttentions]] = self.decoder(
            decoder_input_ids,
            memory,
            target_mask,
            memory_mask,
            return_weights=return_weights,
        )
        return decoded

    def forward(
        self,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        *,
        return_weights: bool = False,
    ) -> TransformerOutput:
        """Run a full forward pass, optionally computing the loss.

        Args:
            source_ids: Integer tensor of shape ``(batch, src_len)``.
            target_ids: Target sequence with padding kept. Used to build the
                decoder input by shifting, unless ``decoder_input_ids`` is given.
            labels: Target sequence with padding replaced by
                :data:`IGNORE_INDEX`. Required to compute the loss.
            decoder_input_ids: Explicit decoder input, bypassing the shift.
            return_weights: Whether to collect the attention weights.

        Returns:
            The forward output.

        Raises:
            ValueError: If neither ``target_ids`` nor ``decoder_input_ids`` is
                supplied.
        """
        if decoder_input_ids is None:
            if target_ids is None:
                raise ValueError("Supply either target_ids or decoder_input_ids.")
            decoder_input_ids = shift_target_right(target_ids, self.config.decoder_start_token_id)

        memory, source_mask, encoder_weights = self.encode(
            source_ids, return_weights=return_weights
        )
        logits, decoder_weights = self.decode(
            decoder_input_ids, memory, source_mask, return_weights=return_weights
        )

        loss: torch.Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )

        return TransformerOutput(
            logits=logits,
            loss=loss,
            encoder_attentions=tuple(encoder_weights),
            decoder_attentions=tuple(decoder_weights),
        )
