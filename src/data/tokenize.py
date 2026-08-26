"""Tokenisation shared by the from scratch model and the T5 runs.

Every model of the comparison uses the T5 tokeniser. The comparison then rests
on the same vocabulary, the same segmentation and the same truncation, so a
score gap can only come from the models themselves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import cast

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.data.config import TokenizerConfig
from src.data.example import Example

#: Value ignored by the cross entropy loss. Padding positions in the labels are
#: set to it so that the model is never trained to predict padding.
IGNORE_INDEX = -100


@lru_cache(maxsize=4)
def build_tokenizer(hf_id: str) -> PreTrainedTokenizerBase:
    """Load a tokeniser, cached per identifier.

    Args:
        hf_id: Hugging Face identifier, for example ``t5-small``.

    Returns:
        The loaded tokeniser.
    """
    # AutoTokenizer.from_pretrained is not annotated upstream, so its return
    # type reaches us as Any. The cast restores the contract for the callers.
    return cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(hf_id),  # type: ignore[no-untyped-call]
    )


@dataclass(frozen=True, slots=True)
class EncodedBatch:
    """A tokenised batch ready to be fed to a sequence to sequence model.

    Attributes:
        input_ids: Token identifiers of the documents.
        attention_mask: Ones on real tokens, zeros on padding.
        labels: Token identifiers of the summaries, padding replaced by
            :data:`IGNORE_INDEX`.
        target_ids: Token identifiers of the summaries, padding kept. Used to
            build the decoder input by shifting.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    target_ids: torch.Tensor

    def to(self, device: torch.device) -> EncodedBatch:
        """Move every tensor of the batch to a device.

        Args:
            device: Target device.

        Returns:
            A new batch whose tensors live on ``device``.
        """
        return EncodedBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
            target_ids=self.target_ids.to(device),
        )

    def __len__(self) -> int:
        """Return the number of examples in the batch.

        Returns:
            The batch size.
        """
        return int(self.input_ids.shape[0])


def encode_batch(
    examples: Sequence[Example],
    tokenizer: PreTrainedTokenizerBase,
    config: TokenizerConfig,
) -> EncodedBatch:
    """Tokenise a batch of examples.

    Args:
        examples: The examples to encode.
        tokenizer: The shared tokeniser.
        config: Prefix and truncation lengths.

    Returns:
        The encoded batch.

    Raises:
        ValueError: If the batch is empty.
    """
    if not examples:
        raise ValueError("Cannot encode an empty batch.")

    sources = [config.source_prefix + example.source for example in examples]
    targets = [example.target for example in examples]

    encoded_sources = tokenizer(
        sources,
        max_length=config.max_source_tokens,
        truncation=True,
        padding="longest",
        return_tensors="pt",
    )
    encoded_targets = tokenizer(
        targets,
        max_length=config.max_target_tokens,
        truncation=True,
        padding="longest",
        return_tensors="pt",
    )

    target_ids = encoded_targets["input_ids"]
    labels = target_ids.masked_fill(encoded_targets["attention_mask"] == 0, IGNORE_INDEX)

    return EncodedBatch(
        input_ids=encoded_sources["input_ids"],
        attention_mask=encoded_sources["attention_mask"],
        labels=labels,
        target_ids=target_ids,
    )


def source_token_lengths(
    examples: Sequence[Example],
    tokenizer: PreTrainedTokenizerBase,
    config: TokenizerConfig,
) -> list[int]:
    """Measure the length each document has once truncated.

    This is the quantity that drives the width of a batch, so it is the sorting
    key of the length grouped sampler. A document cut at ``max_source_tokens``
    costs exactly that, no matter how long it was before truncation, which is
    why the measurement applies the truncation rather than reporting the raw
    length. Use :func:`token_lengths` to study the untruncated distribution.

    Only the integer lengths are kept, never the token identifiers, so the
    memory footprint stays negligible next to the corpus itself.

    Args:
        examples: The examples to measure.
        tokenizer: The shared tokeniser.
        config: Provides the source prefix and the truncation length.

    Returns:
        One length per example, in the order they were given.
    """
    sources = [config.source_prefix + example.source for example in examples]
    if not sources:
        return []

    encoded = tokenizer(
        sources,
        max_length=config.max_source_tokens,
        truncation=True,
    )["input_ids"]
    return [len(ids) for ids in encoded]


def token_lengths(
    examples: Sequence[Example],
    tokenizer: PreTrainedTokenizerBase,
    config: TokenizerConfig,
) -> tuple[list[int], list[int]]:
    """Measure the untruncated token length of every example.

    The result justifies the truncation lengths: it tells how many documents
    the configured limit actually cuts.

    Args:
        examples: The examples to measure.
        tokenizer: The shared tokeniser.
        config: Provides the source prefix.

    Returns:
        A pair of lists holding the source and target token counts.
    """
    sources = [config.source_prefix + example.source for example in examples]
    targets = [example.target for example in examples]

    source_lengths = [len(ids) for ids in tokenizer(sources, truncation=False)["input_ids"]]
    target_lengths = [len(ids) for ids in tokenizer(targets, truncation=False)["input_ids"]]

    return source_lengths, target_lengths
