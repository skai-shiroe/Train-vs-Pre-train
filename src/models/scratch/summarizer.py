"""The from scratch Transformer behind the shared summariser interface.

:mod:`src.models.scratch.generation` returns token identifiers. The pretrained
baseline returns strings. Until the two agree, the evaluator would need a branch
per model, and the branch is exactly where a comparison stops being fair. This
adapter closes the gap: raw documents in, summaries out, the same signature the
baseline exposes.

**The prefix is applied here too.** ``summarize: `` means nothing to a model
that learned its vocabulary from the corpus, but the corpus pipeline prepended
it to every training document. Dropping it at inference would feed the model a
distribution it never saw, and the drop in score would be blamed on the
architecture.

**The identifiers are checked against the tokeniser.** A padding identifier that
disagreed with the tokeniser would leave the padding attended to, and an end of
sequence identifier that disagreed would let generation run to its budget on
every document. Neither raises: both quietly lower the score. They are refused
when the adapter is built.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Self, cast

import torch
from transformers import PreTrainedTokenizerBase

from src.data.config import TokenizerConfig
from src.models.generation import GenerationConfig, iter_batches
from src.models.scratch.generation import generate
from src.models.scratch.transformer import ScratchTransformer


class ScratchSummarizer:
    """The hand written Transformer, exposed like the pretrained baseline."""

    #: Value reported as ``model`` in the run description.
    name: ClassVar[str] = "scratch"

    def __init__(
        self,
        model: ScratchTransformer,
        tokenizer: PreTrainedTokenizerBase,
        config: TokenizerConfig,
        *,
        trained: bool = False,
    ) -> None:
        """Wrap a model and the tokeniser its corpus was encoded with.

        Args:
            model: The Transformer, trained or not.
            tokenizer: The shared tokeniser.
            config: The ``tokenizer`` block of the data pipeline configuration,
                which carries the prefix and the truncation lengths the model
                was trained under.
            trained: Whether the weights went through a training run. Reported
                by :meth:`describe`, never inferred: a randomly initialised
                model produces summaries too, and filing its score as the score
                of the from scratch model would be a fabricated result.

        Raises:
            ValueError: If the padding or end of sequence identifiers disagree
                with the tokeniser, or if the truncation length exceeds what
                the positional encoding supports.
        """
        if model.config.pad_token_id != tokenizer.pad_token_id:
            raise ValueError(
                f"The model pads with {model.config.pad_token_id} and the tokeniser with "
                f"{tokenizer.pad_token_id}. Padding would be attended to instead of masked."
            )
        if model.config.eos_token_id != tokenizer.eos_token_id:
            raise ValueError(
                f"The model ends sequences with {model.config.eos_token_id} and the tokeniser "
                f"with {tokenizer.eos_token_id}. Generation would never stop on its own."
            )
        if config.max_source_tokens > model.config.max_position:
            raise ValueError(
                f"The corpus truncates documents at {config.max_source_tokens} tokens but the "
                f"positional encoding stops at {model.config.max_position}."
            )

        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        self._trained = trained

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def model(self) -> ScratchTransformer:
        """Return the wrapped model.

        Returns:
            The Transformer itself, for the callers that need the module.
        """
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Return the shared tokeniser.

        Returns:
            The tokeniser the corpus was encoded with.
        """
        return self._tokenizer

    @property
    def trained(self) -> bool:
        """Return whether the weights went through a training run.

        Returns:
            ``True`` for a trained model, ``False`` for a freshly initialised one.
        """
        return self._trained

    @property
    def device(self) -> torch.device:
        """Return the device the weights currently live on.

        Returns:
            The device of the first parameter.
        """
        return next(self._model.parameters()).device

    def to(self, device: torch.device) -> Self:
        """Move the weights to a device.

        Args:
            device: Target device, typically from
                :func:`src.utils.device.resolve_device`.

        Returns:
            The summariser itself, so the call can be chained.
        """
        self._model.to(device)
        return self

    def describe(self) -> dict[str, str]:
        """Return the description logged next to the metrics of a run.

        Returns:
            A flat mapping of strings, always JSON serialisable.
        """
        config = self._model.config
        return {
            "model": self.name,
            "mode": "trained" if self._trained else "untrained",
            "parameters": str(self._model.num_parameters),
            "d_model": str(config.d_model),
            "num_heads": str(config.num_heads),
            "encoder_layers": str(config.num_encoder_layers),
            "decoder_layers": str(config.num_decoder_layers),
            "tokenizer": self._config.hf_id,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def encode(self, documents: Sequence[str]) -> torch.Tensor:
        """Tokenise raw documents the way the corpus pipeline does.

        Only the identifiers cross: the model derives its padding mask from the
        padding identifier, so an attention mask would be a second source of
        truth for the same thing.

        Args:
            documents: The documents to summarise, without the prefix.

        Returns:
            Source identifiers of shape ``(batch, src_len)``, on the model device.

        Raises:
            ValueError: If no document is supplied. An empty batch has no width
                to pad to.
        """
        if not documents:
            raise ValueError("Cannot encode an empty batch.")

        encoded = self._tokenizer(
            [self._config.source_prefix + document for document in documents],
            max_length=self._config.max_source_tokens,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
        return cast(torch.Tensor, encoded["input_ids"]).to(self.device)

    def summarize(
        self,
        documents: Sequence[str],
        config: GenerationConfig | None = None,
        *,
        batch_size: int = 8,
    ) -> list[str]:
        """Summarise documents, in batches.

        Args:
            documents: The documents to summarise, without the prefix.
            config: Decoding configuration. Defaults to greedy search over the
                target budget of the corpus, which is the budget the baseline
                defaults to as well.
            batch_size: Number of documents encoded at once.

        Returns:
            One summary per document, in the order they were given. An empty
            input returns an empty list.

        Raises:
            ValueError: If the batch size is not strictly positive.
        """
        settings = config or GenerationConfig(max_new_tokens=self._config.max_target_tokens)

        summaries: list[str] = []
        for batch in iter_batches(documents, batch_size):
            generated = generate(self._model, self.encode(batch), settings)
            summaries.extend(
                text.strip()
                for text in self._tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        return summaries
