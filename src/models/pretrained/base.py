"""Contract shared by every pretrained baseline.

Section 9 asks for a separate module holding the pretrained side of the
comparison, supporting zero shot use, fine tuning, inference and evaluation.
This module holds everything that does not depend on which architecture sits
behind the baseline:

```text
here        prefix and truncation, batched inference, the loss adapter,
            device handling, persistence, the description logged with a run
subclass    which Hugging Face class to load, and the rules that architecture
            imposes on the configuration
```

Three decisions are worth stating.

**The tokeniser is the corpus tokeniser.**
:func:`src.data.tokenize.build_tokenizer` is cached per identifier, so the
baseline and the data pipeline hold the very same object. Section 2.1 requires
the two models to share the vocabulary; sharing the instance is stronger than
repeating an identifier in two configuration files.

**The decoding configuration is the from scratch one.** Both models read
:class:`src.models.generation.GenerationConfig`, so a beam width cannot
silently differ between the two sides of the comparison.

**Zero shot and fine tuned are the same class.** Nothing in the code changes
between them: a zero shot baseline is one whose weights were never updated. The
distinction is carried by a flag, reported in :meth:`PretrainedSummarizer.describe`,
so a result can never be filed under the wrong mode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Self, cast

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.data.config import TokenizerConfig
from src.data.tokenize import IGNORE_INDEX, EncodedBatch, build_tokenizer
from src.models.generation import GenerationConfig, apply_seed, iter_batches


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Identification and input budget of a pretrained baseline.

    The truncation lengths repeat those of :class:`src.data.config.TokenizerConfig`
    on purpose: inference is fed raw strings, without going through the corpus
    pipeline, and it must still cut documents exactly where training cut them.
    :meth:`from_tokenizer_config` is the way to keep the two in step.

    Attributes:
        hf_id: Hugging Face identifier, or a local directory written by
            :meth:`PretrainedSummarizer.save_pretrained`.
        revision: Commit or tag pinned for the download. ``None`` follows the
            default branch, which section 14 tolerates only outside a reported
            experiment.
        source_prefix: Text prepended to every document. Required by the models
            that select their task from a prefix.
        max_source_tokens: Truncation length of the document.
        max_target_tokens: Truncation length of the summary, and default
            generation budget.
    """

    hf_id: str
    revision: str | None = None
    source_prefix: str = ""
    max_source_tokens: int = 512
    max_target_tokens: int = 64

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If the identifier is blank or a budget is not strictly
                positive.
        """
        if not self.hf_id.strip():
            raise ValueError("hf_id must name a model or a local directory.")
        if self.max_source_tokens <= 0:
            raise ValueError(
                f"max_source_tokens must be strictly positive, got {self.max_source_tokens}."
            )
        if self.max_target_tokens <= 0:
            raise ValueError(
                f"max_target_tokens must be strictly positive, got {self.max_target_tokens}."
            )

    @classmethod
    def from_tokenizer_config(
        cls,
        config: TokenizerConfig,
        *,
        hf_id: str | None = None,
        revision: str | None = None,
    ) -> Self:
        """Derive a baseline configuration from the corpus tokeniser settings.

        Args:
            config: The ``tokenizer`` block of the data pipeline configuration.
            hf_id: Model identifier. Defaults to the tokeniser identifier, which
                is the right answer when the baseline is the model the tokeniser
                came from.
            revision: Commit or tag pinned for the download.

        Returns:
            A configuration whose prefix and truncation match the corpus.
        """
        return cls(
            hf_id=hf_id or config.hf_id,
            revision=revision,
            source_prefix=config.source_prefix,
            max_source_tokens=config.max_source_tokens,
            max_target_tokens=config.max_target_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Render the configuration as a flat mapping.

        Returns:
            A mapping suitable for MLflow parameter logging.
        """
        return asdict(self)


def generation_kwargs(config: GenerationConfig) -> dict[str, Any]:
    """Translate the project decoding configuration into ``generate`` arguments.

    Sampling is passed through rather than assumed either way, but it stays off
    everywhere a score is produced: section 14 requires a reported score to be
    reproducible, and no evaluation configuration of this project sets
    ``do_sample``. The API is the only caller that turns it on.

    **The truncations are only sent when they are in use.** ``generate`` warns
    about a temperature or a nucleus supplied next to ``do_sample`` false,
    because it would ignore them, and a warning on every greedy call is a
    warning nobody reads any more.

    **``top_k`` is always sent when sampling.** The default of transformers is
    fifty, not zero, so omitting it would apply a truncation this project never
    asked for and make ``top_k: 0`` mean the opposite of what it says.

    Args:
        config: Decoding configuration.

    Returns:
        The keyword arguments to hand to ``generate``.
    """
    kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "min_new_tokens": config.min_new_tokens,
        "num_beams": config.num_beams,
        "no_repeat_ngram_size": config.no_repeat_ngram_size,
        "do_sample": config.do_sample,
    }
    if config.do_sample:
        kwargs["temperature"] = config.temperature
        kwargs["top_k"] = config.top_k
        kwargs["top_p"] = config.top_p
    if config.num_beams > 1:
        # Both only mean something to a beam search. Passing them with a width
        # of one makes generate() warn about an unused argument.
        kwargs["length_penalty"] = config.length_penalty
        kwargs["early_stopping"] = True
    return kwargs


def build_batch_loss(
    label_smoothing: float = 0.0,
) -> Callable[[nn.Module, EncodedBatch], torch.Tensor]:
    """Build the loss function of a Hugging Face sequence to sequence model.

    The signature is the one :data:`src.training.trainer.BatchLossFn` declares,
    so fine tuning the baseline runs through the same loop, the same token
    weighted averaging and the same checkpoints as the from scratch model. That
    is what makes the two training curves comparable.

    Teacher forcing is left to the model: given ``labels``, a Hugging Face
    sequence to sequence model shifts them right and starts the decoder with the
    same token the from scratch model uses, so both are fed the same decoder
    input.

    Args:
        label_smoothing: Mass taken from the gold token and spread over the
            vocabulary. Zero returns the loss the model computed itself.

    Returns:
        A function mapping a model and a batch to a scalar loss.

    Raises:
        ValueError: If the smoothing mass is outside ``[0, 1)``.
    """
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(f"label_smoothing must lie in [0, 1), got {label_smoothing}.")

    def batch_loss(model: nn.Module, batch: EncodedBatch) -> torch.Tensor:
        output = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
        )
        if label_smoothing == 0.0:
            return cast(torch.Tensor, output.loss)

        # Smoothing is a training hyperparameter, not a property of the model,
        # so the loss is recomputed here rather than configured upstream.
        logits = cast(torch.Tensor, output.logits)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch.labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            label_smoothing=label_smoothing,
        )

    return batch_loss


class PretrainedSummarizer(ABC):
    """A pretrained sequence to sequence model behind a fixed interface."""

    #: Short name under which :mod:`src.models.pretrained.factory` registers the
    #: baseline, and the value reported as ``baseline`` in the run description.
    name: ClassVar[str] = ""

    #: Identifier used when a configuration is not supplied.
    default_hf_id: ClassVar[str] = ""

    def __init__(
        self,
        config: BaselineConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        fine_tuned: bool = False,
    ) -> None:
        """Wrap an already loaded model and tokeniser.

        Args:
            config: Identification and input budget of the baseline.
            model: The loaded Hugging Face model.
            tokenizer: The shared tokeniser.
            fine_tuned: Whether the weights were updated on the working corpus.
                Reported by :meth:`describe`, never inferred.
        """
        self.config = config
        self._model = model
        self._tokenizer = tokenizer
        self._fine_tuned = fine_tuned

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def load_model(cls, config: BaselineConfig) -> PreTrainedModel:
        """Load the Hugging Face model behind the baseline.

        Args:
            config: Identification of the checkpoint to load.

        Returns:
            The loaded model, on the CPU.
        """

    @classmethod
    @abstractmethod
    def check_config(cls, config: BaselineConfig) -> None:
        """Reject a configuration the architecture cannot honour.

        Every baseline states its own requirements, even when that statement is
        that it has none beyond what :class:`BaselineConfig` already validated.
        :meth:`from_pretrained` calls this before a single weight is
        downloaded, so a misconfigured run fails in a second rather than after
        a few hundred megabytes.

        Args:
            config: The configuration about to be used.
        """

    @classmethod
    def from_pretrained(
        cls, config: BaselineConfig | None = None, *, fine_tuned: bool = False
    ) -> Self:
        """Load the baseline from the hub or from a local directory.

        Args:
            config: Identification and input budget. Defaults to
                :attr:`default_hf_id` with the class defaults.
            fine_tuned: Whether the directory being loaded holds weights that
                were trained on the working corpus. Left to the caller, because
                a directory carries no such marker.

        Returns:
            The baseline, ready for inference or for fine tuning.
        """
        settings = config or BaselineConfig(hf_id=cls.default_hf_id)
        cls.check_config(settings)
        return cls(
            settings,
            cls.load_model(settings),
            build_tokenizer(settings.hf_id),
            fine_tuned=fine_tuned,
        )

    @classmethod
    def from_checkpoint(
        cls, config: BaselineConfig, state_dict: Mapping[str, torch.Tensor]
    ) -> Self:
        """Rebuild a fine tuned baseline from the weights of a training run.

        The checkpoint written by :mod:`src.training.checkpoint` holds the
        weights under the ``model`` key. Reading the file is left to the caller
        so that this package never depends on the training package::

            payload = load_checkpoint(run_dir / "best.pt")
            summarizer = T5Summarizer.from_checkpoint(config, payload["model"])

        Args:
            config: The configuration the run was started from.
            state_dict: The weights stored by the checkpoint.

        Returns:
            The baseline, marked as fine tuned.
        """
        summarizer = cls.from_pretrained(config, fine_tuned=True)
        summarizer.model.load_state_dict(state_dict)
        return summarizer

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def model(self) -> PreTrainedModel:
        """Return the wrapped model.

        The trainer needs the module itself, not the wrapper.

        Returns:
            The Hugging Face model.
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
    def fine_tuned(self) -> bool:
        """Return whether the weights were trained on the working corpus.

        Returns:
            ``True`` for a fine tuned baseline, ``False`` for a zero shot one.
        """
        return self._fine_tuned

    @property
    def device(self) -> torch.device:
        """Return the device the weights currently live on.

        Returns:
            The device of the first parameter.
        """
        return next(self._model.parameters()).device

    @property
    def num_parameters(self) -> int:
        """Return the number of trainable parameters.

        Returns:
            The parameter count, counting a tied matrix once.
        """
        return int(self._model.num_parameters(only_trainable=True))

    def to(self, device: torch.device) -> Self:
        """Move the weights to a device.

        Args:
            device: Target device, typically from
                :func:`src.utils.device.resolve_device`.

        Returns:
            The baseline itself, so the call can be chained.
        """
        # Annotating the torch side of the model on the way in: transformers
        # wraps to() and eval() in a decorator whose signature mypy cannot
        # follow, while nn.Module types both precisely.
        module: nn.Module = self._model
        module.to(device)
        return self

    def describe(self) -> dict[str, str]:
        """Return the description logged next to the metrics of a run.

        Section 19 requires a reported score to carry what produced it. The mode
        matters as much as the identifier: the same weights measured zero shot
        and fine tuned are two different results.

        Returns:
            A flat mapping of strings, always JSON serialisable.
        """
        return {
            "baseline": self.name,
            "hf_id": self.config.hf_id,
            "revision": self.config.revision or "default",
            "mode": "fine_tuned" if self._fine_tuned else "zero_shot",
            "parameters": str(self.num_parameters),
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def batch_loss(
        self, label_smoothing: float = 0.0
    ) -> Callable[[nn.Module, EncodedBatch], torch.Tensor]:
        """Return the loss function to hand to the trainer.

        Args:
            label_smoothing: Mass taken from the gold token, usually
                ``TrainingConfig.label_smoothing``.

        Returns:
            A function the trainer calls once per batch.
        """
        return build_batch_loss(label_smoothing)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def encode(self, documents: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise raw documents the way the corpus pipeline does.

        Args:
            documents: The documents to summarise, without the prefix.

        Returns:
            A pair ``(input_ids, attention_mask)``, on the model device.

        Raises:
            ValueError: If no document is supplied. An empty batch has no width
                to pad to, and the tokeniser would fail further down.
        """
        if not documents:
            raise ValueError("Cannot encode an empty batch.")

        encoded = self._tokenizer(
            [self.config.source_prefix + document for document in documents],
            max_length=self.config.max_source_tokens,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
        device = self.device
        input_ids = cast(torch.Tensor, encoded["input_ids"]).to(device)
        attention_mask = cast(torch.Tensor, encoded["attention_mask"]).to(device)
        return input_ids, attention_mask

    @torch.no_grad()
    def generate_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        config: GenerationConfig | None = None,
    ) -> torch.Tensor:
        """Generate one summary per input, as token identifiers.

        Args:
            input_ids: Encoded documents, of shape ``(batch, src_len)``.
            attention_mask: Ones on real tokens, zeros on padding.
            config: Decoding configuration. Defaults to greedy search over the
                target budget of the baseline.

        Returns:
            A tensor of shape ``(batch, generated_len)``, start token excluded.
        """
        settings = config or GenerationConfig(max_new_tokens=self.config.max_target_tokens)
        module: nn.Module = self._model
        module.eval()

        # Seeded here rather than inside generate(): transformers draws from the
        # ambient torch generator, the same one the from scratch decoder samples
        # from, so one seed covers both sides of the comparison.
        apply_seed(settings)

        # generate() reaches mypy as an attribute rather than as a method: the
        # generation mixin of transformers is not annotated well enough to be
        # followed. The cast states the contract the call actually honours: a
        # bare tensor comes back as long as structured outputs stay off, which
        # sampling does not change.
        generate = cast(Callable[..., torch.Tensor], self._model.generate)
        sequences = generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs(settings),
        )
        # An encoder decoder returns the decoder start token at position zero.
        # The from scratch generator does not, so it is dropped here and the two
        # outputs stay directly comparable.
        return sequences[:, 1:]

    def summarize(
        self,
        documents: Sequence[str],
        config: GenerationConfig | None = None,
        *,
        batch_size: int = 8,
    ) -> list[str]:
        """Summarise documents, in batches.

        This is the whole inference path, and also the evaluation path: the
        predictions a metric scores are the strings returned here.

        Args:
            documents: The documents to summarise, without the prefix.
            config: Decoding configuration.
            batch_size: Number of documents encoded at once. Bounds the memory
                a long list needs, which a single batch would not.

        Returns:
            One summary per document, in the order they were given. An empty
            input returns an empty list.

        Raises:
            ValueError: If the batch size is not strictly positive.
        """
        summaries: list[str] = []
        for batch in iter_batches(documents, batch_size):
            input_ids, attention_mask = self.encode(batch)
            generated = self.generate_ids(input_ids, attention_mask, config)
            summaries.extend(
                text.strip()
                for text in self._tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        return summaries

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_pretrained(self, directory: Path) -> Path:
        """Write the weights and the tokeniser in the Hugging Face layout.

        This is the Hugging Face layout, the one ``from_pretrained`` reads. The
        tokeniser is written next to the weights so that the directory reloads
        on a machine that never saw the hub::

            summarizer.save_pretrained(path)
            reloaded = T5Summarizer.from_pretrained(
                BaselineConfig(hf_id=str(path), source_prefix="summarize: "),
                fine_tuned=True,
            )

        Args:
            directory: Destination directory. Created when it does not exist.

        Returns:
            The directory written.
        """
        directory.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(directory)
        self._tokenizer.save_pretrained(directory)
        return directory
