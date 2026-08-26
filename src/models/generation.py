"""Decoding hyperparameters and batching, shared by both sides of the comparison.

The from scratch Transformer decodes with the loops of
:mod:`src.models.scratch.generation`; the two T5 runs decode with the
``generate`` method of the Hugging Face model. Two implementations, one
configuration object.

Sharing it is not a convenience. Section 2.1 compares two models on the same
test set, and a beam width or a length penalty that differed between them would
move the reported score for a reason that has nothing to do with the models. A
single object makes that divergence impossible to express.

The same argument applies to :func:`iter_batches`, which walks a corpus for both
summarisers: the slicing that decides which documents share a padded batch, and
the order they come back in, is written once.

**Sampling is off by default, and a sampled run still names its seed.** Greedy
and beam search return the same summary for the same input, which is what
section 14 requires of a reported score. Sampling trades that away deliberately,
so ``seed`` exists to buy it back: a run that draws its tokens at random is
reproducible as soon as the seed it drew under is known. The evaluation never
sets ``do_sample``, so nothing in the experiment chain moves.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Decoding hyperparameters.

    Attributes:
        max_new_tokens: Hard budget of generated tokens, excluding the start token.
        min_new_tokens: Number of tokens generated before the end of sequence
            token is allowed. Guards against the empty summary a poorly trained
            model tends to produce.
        num_beams: Beam width. A width of one is greedy search.
        length_penalty: Exponent applied to the length when scoring a finished
            beam. Above one favours longer sequences, below one favours shorter.
        no_repeat_ngram_size: Forbid repeating an n-gram of this size. Zero
            disables the constraint.
        do_sample: Draw each token from the distribution instead of taking the
            most likely one. Off by default: an evaluation must be reproducible.
        temperature: Divides the logits before the draw. Below one sharpens the
            distribution towards the greedy choice, above one flattens it.
            Ignored when ``do_sample`` is false.
        top_k: Keep only the ``k`` most likely tokens before the draw. Zero
            disables the cut.
        top_p: Keep the smallest set of tokens whose probabilities sum to
            ``top_p``, the nucleus. One disables the cut.
        seed: Seed the generator draws under. ``None`` leaves the ambient random
            state alone, which is what a deterministic decoding wants; any
            integer makes a sampled run repeatable.
    """

    max_new_tokens: int = 64
    min_new_tokens: int = 0
    num_beams: int = 1
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate the decoding hyperparameters.

        Raises:
            ValueError: If a budget is not positive, if the beam width is below
                one, if the minimum length exceeds the maximum, if a sampling
                knob is outside its domain, or if sampling is asked for
                alongside a beam search.
        """
        if self.max_new_tokens <= 0:
            raise ValueError(
                f"max_new_tokens must be strictly positive, got {self.max_new_tokens}."
            )
        if self.min_new_tokens < 0:
            raise ValueError(f"min_new_tokens must be non negative, got {self.min_new_tokens}.")
        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError("min_new_tokens cannot exceed max_new_tokens.")
        if self.num_beams < 1:
            raise ValueError(f"num_beams must be at least one, got {self.num_beams}.")
        if self.no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be non negative.")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be strictly positive, got {self.temperature}.")
        if self.top_k < 0:
            raise ValueError(f"top_k must be non negative, got {self.top_k}.")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must lie in (0, 1], got {self.top_p}.")
        # Sampling inside a beam search is a third strategy, not the union of
        # two: the beams would have to be scored on draws rather than on
        # probabilities. The from scratch decoder implements greedy and beam
        # search only, and silently dropping one of the two requests here would
        # let a caller believe it got a decoding it did not.
        if self.do_sample and self.num_beams > 1:
            raise ValueError(
                "do_sample cannot be combined with a beam search. "
                f"Set num_beams to one, got {self.num_beams}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Render the configuration as a flat mapping.

        **A deterministic decoding renders without its sampling knobs.** They
        would all hold their defaults, having done nothing, and this mapping is
        what :attr:`src.experiments.record.Record.measurement_key` compares two
        runs on. Emitting five inert keys would make every run recorded before
        sampling existed incomparable with every run recorded after it, and
        ``make ablation`` would refuse a study over a difference that never
        reached a model. A run that did sample keeps them, which is what makes
        it correctly incomparable with one that did not.

        Returns:
            A mapping suitable for MLflow parameter logging.
        """
        rendered = asdict(self)
        if not self.do_sample:
            for knob in ("do_sample", "temperature", "top_k", "top_p", "seed"):
                del rendered[knob]
        return rendered


def apply_seed(config: GenerationConfig) -> None:
    """Seed the ambient generator when the configuration names a seed.

    Both decoders draw from the global torch generator: the from scratch one
    through :func:`torch.multinomial`, the T5 runs inside ``generate``. Seeding
    here rather than in either of them is what makes the same seed mean the
    same thing on every side of the comparison.

    A configuration without a seed is left alone rather than seeded with a
    default. Reseeding on every call would make a sampled endpoint return the
    same summary to every caller, which is the behaviour sampling exists to
    avoid, and touching the global state of a process that did not ask for it
    would silently derail anything else drawing from it.

    Args:
        config: The decoding configuration about to run.
    """
    if config.seed is not None:
        torch.manual_seed(config.seed)


def iter_batches[T](items: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    """Split a sequence into consecutive batches, in order.

    The batch size bounds the memory a long corpus needs; the order guarantees
    that prediction ``i`` still belongs to document ``i`` once the batches are
    concatenated. An evaluation that lost that alignment would score every
    summary against the wrong reference and still return a plausible number.

    The size is validated eagerly, before the first batch is produced, so a
    misconfigured call fails at the call site rather than inside the loop.

    Args:
        items: The sequence to walk.
        batch_size: Number of items per batch.

    Returns:
        An iterator over consecutive slices. An empty sequence yields nothing.

    Raises:
        ValueError: If the batch size is not strictly positive.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be strictly positive, got {batch_size}.")

    return (items[start : start + batch_size] for start in range(0, len(items), batch_size))
