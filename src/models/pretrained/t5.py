"""The T5 baseline, zero shot then fine tuned.

``t5-small`` is the pretrained side of the comparison, locked by section 2.1. It
is measured twice on the same test set: once as it comes off the hub, once after
fine tuning on each ablation proportion of the working corpus. The gap between
the two says how much of its score T5 owes to its pretraining rather than to the
20 000 examples the from scratch model gets.

**The prefix is not optional.** T5 was pretrained on a mixture of supervised
tasks, each selected by a textual prefix. Without ``summarize: `` in front of
the document, the model is not asked for a summary at all, and a zero shot
measurement taken that way reports a number that says nothing about
summarisation. :meth:`T5Summarizer.check_config` therefore refuses a blank
prefix rather than letting the run produce a quietly meaningless score.

**The decoder starts on the padding token.** The T5 tokeniser has no beginning
of sequence token, so T5 uses the padding identifier to start its decoder. The
from scratch Transformer copies that choice, which is why the same tokeniser
serves both models without any adaptation.
"""

from __future__ import annotations

from typing import cast

from transformers import PreTrainedModel, T5ForConditionalGeneration

from src.models.pretrained.base import BaselineConfig, PretrainedSummarizer


class T5Summarizer(PretrainedSummarizer):
    """The ``t5-small`` baseline, behind the shared summariser interface."""

    name = "t5"
    default_hf_id = "t5-small"

    @classmethod
    def check_config(cls, config: BaselineConfig) -> None:
        """Require the task prefix T5 selects its behaviour from.

        Args:
            config: The configuration about to be used.

        Raises:
            ValueError: If the source prefix is blank.
        """
        if not config.source_prefix.strip():
            raise ValueError(
                "T5 selects its task from a textual prefix. Set source_prefix to "
                "'summarize: ': without it the model is not asked to summarise, "
                "and the zero shot score measures nothing."
            )

    @classmethod
    def load_model(cls, config: BaselineConfig) -> PreTrainedModel:
        """Load the conditional generation head of T5.

        Args:
            config: Identification of the checkpoint to load.

        Returns:
            The loaded model, on the CPU.
        """
        # from_pretrained is not annotated upstream, so its return type reaches
        # us as Any. The cast restores the contract for the callers.
        return cast(
            PreTrainedModel,
            T5ForConditionalGeneration.from_pretrained(config.hf_id, revision=config.revision),
        )
