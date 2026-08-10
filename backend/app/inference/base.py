"""What the API holds once a version is loaded.

Section 27 asks the backend to load its models through the registry of section
20 rather than from a path. This module states what the rest of the application
sees of a loaded model, and nothing more: an endpoint never touches a torch
module, and a service never learns which of the two architectures answered.

**The protocol belongs to the summarisers, not to the API.**
:class:`src.models.scratch.summarizer.ScratchSummarizer` and
:class:`src.models.pretrained.base.PretrainedSummarizer` already expose the same
``summarize``, because section 2.1 requires the evaluation to call both the same
way. Restating that surface here, rather than writing an adapter of our own,
is what makes the API serve exactly what the evaluation measured.

**A loaded model carries the version document that describes it.** A summary is
an answer to the question of which weights produced it, and a network in memory
cannot answer that. Binding the two together is what lets ``/predict`` report
the version it used and ``/models`` report the score that justified publishing
it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from backend.app.registry.models import ModelVersion
from src.models.generation import GenerationConfig


class Summarizer(Protocol):
    """The inference surface shared by both sides of the comparison."""

    def summarize(
        self,
        documents: Sequence[str],
        config: GenerationConfig | None = None,
        *,
        batch_size: int = 8,
    ) -> list[str]:
        """Summarise documents, in batches.

        Args:
            documents: The documents to summarise, without any prefix. The
                prefix belongs to the summariser, which applies the one its
                corpus was encoded with.
            config: Decoding configuration. ``None`` leaves the summariser on
                the budget of its own tokeniser configuration.
            batch_size: Documents encoded at once.

        Returns:
            One summary per document, in the order they were given.
        """
        ...  # pragma: no cover - a protocol declares, it does not run

    def describe(self) -> dict[str, str]:
        """Return what the model reports about itself.

        Returns:
            A flat mapping of strings, always JSON serialisable.
        """
        ...  # pragma: no cover - a protocol declares, it does not run


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """One published version, loaded and ready to answer.

    Attributes:
        version: The document the registry resolved, kept so that every answer
            can name the weights that produced it.
        summarizer: The loaded network, behind the shared protocol.
        device: Where those weights live.
        token_budget: Longest summary these weights were trained to produce,
            read from the tokeniser block published with them. A request asking
            for more is capped to it rather than refused: past that length a
            model is generating outside the distribution it saw, and a from
            scratch one runs out of positional encoding and raises.
    """

    version: ModelVersion
    summarizer: Summarizer
    device: torch.device
    token_budget: int

    @property
    def label(self) -> str:
        """Return the identifier used in responses and in logs.

        Returns:
            The ``name:version`` pair of the loaded version.
        """
        return self.version.label

    def describe(self) -> dict[str, str]:
        """Return what the loaded model reports about itself.

        Returns:
            The description of the summariser, with the device it runs on.
        """
        return {**self.summarizer.describe(), "device": self.device.type}

    def summarize(self, text: str, config: GenerationConfig | None = None) -> str:
        """Summarise one document.

        The API answers one request at a time, so the batch dimension of the
        evaluation path is collapsed here rather than at every call site.

        Args:
            text: The document to summarise.
            config: Decoding configuration.

        Returns:
            The generated summary, possibly empty. An empty summary is a
            measurement of the model, not an error: section 10 counts them.
        """
        return self.summarizer.summarize([text], config, batch_size=1)[0]
