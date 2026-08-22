"""Unit tests for putting the measured weights into the store.

Nothing here reaches MLflow. What is pinned is the part that decides: which
runs are allowed into the registry, under what name, which flavour a model
belongs to, and that a store failure is reported rather than raised.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.tracking import model as tracking_model
from src.tracking.model import (
    REGISTRY_PREFIX,
    log_model_safely,
    registered_name,
    should_log,
)

pytestmark = pytest.mark.unit


class Toy(torch.nn.Module):
    """A network that is not a Hugging Face model."""

    def __init__(self) -> None:
        """Build the smallest module that has parameters."""
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the linear projection of the inputs.

        Args:
            inputs: Any batch of two dimensional vectors.

        Returns:
            The projection.
        """
        return self.fc(inputs)


class ToySummarizer:
    """The two attributes :mod:`src.tracking.model` needs."""

    def __init__(self, network: Any) -> None:
        """Wrap a network.

        Args:
            network: The model the summariser exposes.
        """
        self.model = network
        self.tokenizer = object()


# ---------------------------------------------------------------------------
# Which runs, under which name
# ---------------------------------------------------------------------------


def test_the_registry_name_is_prefixed() -> None:
    # Without the prefix the registry of a shared database mixes this
    # project's models with everybody else's.
    assert registered_name("scratch_100") == f"{REGISTRY_PREFIX}-scratch_100"


@pytest.mark.parametrize("status", ["PARTIAL", "FAILED", "NOT_RUN", "MOCK", ""])
def test_only_a_complete_run_is_registered(status: str) -> None:
    assert should_log(status) is False


def test_a_complete_run_is_registered() -> None:
    assert should_log("OK") is True


# ---------------------------------------------------------------------------
# Which flavour
# ---------------------------------------------------------------------------


def test_a_plain_module_is_not_the_pretrained_flavour() -> None:
    assert tracking_model._is_transformers_model(Toy()) is False


def test_a_pretrained_model_is_the_pretrained_flavour() -> None:
    from transformers import AutoConfig, AutoModelForSeq2SeqLM

    # Built from a configuration rather than downloaded: the test pins the
    # dispatch, and pulling t5-small over the network to do it would make a
    # unit test depend on the hub.
    config = AutoConfig.from_pretrained(
        "t5-small",
        d_model=8,
        d_ff=16,
        d_kv=4,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        vocab_size=32,
    )
    assert tracking_model._is_transformers_model(AutoModelForSeq2SeqLM.from_config(config)) is True


# ---------------------------------------------------------------------------
# A store failure never fails a run
# ---------------------------------------------------------------------------


def test_a_failure_is_reported_and_swallowed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def explode(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("magasin injoignable")

    monkeypatch.setattr(tracking_model, "log_model", explode)

    assert log_model_safely(ToySummarizer(Toy()), experiment="scratch_100") is None

    error = capsys.readouterr().err
    assert "scratch_100" in error
    assert "magasin injoignable" in error


def test_the_name_is_returned_when_the_store_accepts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tracking_model,
        "log_model",
        lambda _summarizer, *, experiment, register=True: registered_name(experiment),
    )

    assert log_model_safely(ToySummarizer(Toy()), experiment="scratch_50") == "syntra-scratch_50"
