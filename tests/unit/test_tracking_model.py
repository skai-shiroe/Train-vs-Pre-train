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
    REQUIRED_PACKAGES,
    log_model_safely,
    pip_requirements,
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
# The pins written beside a logged model
# ---------------------------------------------------------------------------


def test_every_installed_package_is_pinned() -> None:
    pins = pip_requirements()

    assert {pin.split("==")[0] for pin in pins} <= set(REQUIRED_PACKAGES)
    assert all("==" in pin for pin in pins)


def test_torch_is_pinned_without_its_local_version_label() -> None:
    # The environment runs 2.13.0+cu130, and that form resolves from the CUDA
    # index and nowhere else. A requirements file no pip install can satisfy
    # helps no one, so the public version is what is published.
    torch_pin = next(pin for pin in pip_requirements() if pin.startswith("torch=="))

    assert "+" not in torch_pin


def test_a_missing_package_is_skipped_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This is the defect the explicit list exists to fix: MLflow's own
    # inference pins torchvision by importing it, and a text only project that
    # never installed it got a logging failure instead of a model.
    import src.tracking.model as module

    def absent(package: str) -> str:
        from importlib.metadata import PackageNotFoundError

        if package == "sentencepiece":
            raise PackageNotFoundError(package)
        return "1.2.3"

    monkeypatch.setattr(module, "REQUIRED_PACKAGES", ("torch", "sentencepiece"))
    monkeypatch.setattr("importlib.metadata.version", absent)

    assert pip_requirements() == ["torch==1.2.3"]


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
