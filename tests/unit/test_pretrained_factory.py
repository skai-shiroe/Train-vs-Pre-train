"""Unit tests for the baseline registry.

An experiment file names its baseline as a string. What is checked here is that
the string resolves, that an unknown one fails with a message naming what does
exist, and that the arguments an experiment sets really reach the object.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from transformers import PreTrainedModel, T5ForConditionalGeneration

from src.models.pretrained.base import BaselineConfig, PretrainedSummarizer
from src.models.pretrained.factory import (
    BASELINES,
    available_baselines,
    build_summarizer,
    get_baseline_class,
)
from src.models.pretrained.t5 import T5Summarizer
from tests.conftest import FakeTokenizer

T5Factory = Callable[..., T5ForConditionalGeneration]


@pytest.fixture
def offline_baseline(
    tiny_t5: T5Factory, fake_tokenizer: FakeTokenizer, monkeypatch: pytest.MonkeyPatch
) -> type[PretrainedSummarizer]:
    """Register a baseline that loads nothing over the network."""

    class StubSummarizer(PretrainedSummarizer):
        name = "stub"
        default_hf_id = "stub-tiny"
        checked: list[BaselineConfig] = []

        @classmethod
        def check_config(cls, config: BaselineConfig) -> None:
            cls.checked.append(config)

        @classmethod
        def load_model(cls, config: BaselineConfig) -> PreTrainedModel:
            return tiny_t5()

    monkeypatch.setattr("src.models.pretrained.base.build_tokenizer", lambda hf_id: fake_tokenizer)
    monkeypatch.setitem(BASELINES, StubSummarizer.name, StubSummarizer)
    return StubSummarizer


def test_the_registry_lists_the_baselines_an_experiment_may_name() -> None:
    assert available_baselines() == ("t5",)


def test_a_name_resolves_to_its_class() -> None:
    assert get_baseline_class("t5") is T5Summarizer


def test_an_unknown_name_says_what_does_exist() -> None:
    with pytest.raises(KeyError, match="Unknown baseline 'mbart'. Available: t5"):
        get_baseline_class("mbart")


def test_building_an_unknown_baseline_fails_the_same_way() -> None:
    with pytest.raises(KeyError, match="Unknown baseline"):
        build_summarizer("marian")


def test_a_registered_baseline_is_built_from_its_name(
    offline_baseline: type[PretrainedSummarizer],
) -> None:
    config = BaselineConfig(hf_id="stub-tiny", source_prefix="summarize: ")

    summarizer = build_summarizer("stub", config)

    assert isinstance(summarizer, offline_baseline)
    assert summarizer.config is config
    assert summarizer.fine_tuned is False


def test_the_configuration_is_checked_before_the_weights_are_loaded(
    offline_baseline: type[PretrainedSummarizer],
) -> None:
    config = BaselineConfig(hf_id="stub-tiny")

    build_summarizer("stub", config)

    assert offline_baseline.checked[-1] is config  # type: ignore[attr-defined]


def test_a_baseline_can_be_built_without_a_configuration(
    offline_baseline: type[PretrainedSummarizer],
) -> None:
    summarizer = build_summarizer("stub")

    assert summarizer.config.hf_id == "stub-tiny"


def test_the_fine_tuned_flag_reaches_the_description(
    offline_baseline: type[PretrainedSummarizer],
) -> None:
    # A fine tuned model reported as zero shot would be a fabricated result.
    summarizer = build_summarizer("stub", fine_tuned=True)

    assert summarizer.fine_tuned is True
    assert summarizer.describe()["mode"] == "fine_tuned"
