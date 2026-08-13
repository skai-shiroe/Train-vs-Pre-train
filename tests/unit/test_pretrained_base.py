"""Unit tests for what every pretrained baseline shares.

The configuration, the translation of the decoding settings into ``generate``
arguments and the loss adapter are checked here on a stub model, so nothing is
downloaded and nothing depends on which architecture is behind the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from src.data.config import TokenizerConfig
from src.data.tokenize import IGNORE_INDEX, EncodedBatch
from src.models.generation import GenerationConfig
from src.models.pretrained.base import BaselineConfig, build_batch_loss, generation_kwargs


@dataclass
class StubOutput:
    loss: torch.Tensor
    logits: torch.Tensor


class StubSeq2Seq(nn.Module):
    """Return canned logits and a canned loss, and record what it was fed."""

    def __init__(self, logits: torch.Tensor, loss: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.loss = loss
        self.seen: dict[str, torch.Tensor] = {}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> StubOutput:
        self.seen = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        return StubOutput(loss=self.loss, logits=self.logits)


def make_batch(labels: list[list[int]]) -> EncodedBatch:
    target_ids = torch.tensor([[max(0, token) for token in row] for row in labels])
    return EncodedBatch(
        input_ids=torch.tensor([[2, 3, 4]] * len(labels)),
        attention_mask=torch.ones(len(labels), 3, dtype=torch.long),
        labels=torch.tensor(labels),
        target_ids=target_ids,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_a_configuration_carries_the_identifier_and_the_budgets() -> None:
    config = BaselineConfig(hf_id="t5-small", source_prefix="summarize: ")

    assert config.max_source_tokens == 512
    assert config.max_target_tokens == 64
    assert config.revision is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"hf_id": "   "}, "hf_id must name"),
        ({"max_source_tokens": 0}, "max_source_tokens"),
        ({"max_target_tokens": -1}, "max_target_tokens"),
    ],
)
def test_an_impossible_configuration_is_refused(overrides: dict[str, object], message: str) -> None:
    settings: dict[str, object] = {"hf_id": "t5-small"}
    settings.update(overrides)

    with pytest.raises(ValueError, match=message):
        BaselineConfig(**settings)  # type: ignore[arg-type]


def test_the_configuration_is_derived_from_the_corpus_tokeniser() -> None:
    # The baseline must cut a document at inference exactly where the pipeline
    # cut it during training, so the two settings come from one source.
    tokenizer_config = TokenizerConfig(
        hf_id="t5-small",
        source_prefix="summarize: ",
        max_source_tokens=256,
        max_target_tokens=32,
    )

    config = BaselineConfig.from_tokenizer_config(tokenizer_config)

    assert config.hf_id == "t5-small"
    assert config.source_prefix == "summarize: "
    assert config.max_source_tokens == 256
    assert config.max_target_tokens == 32


def test_the_model_identifier_can_differ_from_the_tokeniser_one() -> None:
    tokenizer_config = TokenizerConfig(
        hf_id="t5-small", source_prefix="summarize: ", max_source_tokens=64, max_target_tokens=16
    )

    config = BaselineConfig.from_tokenizer_config(
        tokenizer_config, hf_id="runs/pretrained_ft_100", revision="abc123"
    )

    assert config.hf_id == "runs/pretrained_ft_100"
    assert config.revision == "abc123"
    assert config.max_source_tokens == 64


def test_the_configuration_renders_as_a_flat_mapping() -> None:
    config = BaselineConfig(hf_id="t5-small", source_prefix="summarize: ")

    rendered = config.to_dict()

    assert rendered["hf_id"] == "t5-small"
    assert set(rendered) == {
        "hf_id",
        "revision",
        "source_prefix",
        "max_source_tokens",
        "max_target_tokens",
    }


# ---------------------------------------------------------------------------
# Decoding arguments
# ---------------------------------------------------------------------------


def test_greedy_decoding_passes_no_beam_only_argument() -> None:
    # generate() warns when it receives an argument its strategy ignores, and a
    # warning nobody reads is how a real misconfiguration goes unnoticed.
    kwargs = generation_kwargs(GenerationConfig(max_new_tokens=32))

    assert kwargs["num_beams"] == 1
    assert "length_penalty" not in kwargs
    assert "early_stopping" not in kwargs


def test_beam_decoding_passes_the_length_penalty() -> None:
    kwargs = generation_kwargs(GenerationConfig(num_beams=4, length_penalty=0.8))

    assert kwargs["num_beams"] == 4
    assert kwargs["length_penalty"] == 0.8
    assert kwargs["early_stopping"] is True


def test_sampling_is_off_by_default() -> None:
    # A sampled summary changes from one evaluation to the next, which would
    # make a reported score irreproducible. Nothing in the experiment chain sets
    # do_sample, so the default is what every measurement runs under.
    kwargs = generation_kwargs(GenerationConfig())

    assert kwargs["do_sample"] is False
    # generate() warns about a knob its strategy ignores, so the truncations
    # must not be sent when there is no draw to narrow.
    assert "temperature" not in kwargs
    assert "top_k" not in kwargs
    assert "top_p" not in kwargs


def test_a_sampled_configuration_carries_its_truncations() -> None:
    kwargs = generation_kwargs(
        GenerationConfig(do_sample=True, temperature=0.7, top_k=40, top_p=0.9)
    )

    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9
    # Sent even when it disables the cut: the default of transformers is fifty,
    # so omitting it would apply a truncation this project never asked for.
    assert kwargs["top_k"] == 40
    assert generation_kwargs(GenerationConfig(do_sample=True, top_k=0))["top_k"] == 0


def test_every_decoding_constraint_reaches_generate() -> None:
    kwargs = generation_kwargs(
        GenerationConfig(max_new_tokens=48, min_new_tokens=5, no_repeat_ngram_size=3)
    )

    assert kwargs["max_new_tokens"] == 48
    assert kwargs["min_new_tokens"] == 5
    assert kwargs["no_repeat_ngram_size"] == 3


# ---------------------------------------------------------------------------
# Loss adapter
# ---------------------------------------------------------------------------


def test_an_impossible_smoothing_is_refused() -> None:
    with pytest.raises(ValueError, match=r"label_smoothing must lie in \[0, 1\)"):
        build_batch_loss(1.0)


def test_without_smoothing_the_model_loss_is_returned_untouched() -> None:
    # Recomputing it would be one useless cross entropy over the whole
    # vocabulary at every step.
    expected = torch.tensor(1.2345)
    model = StubSeq2Seq(logits=torch.zeros(1, 3, 8), loss=expected)

    loss = build_batch_loss()(model, make_batch([[5, 6, IGNORE_INDEX]]))

    assert loss is expected


def test_the_batch_is_forwarded_with_its_mask_and_its_labels() -> None:
    model = StubSeq2Seq(logits=torch.zeros(1, 3, 8), loss=torch.tensor(0.0))
    batch = make_batch([[5, 6, IGNORE_INDEX]])

    build_batch_loss()(model, batch)

    assert torch.equal(model.seen["input_ids"], batch.input_ids)
    assert torch.equal(model.seen["attention_mask"], batch.attention_mask)
    # Labels rather than target_ids: the model shifts them itself to build the
    # decoder input, exactly as the from scratch model does.
    assert torch.equal(model.seen["labels"], batch.labels)


def test_smoothing_replaces_the_loss_the_model_computed() -> None:
    logits = torch.zeros(1, 3, 8)
    logits[0, 0, 5] = 10.0
    logits[0, 1, 6] = 10.0
    model = StubSeq2Seq(logits=logits, loss=torch.tensor(99.0))

    loss = build_batch_loss(0.1)(model, make_batch([[5, 6, IGNORE_INDEX]]))

    assert float(loss) < 1.0


def test_the_smoothed_loss_ignores_the_padded_positions() -> None:
    batch = make_batch([[5, 6, IGNORE_INDEX]])
    logits = torch.zeros(1, 3, 8)
    logits[0, 0, 5] = 10.0
    logits[0, 1, 6] = 10.0

    reference = build_batch_loss(0.1)(StubSeq2Seq(logits, torch.tensor(0.0)), batch)

    # Only the padded position changes. A loss that covered it would move.
    perturbed = logits.clone()
    perturbed[0, 2, :] = torch.arange(8, dtype=torch.float32) * 5.0
    measured = build_batch_loss(0.1)(StubSeq2Seq(perturbed, torch.tensor(0.0)), batch)

    assert float(measured) == pytest.approx(float(reference))
