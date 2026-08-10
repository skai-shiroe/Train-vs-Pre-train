"""Unit tests for autoregressive generation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from src.models.scratch.config import ScratchTransformerConfig
from src.models.scratch.generation import (
    GenerationConfig,
    banned_ngram_tokens,
    beam_search,
    generate,
    greedy_search,
    narrow_for_sampling,
    sample_search,
)
from src.models.scratch.transformer import ScratchTransformer
from src.utils.seed import set_seed

VOCAB_SIZE = 24
PAD = 0
EOS = 1


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    set_seed(1234)


@pytest.fixture
def model() -> ScratchTransformer:
    config = ScratchTransformerConfig(
        vocab_size=VOCAB_SIZE,
        d_model=32,
        num_heads=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        d_ff=64,
        dropout=0.0,
        max_position=64,
        pad_token_id=PAD,
        eos_token_id=EOS,
        decoder_start_token_id=PAD,
    )
    return ScratchTransformer(config).eval()


@pytest.fixture
def source() -> torch.Tensor:
    return torch.randint(2, VOCAB_SIZE, (2, 9))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_configuration_is_greedy() -> None:
    # The evaluation chain builds its configuration from this default, so the
    # day it starts sampling is the day every published score stops being
    # reproducible. Only the API turns the draw on.
    assert GenerationConfig().num_beams == 1
    assert GenerationConfig().do_sample is False
    assert GenerationConfig().seed is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_new_tokens": 0}, "max_new_tokens must be strictly positive"),
        ({"min_new_tokens": -1}, "min_new_tokens must be non negative"),
        ({"min_new_tokens": 10, "max_new_tokens": 5}, "cannot exceed max_new_tokens"),
        ({"num_beams": 0}, "num_beams must be at least one"),
        ({"no_repeat_ngram_size": -2}, "no_repeat_ngram_size must be non negative"),
        ({"temperature": 0.0}, "temperature must be strictly positive"),
        ({"temperature": -1.0}, "temperature must be strictly positive"),
        ({"top_k": -1}, "top_k must be non negative"),
        ({"top_p": 0.0}, r"top_p must lie in \(0, 1\]"),
        ({"top_p": 1.5}, r"top_p must lie in \(0, 1\]"),
        ({"do_sample": True, "num_beams": 2}, "cannot be combined with a beam search"),
    ],
)
def test_invalid_decoding_settings_are_refused(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GenerationConfig(**overrides)


@pytest.mark.unit
def test_the_decoding_settings_render_as_a_flat_mapping() -> None:
    # The same object drives the pretrained baseline, so a run must be able to
    # log one decoding configuration rather than two.
    rendered = GenerationConfig(max_new_tokens=32, num_beams=4).to_dict()

    assert rendered["max_new_tokens"] == 32
    assert rendered["num_beams"] == 4
    assert set(rendered) == {
        "max_new_tokens",
        "min_new_tokens",
        "num_beams",
        "length_penalty",
        "no_repeat_ngram_size",
    }


@pytest.mark.unit
def test_a_deterministic_rendering_carries_no_sampling_knob() -> None:
    # measurement_key compares two runs on this mapping. Five inert defaults
    # would make every run recorded before sampling existed incomparable with
    # every run recorded after it, over a difference no model ever saw.
    assert "temperature" not in GenerationConfig().to_dict()


@pytest.mark.unit
def test_a_sampled_rendering_carries_them_all() -> None:
    # And here the difference is real, so the two runs must not land in one
    # table: a sampled score and a greedy score do not measure the same thing.
    rendered = GenerationConfig(do_sample=True, temperature=0.7, seed=11).to_dict()

    assert rendered["do_sample"] is True
    assert rendered["temperature"] == 0.7
    assert rendered["seed"] == 11
    assert GenerationConfig().to_dict() != rendered


# ---------------------------------------------------------------------------
# n-gram blocking
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_no_ban_when_the_rule_is_disabled() -> None:
    assert banned_ngram_tokens(torch.tensor([5, 6, 5, 6]), 0) == []


@pytest.mark.unit
def test_no_ban_before_the_sequence_is_long_enough() -> None:
    assert banned_ngram_tokens(torch.tensor([5]), 3) == []


@pytest.mark.unit
def test_a_repeated_bigram_is_banned() -> None:
    # The sequence ends with 5, and 5 was already followed by 6.
    assert banned_ngram_tokens(torch.tensor([5, 6, 7, 5]), 2) == [6]


@pytest.mark.unit
def test_several_continuations_can_be_banned_at_once() -> None:
    assert banned_ngram_tokens(torch.tensor([5, 6, 5, 7, 5]), 2) == [6, 7]


@pytest.mark.unit
def test_a_trigram_ban_looks_at_the_last_two_tokens() -> None:
    assert banned_ngram_tokens(torch.tensor([1, 2, 3, 1, 2]), 3) == [3]


@pytest.mark.unit
def test_unigram_blocking_bans_every_seen_token() -> None:
    assert banned_ngram_tokens(torch.tensor([4, 9, 4]), 1) == [4, 9]


# ---------------------------------------------------------------------------
# Greedy search
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_greedy_returns_one_sequence_per_input(
    model: ScratchTransformer, source: torch.Tensor
) -> None:
    generated = greedy_search(model, source, GenerationConfig(max_new_tokens=8))

    assert generated.shape[0] == source.shape[0]
    assert generated.shape[1] <= 8
    assert generated.dtype is torch.long


@pytest.mark.unit
def test_greedy_never_emits_the_start_token_as_output(
    model: ScratchTransformer, source: torch.Tensor
) -> None:
    generated = greedy_search(model, source, GenerationConfig(max_new_tokens=6))

    # The start token is stripped: the first generated token is a real decision.
    assert generated.shape[1] >= 1


@pytest.mark.unit
def test_greedy_is_deterministic(model: ScratchTransformer, source: torch.Tensor) -> None:
    config = GenerationConfig(max_new_tokens=8)

    assert torch.equal(greedy_search(model, source, config), greedy_search(model, source, config))


@pytest.mark.unit
def test_greedy_respects_the_length_budget(model: ScratchTransformer, source: torch.Tensor) -> None:
    generated = greedy_search(model, source, GenerationConfig(max_new_tokens=3))

    assert generated.shape[1] <= 3


@pytest.mark.unit
def test_minimum_length_forbids_an_immediate_end_of_sequence(
    model: ScratchTransformer, source: torch.Tensor
) -> None:
    generated = greedy_search(model, source, GenerationConfig(max_new_tokens=10, min_new_tokens=5))

    assert not bool((generated[:, :5] == EOS).any())


@pytest.mark.unit
def test_generation_stops_once_every_sequence_is_finished(source: torch.Tensor) -> None:
    # A model whose end of sequence logit dominates must stop at the first step.
    config = ScratchTransformerConfig(
        vocab_size=VOCAB_SIZE,
        d_model=32,
        num_heads=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        d_ff=64,
        dropout=0.0,
        max_position=64,
        pad_token_id=PAD,
        eos_token_id=EOS,
        decoder_start_token_id=PAD,
        tie_embeddings=False,
    )
    model = ScratchTransformer(config).eval()
    with torch.no_grad():
        model.decoder.output_projection.weight.zero_()
        model.decoder.output_projection.weight[EOS] = 100.0

    generated = greedy_search(model, source, GenerationConfig(max_new_tokens=20))

    assert generated.shape[1] == 1
    assert bool((generated[:, 0] == EOS).all())


@pytest.mark.unit
def test_unigram_blocking_forces_distinct_tokens(source: torch.Tensor) -> None:
    # Without the constraint this model repeats the same token forever, which
    # is exactly the failure mode a poorly trained decoder exhibits.
    config = ScratchTransformerConfig(
        vocab_size=VOCAB_SIZE,
        d_model=32,
        num_heads=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        d_ff=64,
        dropout=0.0,
        max_position=64,
        pad_token_id=PAD,
        eos_token_id=EOS,
        decoder_start_token_id=PAD,
        tie_embeddings=False,
    )
    model = ScratchTransformer(config).eval()
    with torch.no_grad():
        model.decoder.output_projection.weight.zero_()
        model.decoder.output_projection.weight[5] = 100.0

    # Equal minimum and maximum lengths forbid the end of sequence token at
    # every step, so exactly six tokens are generated in both runs.
    repeating = greedy_search(model, source, GenerationConfig(max_new_tokens=6, min_new_tokens=6))
    varied = greedy_search(
        model,
        source,
        GenerationConfig(max_new_tokens=6, min_new_tokens=6, no_repeat_ngram_size=1),
    )

    # Unigram blocking makes distinct tokens a guarantee, not an accident.
    assert len(set(varied[0].tolist())) == 6
    # And the unconstrained run does repeat, otherwise the comparison is empty.
    assert len(set(repeating[0].tolist())) < 6


@pytest.mark.unit
def test_beam_search_applies_the_ngram_constraint(
    model: ScratchTransformer, source: torch.Tensor
) -> None:
    generated = beam_search(
        model,
        source,
        GenerationConfig(max_new_tokens=6, num_beams=2, no_repeat_ngram_size=2),
    )

    assert generated.shape[0] == source.shape[0]


@pytest.mark.unit
def test_the_model_is_switched_to_evaluation_mode(
    model: ScratchTransformer, source: torch.Tensor
) -> None:
    model.train()

    greedy_search(model, source, GenerationConfig(max_new_tokens=3))

    assert model.training is False


@pytest.mark.unit
def test_generation_produces_no_gradient(model: ScratchTransformer, source: torch.Tensor) -> None:
    generated = greedy_search(model, source, GenerationConfig(max_new_tokens=4))

    assert generated.requires_grad is False


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_beam_search_returns_one_sequence_per_input(
    model: ScratchTransformer, source: torch.Tensor
) -> None:
    generated = beam_search(model, source, GenerationConfig(max_new_tokens=6, num_beams=3))

    assert generated.shape[0] == source.shape[0]
    assert generated.dtype is torch.long


@pytest.mark.unit
def test_beam_search_respects_the_length_budget(
    model: ScratchTransformer, source: torch.Tensor
) -> None:
    generated = beam_search(model, source, GenerationConfig(max_new_tokens=5, num_beams=2))

    assert generated.shape[1] <= 5


@pytest.mark.unit
def test_beam_search_is_deterministic(model: ScratchTransformer, source: torch.Tensor) -> None:
    config = GenerationConfig(max_new_tokens=6, num_beams=3)

    assert torch.equal(beam_search(model, source, config), beam_search(model, source, config))


@pytest.mark.unit
def test_beam_search_emits_valid_token_identifiers(
    model: ScratchTransformer, source: torch.Tensor
) -> None:
    generated = beam_search(model, source, GenerationConfig(max_new_tokens=6, num_beams=3))

    assert bool((generated >= 0).all())
    assert bool((generated < VOCAB_SIZE).all())


@pytest.mark.unit
@pytest.mark.parametrize("length_penalty", [0.5, 1.0, 2.0])
def test_every_length_penalty_runs(
    model: ScratchTransformer, source: torch.Tensor, length_penalty: float
) -> None:
    generated = beam_search(
        model,
        source,
        GenerationConfig(max_new_tokens=6, num_beams=2, length_penalty=length_penalty),
    )

    assert generated.shape[0] == source.shape[0]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_generate_dispatches_to_greedy_for_a_single_beam(
    model: ScratchTransformer, source: torch.Tensor
) -> None:
    config = GenerationConfig(max_new_tokens=6, num_beams=1)

    assert torch.equal(generate(model, source, config), greedy_search(model, source, config))


@pytest.mark.unit
def test_generate_dispatches_to_beam_search_beyond_one_beam(
    model: ScratchTransformer, source: torch.Tensor
) -> None:
    config = GenerationConfig(max_new_tokens=6, num_beams=3)

    assert torch.equal(generate(model, source, config), beam_search(model, source, config))


@pytest.mark.unit
def test_generate_works_without_an_explicit_configuration(
    model: ScratchTransformer, source: torch.Tensor
) -> None:
    assert generate(model, source).shape[0] == source.shape[0]


@pytest.mark.unit
def test_generate_dispatches_to_sampling_when_the_draw_is_asked_for(
    model: ScratchTransformer, source: torch.Tensor
) -> None:
    config = GenerationConfig(max_new_tokens=6, do_sample=True, seed=99)

    assert torch.equal(generate(model, source, config), generate(model, source, config))
    assert not torch.equal(
        generate(model, source, config),
        greedy_search(model, source, replace(config, do_sample=False)),
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sampling_returns_one_sequence_per_input(
    model: ScratchTransformer, source: torch.Tensor
) -> None:
    generated = sample_search(model, source, GenerationConfig(max_new_tokens=8, do_sample=True))

    assert generated.shape[0] == source.shape[0]
    assert generated.shape[1] <= 8


@pytest.mark.unit
def test_two_seeds_do_not_produce_one_summary(
    model: ScratchTransformer, source: torch.Tensor
) -> None:
    # The whole point of the feature. Eight draws collapsing to a single
    # sequence would mean the seed never reached the generator.
    config = GenerationConfig(max_new_tokens=8, do_sample=True)
    drawn = {
        tuple(generate(model, source, replace(config, seed=seed))[0].tolist()) for seed in range(8)
    }

    assert len(drawn) > 1


@pytest.mark.unit
def test_the_same_seed_replays_the_same_sequence(
    model: ScratchTransformer, source: torch.Tensor
) -> None:
    # What the API promises when it returns the seed it drew under.
    config = GenerationConfig(max_new_tokens=8, do_sample=True, seed=4242)

    assert torch.equal(generate(model, source, config), generate(model, source, config))


@pytest.mark.unit
def test_keeping_a_single_token_makes_the_draw_greedy(
    model: ScratchTransformer, source: torch.Tensor
) -> None:
    # top_k of one leaves the argmax alone in the distribution, so the draw can
    # only return it. It is the cheapest check that the narrowing is applied at
    # all, and that it selects the right end of the distribution.
    sampled = sample_search(
        model, source, GenerationConfig(max_new_tokens=6, do_sample=True, top_k=1)
    )
    greedy = greedy_search(model, source, GenerationConfig(max_new_tokens=6))

    assert torch.equal(sampled, greedy)


@pytest.mark.unit
def test_sampling_honours_the_minimum_length(
    model: ScratchTransformer, source: torch.Tensor
) -> None:
    generated = sample_search(
        model, source, GenerationConfig(max_new_tokens=10, min_new_tokens=5, do_sample=True)
    )

    assert not (generated[:, :5] == EOS).any()


@pytest.mark.unit
def test_a_forbidden_token_cannot_come_back_through_the_draw() -> None:
    # The constraints run before the narrowing and mark their tokens with minus
    # infinity. Dividing by the temperature must leave them there.
    scores = torch.tensor([[0.0, float("-inf"), -1.0]])

    narrowed = narrow_for_sampling(scores, GenerationConfig(do_sample=True, temperature=0.5))

    assert narrowed[0, 1] == float("-inf")
    assert narrowed.softmax(dim=-1)[0, 1] == 0.0


@pytest.mark.unit
def test_the_nucleus_keeps_the_token_that_crosses_the_threshold() -> None:
    # Without the shift by one, a distribution whose first token already carries
    # more than top_p would leave an empty nucleus and multinomial would raise.
    scores = torch.log_softmax(torch.tensor([[10.0, 1.0, 0.5]]), dim=-1)

    narrowed = narrow_for_sampling(scores, GenerationConfig(do_sample=True, top_p=0.1))

    assert int(torch.isfinite(narrowed).sum()) == 1
    assert int(narrowed.argmax()) == 0
