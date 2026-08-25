"""Unit tests for the assembled encoder decoder Transformer.

The two tests that matter most are the causality check and the padding
invariance check. A broken causal mask still produces a falling training loss,
so only an explicit test catches it.
"""

from __future__ import annotations

import pytest
import torch

from src.models.scratch.config import ScratchTransformerConfig
from src.models.scratch.transformer import (
    IGNORE_INDEX,
    ScratchTransformer,
    shift_target_right,
)
from src.utils.seed import set_seed

VOCAB_SIZE = 40
PAD = 0
EOS = 1


def build_config(**overrides: object) -> ScratchTransformerConfig:
    """Return a small configuration, fast enough for a unit test."""
    defaults: dict[str, object] = {
        "vocab_size": VOCAB_SIZE,
        "d_model": 32,
        "num_heads": 4,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "d_ff": 64,
        "dropout": 0.0,
        "max_position": 64,
        "pad_token_id": PAD,
        "eos_token_id": EOS,
        "decoder_start_token_id": PAD,
    }
    defaults.update(overrides)
    return ScratchTransformerConfig(**defaults)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    set_seed(1234)


@pytest.fixture
def model() -> ScratchTransformer:
    return ScratchTransformer(build_config()).eval()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_d_head_splits_the_width() -> None:
    assert build_config(d_model=64, num_heads=8).d_head == 8


@pytest.mark.unit
def test_heads_must_divide_the_width() -> None:
    with pytest.raises(ValueError, match="must be divisible by num_heads"):
        build_config(d_model=30, num_heads=4)


@pytest.mark.unit
@pytest.mark.parametrize("field", ["vocab_size", "d_model", "num_heads", "d_ff", "max_position"])
def test_dimensions_must_be_positive(field: str) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        build_config(**{field: 0})


@pytest.mark.unit
def test_each_stack_needs_at_least_one_layer() -> None:
    with pytest.raises(ValueError, match="at least one layer"):
        build_config(num_encoder_layers=0)


@pytest.mark.unit
@pytest.mark.parametrize("dropout", [-0.1, 1.0, 1.5])
def test_dropout_must_stay_below_one(dropout: float) -> None:
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\)"):
        build_config(dropout=dropout)


@pytest.mark.unit
def test_configuration_serialises_flat() -> None:
    payload = build_config().to_dict()

    assert payload["d_model"] == 32
    assert all(not isinstance(value, dict) for value in payload.values())


# ---------------------------------------------------------------------------
# Teacher forcing shift
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_shift_inserts_the_start_token_and_drops_the_last() -> None:
    target = torch.tensor([[7, 8, 9]])

    shifted = shift_target_right(target, decoder_start_token_id=PAD)

    assert shifted.tolist() == [[PAD, 7, 8]]


@pytest.mark.unit
def test_shift_keeps_the_shape_and_dtype() -> None:
    target = torch.randint(2, VOCAB_SIZE, (4, 9))

    shifted = shift_target_right(target, PAD)

    assert shifted.shape == target.shape
    assert shifted.dtype == target.dtype


@pytest.mark.unit
def test_shift_does_not_modify_its_input() -> None:
    target = torch.tensor([[7, 8, 9]])
    original = target.clone()

    shift_target_right(target, PAD)

    assert torch.equal(target, original)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_logits_have_one_score_per_vocabulary_entry(model: ScratchTransformer) -> None:
    source = torch.randint(2, VOCAB_SIZE, (3, 11))
    target = torch.randint(2, VOCAB_SIZE, (3, 6))

    output = model(source, target_ids=target)

    assert output.logits.shape == (3, 6, VOCAB_SIZE)
    assert output.loss is None


@pytest.mark.unit
def test_loss_is_computed_when_labels_are_supplied(model: ScratchTransformer) -> None:
    source = torch.randint(2, VOCAB_SIZE, (2, 9))
    target = torch.randint(2, VOCAB_SIZE, (2, 5))

    output = model(source, target_ids=target, labels=target)

    assert output.loss is not None
    assert torch.isfinite(output.loss)
    assert float(output.loss.detach()) > 0


@pytest.mark.unit
def test_ignored_labels_do_not_contribute_to_the_loss(model: ScratchTransformer) -> None:
    source = torch.randint(2, VOCAB_SIZE, (1, 9))
    target = torch.randint(2, VOCAB_SIZE, (1, 6))

    labels = target.clone()
    full = model(source, target_ids=target, labels=labels).loss

    masked_labels = target.clone()
    masked_labels[:, 3:] = IGNORE_INDEX
    partial = model(source, target_ids=target, labels=masked_labels).loss

    assert full is not None
    assert partial is not None
    assert not torch.allclose(full, partial)


@pytest.mark.unit
def test_supplying_neither_target_nor_decoder_input_is_refused(
    model: ScratchTransformer,
) -> None:
    with pytest.raises(ValueError, match="Supply either target_ids or decoder_input_ids"):
        model(torch.randint(2, VOCAB_SIZE, (1, 5)))


@pytest.mark.unit
def test_attention_weights_are_collected_on_demand(model: ScratchTransformer) -> None:
    source = torch.randint(2, VOCAB_SIZE, (1, 7))
    target = torch.randint(2, VOCAB_SIZE, (1, 4))

    output = model(source, target_ids=target, return_weights=True)

    assert len(output.encoder_attentions) == model.config.num_encoder_layers
    assert len(output.decoder_attentions) == model.config.num_decoder_layers
    assert output.decoder_attentions[0].cross_attention is not None
    assert output.decoder_attentions[0].cross_attention.shape == (1, 4, 4, 7)


@pytest.mark.unit
def test_attention_weights_are_discarded_by_default(model: ScratchTransformer) -> None:
    output = model(
        torch.randint(2, VOCAB_SIZE, (1, 7)),
        target_ids=torch.randint(2, VOCAB_SIZE, (1, 4)),
    )

    assert output.encoder_attentions == ()
    assert output.decoder_attentions == ()


# ---------------------------------------------------------------------------
# The two properties the architecture must satisfy
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_projecting_the_last_position_gives_the_same_scores(model: ScratchTransformer) -> None:
    # C'est la seule ligne qu'une boucle de generation lit, et projeter les
    # autres est ce qui coute : au dela de la derniere position, le tenseur
    # produit est le seul du modele dont la derniere dimension est le
    # vocabulaire. Les valeurs doivent etre identiques, sinon le decodage
    # change de resultat.
    source = torch.randint(2, VOCAB_SIZE, (2, 9))
    decoder_input = torch.randint(2, VOCAB_SIZE, (2, 6))
    memory, memory_mask, _ = model.encode(source)

    whole, _ = model.decode(decoder_input, memory, memory_mask)
    last, _ = model.decode(decoder_input, memory, memory_mask, last_position_only=True)

    assert last.shape == (2, 1, VOCAB_SIZE)
    assert torch.allclose(last[:, -1, :], whole[:, -1, :], atol=1e-6)


@pytest.mark.unit
def test_a_decoder_position_never_reads_the_future(model: ScratchTransformer) -> None:
    source = torch.randint(2, VOCAB_SIZE, (1, 8))
    decoder_input = torch.randint(2, VOCAB_SIZE, (1, 7))
    cut = 4

    reference = model(source, decoder_input_ids=decoder_input).logits

    edited = decoder_input.clone()
    edited[0, cut] = (int(edited[0, cut]) + 5) % VOCAB_SIZE
    modified = model(source, decoder_input_ids=edited).logits

    # Everything before the edited position must be bit for bit unaffected.
    assert torch.allclose(reference[:, :cut], modified[:, :cut], atol=1e-6)
    # And the edited position itself must react, otherwise the test proves nothing.
    assert not torch.allclose(reference[:, cut], modified[:, cut], atol=1e-6)


@pytest.mark.unit
def test_source_padding_does_not_change_the_prediction(model: ScratchTransformer) -> None:
    source = torch.randint(2, VOCAB_SIZE, (1, 6))
    padded = torch.cat([source, torch.full((1, 5), PAD)], dim=1)
    decoder_input = torch.randint(2, VOCAB_SIZE, (1, 4))

    without = model(source, decoder_input_ids=decoder_input).logits
    with_padding = model(padded, decoder_input_ids=decoder_input).logits

    assert torch.allclose(without, with_padding, atol=1e-5)


@pytest.mark.unit
def test_a_fully_padded_source_does_not_produce_nan(model: ScratchTransformer) -> None:
    source = torch.full((1, 6), PAD)
    decoder_input = torch.randint(2, VOCAB_SIZE, (1, 3))

    logits = model(source, decoder_input_ids=decoder_input).logits

    assert not bool(torch.isnan(logits).any())


@pytest.mark.unit
def test_examples_of_a_batch_do_not_influence_each_other(model: ScratchTransformer) -> None:
    source = torch.randint(2, VOCAB_SIZE, (3, 7))
    decoder_input = torch.randint(2, VOCAB_SIZE, (3, 5))

    batched = model(source, decoder_input_ids=decoder_input).logits
    alone = model(source[1:2], decoder_input_ids=decoder_input[1:2]).logits

    assert torch.allclose(batched[1:2], alone, atol=1e-5)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tying_shares_the_embedding_matrix() -> None:
    model = ScratchTransformer(build_config(tie_embeddings=True))

    assert model.decoder.output_projection.weight is model.embedding.weight


@pytest.mark.unit
def test_untying_creates_a_separate_matrix() -> None:
    model = ScratchTransformer(build_config(tie_embeddings=False))

    assert model.decoder.output_projection.weight is not model.embedding.weight


@pytest.mark.unit
def test_tying_reduces_the_parameter_count() -> None:
    tied = ScratchTransformer(build_config(tie_embeddings=True)).num_parameters
    untied = ScratchTransformer(build_config(tie_embeddings=False)).num_parameters

    assert untied - tied == VOCAB_SIZE * 32


@pytest.mark.unit
def test_both_towers_share_one_embedding_table(model: ScratchTransformer) -> None:
    assert model.encoder.embedding is model.decoder.embedding


@pytest.mark.unit
def test_every_parameter_receives_a_gradient(model: ScratchTransformer) -> None:
    model.train()
    source = torch.randint(2, VOCAB_SIZE, (2, 7))
    target = torch.randint(2, VOCAB_SIZE, (2, 5))

    output = model(source, target_ids=target, labels=target)
    assert output.loss is not None
    output.loss.backward()

    without_gradient = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]

    assert without_gradient == []


@pytest.mark.unit
@pytest.mark.parametrize("norm_first", [True, False])
def test_both_normalisation_layouts_run(norm_first: bool) -> None:
    model = ScratchTransformer(build_config(norm_first=norm_first)).eval()

    output = model(
        torch.randint(2, VOCAB_SIZE, (1, 6)),
        target_ids=torch.randint(2, VOCAB_SIZE, (1, 4)),
    )

    assert output.logits.shape == (1, 4, VOCAB_SIZE)


@pytest.mark.unit
def test_the_model_can_overfit_a_single_batch() -> None:
    # The strongest end to end check available without a real training run: if
    # the loss does not collapse on one batch, something in the wiring is wrong.
    set_seed(0)
    model = ScratchTransformer(build_config(dropout=0.0))
    model.train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)

    source = torch.randint(2, VOCAB_SIZE, (2, 8))
    target = torch.randint(2, VOCAB_SIZE, (2, 5))

    first_loss = None
    last_loss = None
    for _ in range(60):
        optimiser.zero_grad()
        output = model(source, target_ids=target, labels=target)
        assert output.loss is not None
        last_loss = float(output.loss.detach())
        output.loss.backward()
        optimiser.step()
        if first_loss is None:
            first_loss = last_loss

    assert first_loss is not None
    assert last_loss is not None
    assert last_loss < first_loss * 0.2
