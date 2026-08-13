"""Unit tests for the optimiser construction."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.training.config import TrainingConfig
from src.training.optimizer import build_optimizer, split_decay_parameters


class TinyModel(nn.Module):
    """A model holding one matrix, one bias and one normalisation gain."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.embedding = nn.Embedding(6, 4)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(self.embedding(tokens)))


@pytest.mark.unit
def test_matrices_are_decayed_and_offsets_are_not() -> None:
    decayed, undecayed = split_decay_parameters(TinyModel())

    # linear.weight and embedding.weight on one side, linear.bias plus the
    # LayerNorm gain and bias on the other.
    assert [parameter.ndim for parameter in decayed] == [2, 2]
    assert [parameter.ndim for parameter in undecayed] == [1, 1, 1]


@pytest.mark.unit
def test_a_tied_matrix_is_collected_once() -> None:
    model = TinyModel()
    model.linear.weight = model.embedding.weight

    decayed, _ = split_decay_parameters(model)

    assert len(decayed) == 1


@pytest.mark.unit
def test_frozen_parameters_are_ignored() -> None:
    model = TinyModel()
    model.embedding.weight.requires_grad_(False)

    decayed, _ = split_decay_parameters(model)

    assert len(decayed) == 1


@pytest.mark.unit
def test_the_optimiser_carries_the_two_groups() -> None:
    config = TrainingConfig(weight_decay=0.05, learning_rate=1e-3)

    optimizer = build_optimizer(TinyModel(), config)

    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.05, 0.0]
    assert all(group["lr"] == 1e-3 for group in optimizer.param_groups)


@pytest.mark.unit
def test_the_moment_hyperparameters_are_honoured() -> None:
    config = TrainingConfig(adam_beta1=0.8, adam_beta2=0.95, adam_epsilon=1e-6)

    optimizer = build_optimizer(TinyModel(), config)

    assert optimizer.param_groups[0]["betas"] == (0.8, 0.95)
    assert optimizer.param_groups[0]["eps"] == 1e-6


@pytest.mark.unit
def test_a_model_without_trainable_parameter_is_refused() -> None:
    model = TinyModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with pytest.raises(ValueError, match="no trainable parameter"):
        build_optimizer(model, TrainingConfig())


@pytest.mark.unit
def test_a_step_moves_the_weights() -> None:
    model = TinyModel()
    optimizer = build_optimizer(model, TrainingConfig(learning_rate=0.1))
    before = model.linear.weight.detach().clone()

    model(torch.zeros(2, 3, dtype=torch.long)).sum().backward()
    optimizer.step()

    assert not torch.equal(before, model.linear.weight.detach())
