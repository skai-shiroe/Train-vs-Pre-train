"""Optimiser construction.

AdamW is used rather than Adam because the two do not apply weight decay the
same way. Adam folds the decay into the gradient, so the adaptive denominator
rescales it and parameters with small gradients end up barely regularised.
AdamW applies the decay directly to the weights, independently of the moment
estimates, which is the behaviour the hyperparameter is meant to have.

The decay is applied to the matrices only. Biases and normalisation gains are
one dimensional and act as offsets: shrinking them towards zero constrains the
representation without any regularisation benefit.
"""

from __future__ import annotations

from torch import nn, optim

from src.training.config import TrainingConfig


def split_decay_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split the trainable parameters into decayed and undecayed groups.

    The split uses the number of dimensions rather than the parameter names.
    Matrices and embedding tables have two dimensions or more, biases and
    normalisation gains have one. That rule holds whatever the layers are
    called, so it cannot silently miss a renamed module.

    A tied matrix appears twice in ``named_parameters``. It is kept once,
    otherwise the optimiser would refuse the duplicate.

    Args:
        model: The model whose parameters are collected.

    Returns:
        A pair ``(decayed, undecayed)``.
    """
    decayed: list[nn.Parameter] = []
    undecayed: list[nn.Parameter] = []
    seen: set[int] = set()

    for parameter in model.parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if parameter.ndim >= 2:
            decayed.append(parameter)
        else:
            undecayed.append(parameter)

    return decayed, undecayed


def build_optimizer(model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
    """Build the AdamW optimiser of a run.

    Args:
        model: The model to optimise.
        config: Learning rate, decay and moment hyperparameters.

    Returns:
        The configured optimiser, holding two parameter groups.

    Raises:
        ValueError: If the model has no trainable parameter.
    """
    decayed, undecayed = split_decay_parameters(model)
    if not decayed and not undecayed:
        raise ValueError("The model exposes no trainable parameter.")

    groups = [
        {"params": decayed, "weight_decay": config.weight_decay},
        {"params": undecayed, "weight_decay": 0.0},
    ]

    return optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )
