"""Unit tests for the training configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.training.config import TrainingConfig


@pytest.mark.unit
def test_the_defaults_are_valid() -> None:
    config = TrainingConfig()

    assert config.epochs > 0
    assert config.batch_size > 0
    assert config.scheduler == "linear"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epochs", 0),
        ("batch_size", 0),
        ("gradient_accumulation_steps", 0),
        ("learning_rate", 0.0),
        ("weight_decay", -0.1),
        ("warmup_ratio", 1.0),
        ("max_grad_norm", 0.0),
        ("label_smoothing", 1.0),
        ("early_stopping_patience", -1),
        ("mega_batch_factor", 0),
        ("keep_last_checkpoints", 0),
        ("log_every_steps", 0),
        ("max_steps", 0),
        ("seed", -1),
    ],
)
def test_out_of_range_values_are_refused(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(**{field: value})  # type: ignore[arg-type]


@pytest.mark.unit
def test_an_unknown_field_is_refused() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(learning_rat=1e-4)  # type: ignore[call-arg]


@pytest.mark.unit
def test_an_unknown_scheduler_is_refused() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(scheduler="triangular")  # type: ignore[arg-type]


@pytest.mark.unit
def test_an_unknown_device_is_refused() -> None:
    with pytest.raises(ValidationError, match="device must be one of"):
        TrainingConfig(device="tpu")


@pytest.mark.unit
def test_the_configuration_is_frozen() -> None:
    config = TrainingConfig()

    with pytest.raises(ValidationError):
        config.epochs = 3


@pytest.mark.unit
def test_the_effective_batch_size_multiplies_the_accumulation() -> None:
    config = TrainingConfig(batch_size=8, gradient_accumulation_steps=4)

    assert config.effective_batch_size == 32


@pytest.mark.unit
def test_the_warmup_is_a_share_of_the_run() -> None:
    config = TrainingConfig(warmup_ratio=0.1)

    assert config.warmup_steps(1000) == 100


@pytest.mark.unit
def test_a_short_run_still_warms_up_for_one_step() -> None:
    # Rounding down would give zero, and the first step would then be taken at
    # the peak rate, which is what the warmup exists to avoid.
    config = TrainingConfig(warmup_ratio=0.06)

    assert config.warmup_steps(5) == 1


@pytest.mark.unit
def test_a_zero_ratio_disables_the_warmup() -> None:
    assert TrainingConfig(warmup_ratio=0.0).warmup_steps(1000) == 0


@pytest.mark.unit
def test_an_empty_run_needs_no_warmup() -> None:
    assert TrainingConfig(warmup_ratio=0.1).warmup_steps(0) == 0
