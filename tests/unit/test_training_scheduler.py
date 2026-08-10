"""Unit tests for the learning rate schedules."""

from __future__ import annotations

import pytest
import torch
from torch import nn, optim

from src.training.scheduler import build_lambda, build_scheduler

TOTAL = 100
WARMUP = 10


@pytest.fixture
def optimizer() -> optim.Optimizer:
    return optim.SGD(nn.Linear(2, 2).parameters(), lr=1.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_an_unknown_schedule_is_refused() -> None:
    with pytest.raises(ValueError, match="Scheduler must be one of"):
        build_lambda("triangular", WARMUP, TOTAL)  # type: ignore[arg-type]


@pytest.mark.unit
def test_an_empty_run_is_refused() -> None:
    with pytest.raises(ValueError, match="total_steps must be strictly positive"):
        build_lambda("linear", 0, 0)


@pytest.mark.unit
def test_a_negative_warmup_is_refused() -> None:
    with pytest.raises(ValueError, match="warmup_steps must be non negative"):
        build_lambda("linear", -1, TOTAL)


@pytest.mark.unit
def test_a_warmup_longer_than_the_run_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot be longer than the run"):
        build_lambda("linear", TOTAL + 1, TOTAL)


# ---------------------------------------------------------------------------
# Warmup, shared by every schedule
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("name", ["linear", "cosine", "inverse_sqrt", "constant"])
def test_the_warmup_ramps_up_to_the_peak(name: str) -> None:
    schedule = build_lambda(name, WARMUP, TOTAL)  # type: ignore[arg-type]

    factors = [schedule(step) for step in range(WARMUP)]

    assert factors == sorted(factors)
    assert factors[-1] == pytest.approx(1.0)


@pytest.mark.unit
def test_the_first_step_is_never_taken_at_a_rate_of_zero() -> None:
    assert build_lambda("linear", WARMUP, TOTAL)(0) > 0.0


@pytest.mark.unit
@pytest.mark.parametrize("name", ["linear", "cosine", "constant"])
def test_a_run_without_warmup_starts_at_the_peak(name: str) -> None:
    assert build_lambda(name, 0, TOTAL)(0) == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------------------
# Shapes after the warmup
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_linear_schedule_reaches_zero_at_the_end() -> None:
    schedule = build_lambda("linear", WARMUP, TOTAL)

    assert schedule(TOTAL - 1) < 0.05
    assert schedule(TOTAL) == pytest.approx(0.0)


@pytest.mark.unit
def test_the_linear_schedule_never_goes_negative() -> None:
    schedule = build_lambda("linear", WARMUP, TOTAL)

    assert schedule(TOTAL * 3) == pytest.approx(0.0)


@pytest.mark.unit
def test_the_cosine_schedule_is_flat_at_both_ends() -> None:
    schedule = build_lambda("cosine", WARMUP, TOTAL)

    just_after_warmup = schedule(WARMUP + 1) - schedule(WARMUP + 2)
    at_the_middle = schedule(TOTAL // 2) - schedule(TOTAL // 2 + 1)

    assert just_after_warmup < at_the_middle
    assert schedule(TOTAL) == pytest.approx(0.0)


@pytest.mark.unit
def test_the_constant_schedule_holds_the_peak() -> None:
    schedule = build_lambda("constant", WARMUP, TOTAL)

    assert schedule(WARMUP) == 1.0
    assert schedule(TOTAL * 2) == 1.0


@pytest.mark.unit
def test_the_inverse_square_root_schedule_decays_slowly() -> None:
    schedule = build_lambda("inverse_sqrt", WARMUP, TOTAL)

    # sqrt(warmup / step): at four times the warmup the rate is halved.
    assert schedule(4 * WARMUP - 1) == pytest.approx(0.5)
    assert schedule(TOTAL) > 0.0


@pytest.mark.unit
def test_the_inverse_square_root_schedule_survives_a_missing_warmup() -> None:
    assert build_lambda("inverse_sqrt", 0, TOTAL)(5) == 1.0


@pytest.mark.unit
@pytest.mark.parametrize("name", ["linear", "cosine", "inverse_sqrt", "constant"])
def test_no_schedule_ever_returns_a_negative_factor(name: str) -> None:
    schedule = build_lambda(name, WARMUP, TOTAL)  # type: ignore[arg-type]

    assert all(schedule(step) >= 0.0 for step in range(TOTAL * 2))


# ---------------------------------------------------------------------------
# Wiring to a real optimiser
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_scheduler_drives_the_optimiser_rate(optimizer: optim.Optimizer) -> None:
    scheduler = build_scheduler(optimizer, "linear", warmup_steps=WARMUP, total_steps=TOTAL)
    rates = []

    for _ in range(TOTAL):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])

    assert rates[WARMUP - 1] == pytest.approx(max(rates))
    assert rates[-1] < rates[WARMUP - 1]


@pytest.mark.unit
def test_the_schedule_position_survives_a_reload(optimizer: optim.Optimizer) -> None:
    scheduler = build_scheduler(optimizer, "cosine", warmup_steps=WARMUP, total_steps=TOTAL)
    for _ in range(30):
        optimizer.step()
        scheduler.step()
    state = scheduler.state_dict()

    fresh_optimizer = optim.SGD(nn.Linear(2, 2).parameters(), lr=1.0)
    reloaded = build_scheduler(fresh_optimizer, "cosine", warmup_steps=WARMUP, total_steps=TOTAL)
    reloaded.load_state_dict(state)

    assert reloaded.get_last_lr() == pytest.approx(scheduler.get_last_lr())


@pytest.mark.unit
def test_the_rate_is_scaled_from_the_peak_of_each_group() -> None:
    model = nn.Linear(2, 2)
    optimizer = optim.SGD(
        [
            {"params": [model.weight], "lr": 1.0},
            {"params": [model.bias], "lr": 0.5},
        ]
    )
    scheduler = build_scheduler(optimizer, "constant", warmup_steps=0, total_steps=TOTAL)

    optimizer.step()
    scheduler.step()

    assert torch.tensor(optimizer.param_groups[0]["lr"]) == pytest.approx(1.0)
    assert torch.tensor(optimizer.param_groups[1]["lr"]) == pytest.approx(0.5)
