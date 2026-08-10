"""Unit tests for early stopping."""

from __future__ import annotations

import math

import pytest

from src.training.early_stopping import EarlyStopping

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_negative_patience_is_refused() -> None:
    with pytest.raises(ValueError, match="patience must be non negative"):
        EarlyStopping(-1)


@pytest.mark.unit
def test_a_negative_delta_is_refused() -> None:
    with pytest.raises(ValueError, match="min_delta must be non negative"):
        EarlyStopping(2, min_delta=-0.1)


@pytest.mark.unit
def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        EarlyStopping(2, mode="highest")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Minimisation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_first_value_is_always_an_improvement() -> None:
    stopper = EarlyStopping(2)

    assert stopper.update(4.2) is True
    assert stopper.best == 4.2


@pytest.mark.unit
def test_the_best_value_and_its_step_are_kept() -> None:
    stopper = EarlyStopping(2)

    stopper.update(1.0, step=0)
    stopper.update(0.5, step=1)
    stopper.update(0.9, step=2)

    assert stopper.best == 0.5
    assert stopper.best_step == 1


@pytest.mark.unit
def test_the_run_stops_after_the_patience_is_spent() -> None:
    stopper = EarlyStopping(2)

    stopper.update(1.0)
    assert stopper.update(1.1) is False
    assert stopper.should_stop is False
    assert stopper.update(1.2) is False
    assert stopper.should_stop is True


@pytest.mark.unit
def test_an_improvement_resets_the_patience() -> None:
    stopper = EarlyStopping(2)

    stopper.update(1.0)
    stopper.update(1.1)
    stopper.update(0.9)

    assert stopper.num_bad_evaluations == 0
    assert stopper.should_stop is False


@pytest.mark.unit
def test_a_change_below_the_delta_does_not_count() -> None:
    stopper = EarlyStopping(1, min_delta=0.01)

    stopper.update(1.0)
    stopper.update(0.995)

    assert stopper.best == 1.0
    assert stopper.should_stop is True


@pytest.mark.unit
def test_a_zero_patience_disables_stopping() -> None:
    stopper = EarlyStopping(0)

    stopper.update(1.0)
    for value in (2.0, 3.0, 4.0):
        stopper.update(value)

    assert stopper.enabled is False
    assert stopper.should_stop is False
    assert stopper.best == 1.0


# ---------------------------------------------------------------------------
# Maximisation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_maximisation_keeps_the_highest_value() -> None:
    stopper = EarlyStopping(2, mode="max")

    assert stopper.best == -math.inf
    assert stopper.update(0.31) is True
    assert stopper.update(0.28) is False
    assert stopper.update(0.35) is True
    assert stopper.best == 0.35


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_state_survives_a_round_trip() -> None:
    stopper = EarlyStopping(3, min_delta=0.01)
    stopper.update(1.0, step=0)
    stopper.update(1.2, step=1)

    reloaded = EarlyStopping(3, min_delta=0.01)
    reloaded.load_state_dict(stopper.state_dict())

    assert reloaded.best == stopper.best
    assert reloaded.best_step == stopper.best_step
    assert reloaded.num_bad_evaluations == stopper.num_bad_evaluations
    assert reloaded.should_stop == stopper.should_stop


@pytest.mark.unit
def test_a_resumed_run_stops_at_the_same_evaluation() -> None:
    # A stopper that forgot its counters would grant a fresh patience after
    # every interruption, so a resumed run would train longer than an
    # uninterrupted one and the two would not produce the same model.
    uninterrupted = EarlyStopping(2)
    for value in (1.0, 1.1, 1.2):
        uninterrupted.update(value)

    interrupted = EarlyStopping(2)
    interrupted.update(1.0)
    interrupted.update(1.1)
    resumed = EarlyStopping(2)
    resumed.load_state_dict(interrupted.state_dict())
    resumed.update(1.2)

    assert resumed.should_stop == uninterrupted.should_stop is True
