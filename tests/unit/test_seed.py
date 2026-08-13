"""Unit tests for the centralised seeding helper."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.utils.seed import DEFAULT_SEED, SeedState, seed_worker, set_seed


@pytest.mark.unit
def test_set_seed_returns_state() -> None:
    state = set_seed(123)

    assert isinstance(state, SeedState)
    assert state.seed == 123
    assert state.deterministic is True
    assert state.cuda_available == torch.cuda.is_available()


@pytest.mark.unit
def test_set_seed_makes_python_random_reproducible() -> None:
    set_seed(DEFAULT_SEED)
    first = [random.random() for _ in range(5)]

    set_seed(DEFAULT_SEED)
    second = [random.random() for _ in range(5)]

    assert first == second


@pytest.mark.unit
def test_set_seed_makes_numpy_reproducible() -> None:
    set_seed(7)
    first = np.random.rand(4)

    set_seed(7)
    second = np.random.rand(4)

    np.testing.assert_array_equal(first, second)


@pytest.mark.unit
def test_set_seed_makes_torch_reproducible() -> None:
    set_seed(7)
    first = torch.randn(3, 3)

    set_seed(7)
    second = torch.randn(3, 3)

    assert torch.equal(first, second)


@pytest.mark.unit
def test_different_seeds_produce_different_draws() -> None:
    set_seed(1)
    first = torch.randn(8)

    set_seed(2)
    second = torch.randn(8)

    assert not torch.equal(first, second)


@pytest.mark.unit
def test_deterministic_flag_drives_cudnn_settings() -> None:
    set_seed(0, deterministic=False)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True

    set_seed(0, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


@pytest.mark.unit
def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="non negative"):
        set_seed(-1)


@pytest.mark.unit
def test_seed_worker_is_reproducible_for_a_given_worker() -> None:
    set_seed(DEFAULT_SEED)
    seed_worker(3)
    first = [random.random(), float(np.random.rand())]

    set_seed(DEFAULT_SEED)
    seed_worker(3)
    second = [random.random(), float(np.random.rand())]

    assert first == second
