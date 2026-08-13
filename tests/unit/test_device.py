"""Unit tests for torch device resolution."""

from __future__ import annotations

import pytest
import torch

from src.utils.device import (
    VALID_DEVICES,
    describe_hardware,
    resolve_device,
    supports_mixed_precision,
)


@pytest.mark.unit
def test_cpu_is_always_resolvable() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


@pytest.mark.unit
def test_auto_matches_cuda_availability() -> None:
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto").type == expected


@pytest.mark.unit
@pytest.mark.parametrize("value", ["AUTO", " Cpu ", "CUDA"])
def test_resolution_is_case_and_space_insensitive(value: str) -> None:
    if value.strip().lower() == "cuda" and not torch.cuda.is_available():
        pytest.skip("No CUDA device visible on this runner.")
    assert resolve_device(value).type in VALID_DEVICES


@pytest.mark.unit
def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="Device must be one of"):
        resolve_device("tpu")


@pytest.mark.unit
def test_explicit_cuda_fails_when_unavailable() -> None:
    if torch.cuda.is_available():
        pytest.skip("A CUDA device is visible, the failure path cannot trigger.")
    with pytest.raises(RuntimeError, match="no GPU is visible"):
        resolve_device("cuda")


@pytest.mark.unit
def test_mixed_precision_only_on_cuda() -> None:
    assert supports_mixed_precision(torch.device("cpu")) is False
    assert supports_mixed_precision(torch.device("cuda")) is True


@pytest.mark.unit
def test_hardware_summary_is_flat_and_serialisable() -> None:
    summary = describe_hardware()

    assert summary["torch_version"] == torch.__version__
    assert set(summary) >= {"torch_version", "cuda_available"}
    assert all(isinstance(value, str) for value in summary.values())
