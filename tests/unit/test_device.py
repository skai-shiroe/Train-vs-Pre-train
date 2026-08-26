"""Unit tests for torch device resolution."""

from __future__ import annotations

import pytest
import torch

from src.utils.device import (
    VALID_DEVICES,
    describe_hardware,
    release_accelerator,
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


@pytest.mark.unit
def test_releasing_the_accelerator_is_safe_without_cuda() -> None:
    """The campaign calls this after every experiment, GPU or not.

    A chain that only ran on the reference machine would break the moment it
    was replayed on a laptop without CUDA, and the failure would land in the
    middle of a campaign rather than at its first line.
    """
    release_accelerator()
    release_accelerator()


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="demande un GPU CUDA")
def test_releasing_the_accelerator_returns_reserved_memory() -> None:
    """What the caching allocator holds must go back before the next run.

    Ten experiments share one process. Without this, what the earlier ones
    reserved is still held when a later one builds its model, and the card of
    the reference machine fails on a driver level out of memory although each
    experiment fits by itself.
    """
    block = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    del block
    reserved_before = torch.cuda.memory_reserved()

    release_accelerator()

    assert torch.cuda.memory_reserved() < reserved_before
