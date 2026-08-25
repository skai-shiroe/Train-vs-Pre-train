"""Torch device resolution shared by training and evaluation.

An experiment file declares ``device: auto``. Resolving that string in a single
place avoids the drift between what the trainer used and what the evaluation
runs on, which would make a duration incomparable with the one the record
reports.
"""

from __future__ import annotations

import gc

import torch

VALID_DEVICES = ("auto", "cpu", "cuda")


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a device string into a concrete :class:`torch.device`.

    Args:
        requested: One of ``auto``, ``cpu`` or ``cuda``. ``auto`` selects CUDA
            when a GPU is visible and falls back to CPU otherwise.

    Returns:
        The resolved device.

    Raises:
        ValueError: If ``requested`` is not a supported value.
        RuntimeError: If ``cuda`` is requested explicitly but unavailable.
    """
    normalised = requested.strip().lower()
    if normalised not in VALID_DEVICES:
        raise ValueError(f"Device must be one of {VALID_DEVICES}, got {requested!r}.")

    if normalised == "cpu":
        return torch.device("cpu")

    if normalised == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested explicitly but no GPU is visible.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def supports_mixed_precision(device: torch.device) -> bool:
    """Return whether automatic mixed precision is worth enabling.

    Mixed precision only pays off on CUDA. Enabling it on CPU adds casting
    overhead without any speed gain.

    Args:
        device: The resolved training device.

    Returns:
        ``True`` when the device is a CUDA device.
    """
    return device.type == "cuda"


def describe_hardware() -> dict[str, str]:
    """Return a hardware summary, logged to MLflow with every run.

    Section 19 requires the hardware to be traced alongside the metrics so that
    a reported duration can be interpreted.

    Returns:
        A flat dictionary of hardware attributes, always JSON serialisable.
    """
    summary = {
        "torch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        summary["cuda_version"] = torch.version.cuda or "unknown"
        summary["gpu_name"] = torch.cuda.get_device_name(0)
        summary["gpu_count"] = str(torch.cuda.device_count())
        major, minor = torch.cuda.get_device_capability(0)
        summary["gpu_capability"] = f"sm_{major}{minor}"
    return summary


def release_accelerator() -> None:
    """Give the GPU memory of a finished run back to the driver.

    A campaign runs its experiments in one process, one after the other. Torch
    keeps freed blocks in its caching allocator rather than returning them, so
    what the previous experiments reserved is still held when the next one
    builds its model. On a card with little memory the sixth run then fails on
    a driver level ``CUDA error: out of memory`` although it fits by itself,
    and every run after it fails the same way: that error leaves the CUDA
    context unusable for the rest of the process.

    Collecting first matters. A model, its optimiser and its scheduler
    reference each other, so the cycle collector is what drops them; without
    it, ``empty_cache`` returns blocks that are still owned and frees almost
    nothing.

    This is a no-op without CUDA.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
