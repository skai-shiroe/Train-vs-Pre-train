"""Torch device resolution shared by training and evaluation.

An experiment file declares ``device: auto``. Resolving that string in a single
place avoids the drift between what the trainer used and what the evaluation
runs on, which would make a duration incomparable with the one the record
reports.
"""

from __future__ import annotations

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
