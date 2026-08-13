"""Centralised seeding, required by section 14 of the specification.

Every experiment calls :func:`set_seed` exactly once, before building the data
loaders and the model. Without it the ablation study on corpus size cannot be
reproduced, and the comparison between the from scratch model and the
pretrained baseline loses its meaning.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

#: Seed used by every experiment configuration unless overridden.
DEFAULT_SEED = 42


@dataclass(frozen=True)
class SeedState:
    """Snapshot of what :func:`set_seed` configured.

    Attributes:
        seed: The seed applied to every generator.
        deterministic: Whether cuDNN was forced into deterministic mode.
        cuda_available: Whether a CUDA device was visible at seeding time.
    """

    seed: int
    deterministic: bool
    cuda_available: bool


def set_seed(seed: int = DEFAULT_SEED, *, deterministic: bool = True) -> SeedState:
    """Seed every random generator used by the project.

    The function covers the four sources listed in section 14: ``random``,
    ``numpy``, ``torch`` and ``torch.cuda``. It also pins ``PYTHONHASHSEED`` so
    that set and dict ordering stays stable across processes.

    Deterministic mode trades throughput for reproducibility. It is enabled by
    default because the scientific claims of the project depend on it. Disable
    it only for exploratory runs whose results are not reported.

    Args:
        seed: Value applied to every generator. Must be non negative.
        deterministic: When ``True``, force cuDNN into deterministic mode and
            disable the autotuner benchmark.

    Returns:
        A :class:`SeedState` describing what was configured.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non negative, got {seed}.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    return SeedState(seed=seed, deterministic=deterministic, cuda_available=cuda_available)


def capture_rng_state() -> dict[str, Any]:
    """Snapshot every random generator, for a checkpoint.

    Resuming a run that only restored the weights would replay a different
    shuffling and a different dropout mask than an uninterrupted run, so the
    two would not converge to the same model. Storing the generator states
    makes ``resume`` a continuation rather than a restart.

    Every value is a primitive or a tensor, never a NumPy array or an arbitrary
    object, so the checkpoint can be reloaded with ``weights_only=True``.

    Returns:
        A mapping accepted by :func:`restore_rng_state`.
    """
    kind, keys, position, has_gauss, cached_gaussian = np.random.get_state()

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "kind": str(kind),
            "keys": [int(key) for key in keys],
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        },
        "torch": torch.get_rng_state(),
        "cuda": list(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else [],
    }
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore the generators snapshotted by :func:`capture_rng_state`.

    The CUDA states are restored only when the resumed run sees the same number
    of devices. Restoring a four GPU state onto one GPU would raise, and a run
    that changed hardware cannot claim to be a bit exact continuation anyway.

    Args:
        state: Mapping produced by :func:`capture_rng_state`.
    """
    python_state = state["python"]
    random.setstate((python_state[0], tuple(python_state[1]), python_state[2]))

    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["kind"],
            np.array(numpy_state["keys"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )

    torch.set_rng_state(state["torch"].to(torch.uint8))

    cuda_states = state.get("cuda") or []
    restorable = torch.cuda.is_available() and len(cuda_states) == torch.cuda.device_count()
    if cuda_states and restorable:
        torch.cuda.set_rng_state_all([tensor.to(torch.uint8) for tensor in cuda_states])


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker process.

    PyTorch gives each worker a distinct base seed derived from the main
    process. Deriving the ``random`` and ``numpy`` seeds from it keeps the
    shuffling reproducible when ``num_workers`` is greater than zero.

    Args:
        worker_id: Index of the worker, supplied by the DataLoader.
    """
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
