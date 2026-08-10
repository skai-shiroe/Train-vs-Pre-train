"""Writing, rotating and reloading checkpoints.

A checkpoint holds everything needed to continue a run, not only the weights:
the optimiser moments, the scheduler position, the gradient scaler, the early
stopping counters and the state of every random generator. Restoring the
weights alone would produce a different model than an uninterrupted run, which
would break the reproducibility the ablation study rests on.

Every payload is made of tensors and primitives only, so it reloads with
``weights_only=True``. A checkpoint is data, and loading it must never be able
to execute code, even when it comes from the project's own artifact store.

Writes go to a temporary file and are then renamed. A crash during a write
leaves the previous checkpoint intact rather than a truncated one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn, optim
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LRScheduler

from src.training.early_stopping import EarlyStopping
from src.utils.seed import capture_rng_state, restore_rng_state

#: Name of the checkpoint holding the best validation loss.
BEST_NAME = "best.pt"

#: Format of the periodic checkpoints. Zero padded so that a lexicographic sort
#: matches the numeric one.
CHECKPOINT_TEMPLATE = "checkpoint-step-{step:08d}.pt"
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-step-(\d{8})\.pt$")

#: Version of the payload layout, checked when reloading.
PAYLOAD_VERSION = 1


@dataclass(frozen=True, slots=True)
class ResumeState:
    """Where a reloaded run must pick up.

    Attributes:
        epoch: Index of the epoch that was completed last.
        global_step: Number of optimisation steps already performed.
        best_validation_loss: Best validation loss reached so far.
    """

    epoch: int
    global_step: int
    best_validation_loss: float


def build_payload(
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: LRScheduler | None = None,
    scaler: GradScaler | None = None,
    early_stopping: EarlyStopping | None = None,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the mapping a checkpoint stores.

    Args:
        model: The model whose weights are stored.
        optimizer: The optimiser whose moments are stored.
        scheduler: The learning rate scheduler, when one is used.
        scaler: The gradient scaler, when mixed precision is on.
        early_stopping: The stopper whose counters are stored.
        epoch: Index of the epoch just completed.
        global_step: Number of optimisation steps performed.
        best_validation_loss: Best validation loss reached so far.
        metadata: Free form mapping of primitives, typically the run
            configuration and the dataset version.

    Returns:
        The payload, ready to be written.
    """
    return {
        "version": PAYLOAD_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "early_stopping": early_stopping.state_dict() if early_stopping is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "rng": capture_rng_state(),
        "metadata": metadata or {},
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> Path:
    """Write a payload to disk atomically.

    Args:
        path: Destination file. Parent directories are created.
        payload: Mapping produced by :func:`build_payload`.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    # nosec B614 - writing is not a deserialisation risk, and the payload holds
    # only tensors and primitives, which is what lets the reader stay on
    # weights_only=True.
    torch.save(payload, temporary)  # nosec B614
    temporary.replace(path)
    return path


def load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Read a checkpoint from disk.

    Args:
        path: Source file.
        map_location: Device the tensors are read onto. Reading onto the CPU
            first lets a run trained on GPU resume on a machine without one.

    Returns:
        The stored payload.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the payload layout is not the expected version.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # nosec B614 - the finding asks for weights_only, which is already set. The
    # rule flags every torch.load call, so the exception is on the false
    # positive, never on the protection itself.
    payload: dict[str, Any] = torch.load(  # nosec B614
        path, map_location=map_location, weights_only=True
    )

    version = payload.get("version")
    if version != PAYLOAD_VERSION:
        raise ValueError(
            f"{path} was written by payload version {version!r}, "
            f"this build reads version {PAYLOAD_VERSION}."
        )
    return payload


def restore(
    payload: dict[str, Any],
    *,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    scaler: GradScaler | None = None,
    early_stopping: EarlyStopping | None = None,
    restore_rng: bool = True,
) -> ResumeState:
    """Load a payload into the objects of a run.

    Args:
        payload: Mapping read by :func:`load_checkpoint`.
        model: The model the weights are loaded into.
        optimizer: The optimiser the moments are loaded into.
        scheduler: The scheduler whose position is restored.
        scaler: The gradient scaler whose scale is restored.
        early_stopping: The stopper whose counters are restored.
        restore_rng: Whether to restore the random generators. Turned off when
            the checkpoint is loaded for inference rather than for a resume.

    Returns:
        Where the run must pick up.
    """
    model.load_state_dict(payload["model"])

    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if early_stopping is not None and payload.get("early_stopping") is not None:
        early_stopping.load_state_dict(payload["early_stopping"])
    if restore_rng and payload.get("rng"):
        restore_rng_state(payload["rng"])

    return ResumeState(
        epoch=int(payload["epoch"]),
        global_step=int(payload["global_step"]),
        best_validation_loss=float(payload["best_validation_loss"]),
    )


class CheckpointManager:
    """Own the checkpoint directory of one run."""

    def __init__(self, directory: Path, *, keep_last: int = 2) -> None:
        """Build the manager.

        Args:
            directory: Directory holding the checkpoints of the run.
            keep_last: Number of periodic checkpoints kept. The best one is
                kept whatever this value is, so a long run cannot rotate away
                the model it is going to report.

        Raises:
            ValueError: If ``keep_last`` is not strictly positive.
        """
        if keep_last <= 0:
            raise ValueError(f"keep_last must be strictly positive, got {keep_last}.")
        self.directory = directory
        self.keep_last = keep_last

    @property
    def best_path(self) -> Path:
        """Return the path of the best checkpoint.

        Returns:
            The path, whether or not the file exists yet.
        """
        return self.directory / BEST_NAME

    def periodic_checkpoints(self) -> list[Path]:
        """List the periodic checkpoints, oldest first.

        Returns:
            The paths, sorted by step.
        """
        if not self.directory.is_dir():
            return []
        found = [path for path in self.directory.iterdir() if CHECKPOINT_PATTERN.match(path.name)]
        return sorted(found, key=lambda path: path.name)

    def latest(self) -> Path | None:
        """Return the checkpoint a resume must start from.

        Returns:
            The most recent periodic checkpoint, or the best one when no
            periodic checkpoint survived the rotation, or ``None``.
        """
        periodic = self.periodic_checkpoints()
        if periodic:
            return periodic[-1]
        return self.best_path if self.best_path.is_file() else None

    def save(self, payload: dict[str, Any], *, step: int, is_best: bool = False) -> Path:
        """Write a checkpoint and rotate the older ones.

        Args:
            payload: Mapping produced by :func:`build_payload`.
            step: Optimisation step the checkpoint was taken at.
            is_best: Whether this checkpoint also becomes the best one.

        Returns:
            The path of the periodic checkpoint written.
        """
        path = save_checkpoint(self.directory / CHECKPOINT_TEMPLATE.format(step=step), payload)
        if is_best:
            save_checkpoint(self.best_path, payload)
        self.prune()
        return path

    def prune(self) -> list[Path]:
        """Delete the periodic checkpoints beyond the retention limit.

        Returns:
            The paths that were removed.
        """
        periodic = self.periodic_checkpoints()
        removed: list[Path] = []
        for path in periodic[: max(0, len(periodic) - self.keep_last)]:
            path.unlink()
            removed.append(path)
        return removed
