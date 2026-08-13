"""Records exchanged between the trainer, the callbacks and the reports.

Keeping these in their own module lets the callbacks describe what they observe
without importing the trainer, and lets the trainer emit events without
importing a concrete callback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainingState:
    """Mutable view of a run in progress, handed to every callback.

    Attributes:
        epoch: Zero based index of the epoch being run.
        epochs: Total number of epochs planned.
        global_step: Number of optimisation steps completed since the start.
        total_steps: Total number of optimisation steps planned.
        learning_rate: Rate applied at the last step.
        last_loss: Loss of the last batch, before accumulation scaling.
        running_loss: Mean training loss over the epoch so far.
        grad_norm: Global gradient norm measured at the last step, before
            clipping. A norm that stays pinned at the clipping threshold means
            the learning rate is too high.
        stopped_early: Whether early stopping ended the run.
    """

    epoch: int = 0
    epochs: int = 0
    global_step: int = 0
    total_steps: int = 0
    learning_rate: float = 0.0
    last_loss: float = 0.0
    running_loss: float = 0.0
    grad_norm: float = 0.0
    stopped_early: bool = False


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """What one epoch produced.

    Attributes:
        epoch: Zero based epoch index.
        train_loss: Mean training loss over the epoch.
        validation_loss: Loss on the validation split.
        learning_rate: Rate applied at the last step of the epoch.
        steps: Number of optimisation steps performed during the epoch.
        duration_seconds: Wall clock duration of the epoch.
        improved: Whether the validation loss improved on the best so far.
    """

    epoch: int
    train_loss: float
    validation_loss: float
    learning_rate: float
    steps: int
    duration_seconds: float
    improved: bool

    def to_dict(self) -> dict[str, Any]:
        """Render the metrics as a flat mapping.

        Returns:
            A mapping suitable for MLflow and for the results CSV files.
        """
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """What a whole run produced.

    Attributes:
        epochs: Metrics of every epoch actually run.
        best_epoch: Index of the epoch holding the best validation loss.
        best_validation_loss: The best validation loss reached.
        stopped_early: Whether early stopping ended the run.
        global_step: Number of optimisation steps performed.
        duration_seconds: Wall clock duration of the whole run.
        best_checkpoint: Path of the checkpoint holding the best epoch.
    """

    epochs: tuple[EpochMetrics, ...] = ()
    best_epoch: int = -1
    best_validation_loss: float = float("inf")
    stopped_early: bool = False
    global_step: int = 0
    duration_seconds: float = 0.0
    best_checkpoint: Path | None = None
    history: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render the result as a JSON serialisable mapping.

        Returns:
            A mapping holding the summary and the per epoch metrics.
        """
        return {
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "stopped_early": self.stopped_early,
            "global_step": self.global_step,
            "duration_seconds": self.duration_seconds,
            "best_checkpoint": str(self.best_checkpoint) if self.best_checkpoint else None,
            "epochs": [metrics.to_dict() for metrics in self.epochs],
            "history": self.history,
        }
