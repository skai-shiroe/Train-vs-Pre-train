"""Early stopping on the validation metric.

A randomly initialised T5 trained on 20 000 examples can overfit before the
last planned epoch. Running to the end would report the score of an overfitted
model, and would make the ablation on corpus size measure patience rather than
data. Early stopping cuts the run at the best validation loss instead.

The monitored value is the validation loss, minimised. The class also supports
maximisation so that a later run can monitor ROUGE-L directly.
"""

from __future__ import annotations

import math
from typing import Any, Literal

Mode = Literal["min", "max"]


class EarlyStopping:
    """Track a validation metric and decide when to stop."""

    def __init__(
        self,
        patience: int,
        *,
        min_delta: float = 0.0,
        mode: Mode = "min",
    ) -> None:
        """Build the stopper.

        Args:
            patience: Number of evaluations without improvement tolerated.
                Zero disables early stopping entirely.
            min_delta: Smallest change that counts as an improvement. Guards
                against stopping on numerical noise, and against running on
                for epochs that only move the fourth decimal.
            mode: ``min`` to minimise the metric, ``max`` to maximise it.

        Raises:
            ValueError: If ``patience`` or ``min_delta`` is negative, or if the
                mode is unknown.
        """
        if patience < 0:
            raise ValueError(f"patience must be non negative, got {patience}.")
        if min_delta < 0:
            raise ValueError(f"min_delta must be non negative, got {min_delta}.")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}.")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float = math.inf if mode == "min" else -math.inf
        self.best_step = -1
        self.num_bad_evaluations = 0
        self.should_stop = False

    @property
    def enabled(self) -> bool:
        """Return whether early stopping can ever trigger.

        Returns:
            ``False`` when the patience is zero.
        """
        return self.patience > 0

    def is_improvement(self, value: float) -> bool:
        """Return whether a value beats the best seen so far.

        Args:
            value: The metric of the current evaluation.

        Returns:
            ``True`` when the value improves on the best by at least
            ``min_delta``.
        """
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def update(self, value: float, *, step: int = -1) -> bool:
        """Record one evaluation.

        Args:
            value: The metric of the current evaluation.
            step: Optional identifier of the evaluation, stored alongside the
                best value so that the caller knows which checkpoint won.

        Returns:
            ``True`` when the value is an improvement, which the caller uses to
            decide whether to write the best checkpoint.
        """
        if self.is_improvement(value):
            self.best = value
            self.best_step = step
            self.num_bad_evaluations = 0
            return True

        self.num_bad_evaluations += 1
        if self.enabled and self.num_bad_evaluations >= self.patience:
            self.should_stop = True
        return False

    def state_dict(self) -> dict[str, Any]:
        """Return the state to store in a checkpoint.

        Returns:
            A JSON serialisable mapping.
        """
        return {
            "best": self.best,
            "best_step": self.best_step,
            "num_bad_evaluations": self.num_bad_evaluations,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the state read from a checkpoint.

        Without this, a resumed run would forget its best value and could stop
        several evaluations later than an uninterrupted run, which would make
        the two produce different models.

        Args:
            state: Mapping produced by :meth:`state_dict`.
        """
        self.best = float(state["best"])
        self.best_step = int(state["best_step"])
        self.num_bad_evaluations = int(state["num_bad_evaluations"])
        self.should_stop = bool(state["should_stop"])
