"""Typed configuration of a training run.

The ``training`` block of an experiment file is validated by this model before
a single tensor is allocated. A run that would fail on a malformed learning
rate must fail immediately, not after the corpus has been tokenised.

The same object is logged to MLflow as the hyperparameters of the run, so the
configuration and the trace can never disagree.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Learning rate schedules supported by :mod:`src.training.scheduler`.
SchedulerName = Literal["linear", "cosine", "inverse_sqrt", "constant"]


class TrainingConfig(BaseModel):
    """Hyperparameters of the training loop.

    Attributes:
        epochs: Number of passes over the training split.
        batch_size: Number of examples per optimisation step, before gradient
            accumulation.
        gradient_accumulation_steps: Number of batches accumulated before the
            optimiser steps. The effective batch size is the product of the
            two, which is how an 8 GB card reaches a larger batch than it can
            hold at once.
        learning_rate: Peak learning rate, reached at the end of the warmup.
        weight_decay: Decoupled weight decay of AdamW. Applied to the matrices
            only, never to the biases and normalisation gains.
        adam_beta1: First moment decay of AdamW.
        adam_beta2: Second moment decay of AdamW.
        adam_epsilon: Term added to the denominator for numerical stability.
        scheduler: Shape of the learning rate schedule.
        warmup_ratio: Share of the total steps spent ramping the learning rate
            up from zero. Expressed as a ratio rather than a step count so that
            the same configuration behaves consistently across the 10, 50 and
            100 percent ablation points, which have different step counts.
        max_grad_norm: Global norm the gradients are clipped to. Clipping is
            what keeps a from scratch Transformer from diverging on its first
            few hundred steps.
        label_smoothing: Mass taken from the gold token and spread over the
            vocabulary. Zero disables it.
        mixed_precision: Whether to run the forward pass in reduced precision.
            Honoured on CUDA only, since it costs more than it saves on CPU.
        early_stopping_patience: Number of evaluations without improvement
            tolerated before stopping. Zero disables early stopping.
        early_stopping_min_delta: Improvement below this threshold counts as no
            improvement at all.
        group_by_length: Whether to batch documents of similar length together.
            See :mod:`src.training.sampler` for the measured effect.
        mega_batch_factor: Number of batches inside one sorting window of the
            length grouped sampler.
        num_workers: Number of DataLoader worker processes.
        keep_last_checkpoints: Number of periodic checkpoints kept on disk. The
            best checkpoint is kept regardless.
        log_every_steps: Interval, in optimisation steps, between two progress
            records.
        max_steps: Optional hard cap on the number of optimisation steps, used
            by the quick mode of ``make reproduce`` to exercise the pipeline
            without producing a reportable result.
        seed: Seed applied before the run.
        device: Requested device, resolved by :func:`src.utils.device.resolve_device`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    epochs: int = Field(default=10, gt=0, description="Number of passes over the training split.")
    batch_size: int = Field(default=8, gt=0, description="Examples per batch.")
    gradient_accumulation_steps: int = Field(
        default=1, gt=0, description="Batches accumulated before an optimiser step."
    )

    learning_rate: float = Field(default=3e-4, gt=0, description="Peak learning rate.")
    weight_decay: float = Field(default=0.01, ge=0, description="Decoupled weight decay of AdamW.")
    adam_beta1: float = Field(default=0.9, gt=0, lt=1, description="First moment decay.")
    adam_beta2: float = Field(default=0.999, gt=0, lt=1, description="Second moment decay.")
    adam_epsilon: float = Field(default=1e-8, gt=0, description="Numerical stability term.")

    scheduler: SchedulerName = Field(default="linear", description="Learning rate schedule.")
    warmup_ratio: float = Field(
        default=0.06, ge=0, lt=1, description="Share of the steps spent warming up."
    )

    max_grad_norm: float = Field(default=1.0, gt=0, description="Gradient clipping threshold.")
    label_smoothing: float = Field(default=0.0, ge=0, lt=1, description="Label smoothing mass.")
    mixed_precision: bool = Field(default=True, description="Use reduced precision on CUDA.")

    early_stopping_patience: int = Field(
        default=3, ge=0, description="Evaluations without improvement tolerated."
    )
    early_stopping_min_delta: float = Field(
        default=0.0, ge=0, description="Smallest improvement that counts."
    )

    group_by_length: bool = Field(default=True, description="Batch similar lengths together.")
    mega_batch_factor: int = Field(default=50, gt=0, description="Batches per sorting window.")
    num_workers: int = Field(default=0, ge=0, description="DataLoader worker processes.")

    keep_last_checkpoints: int = Field(default=2, gt=0, description="Periodic checkpoints kept.")
    log_every_steps: int = Field(default=50, gt=0, description="Steps between two progress logs.")
    max_steps: int | None = Field(default=None, gt=0, description="Optional cap on the steps.")

    seed: int = Field(default=42, ge=0, description="Seed applied before the run.")
    device: str = Field(default="auto", description="Requested device: auto, cpu or cuda.")

    @model_validator(mode="after")
    def _check_device(self) -> TrainingConfig:
        """Reject a device string the resolver cannot honour.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If the device is not one of the supported values.
        """
        from src.utils.device import VALID_DEVICES

        if self.device not in VALID_DEVICES:
            raise ValueError(f"device must be one of {VALID_DEVICES}, got {self.device!r}.")
        return self

    @property
    def effective_batch_size(self) -> int:
        """Return the number of examples behind one optimiser step.

        Returns:
            The batch size multiplied by the accumulation depth.
        """
        return self.batch_size * self.gradient_accumulation_steps

    def warmup_steps(self, total_steps: int) -> int:
        """Return the number of warmup steps for a run of a given length.

        Args:
            total_steps: Total number of optimisation steps planned.

        Returns:
            The number of steps spent ramping the learning rate up. Always at
            least one when the ratio is non zero, so that a short run still
            warms up instead of starting at the peak rate.
        """
        if self.warmup_ratio <= 0.0 or total_steps <= 0:
            return 0
        return max(1, int(total_steps * self.warmup_ratio))
