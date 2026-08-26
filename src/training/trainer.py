"""The training loop.

One class owns the optimisation and nothing else. Observation goes to the
callbacks, persistence to the checkpoint manager, and the stopping decision to
the early stopper, so the loop below stays readable enough to be checked
against what it claims to do.

What the loop handles, following section 15 of the specification:

```text
train loss            token weighted mean over the epoch
validation loss       same measure, on the validation split
gradient clipping     global norm, measured before clipping and reported
checkpoint            periodic, rotated, plus the best one kept apart
early stopping        on the validation loss
scheduler             stepped per optimisation step, never per epoch
mixed precision       on CUDA only, with a gradient scaler
resume training       weights, moments, scheduler, scaler, counters, generators
```

The loss is token weighted rather than batch weighted. Batches hold a variable
number of target tokens, so a plain mean over batches would silently weight a
short summary as much as a long one, and the training curve would not be
comparable with the validation curve.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader

from src.data.example import Example
from src.data.tokenize import IGNORE_INDEX, EncodedBatch
from src.training.callbacks import Callback, CallbackList, default_callbacks
from src.training.checkpoint import CheckpointManager, build_payload, load_checkpoint, restore
from src.training.config import TrainingConfig
from src.training.early_stopping import EarlyStopping
from src.training.optimizer import build_optimizer
from src.training.sampler import LengthGroupedSampler
from src.training.scheduler import build_scheduler
from src.training.state import EpochMetrics, TrainingResult, TrainingState
from src.utils.device import resolve_device, supports_mixed_precision

#: Signature of a function turning one batch into a scalar loss. Making it a
#: parameter is what lets the same trainer drive the hand written Transformer
#: and the two T5 runs, whose forward signatures differ.
BatchLossFn = Callable[[nn.Module, EncodedBatch], torch.Tensor]


def make_scratch_batch_loss(label_smoothing: float = 0.0) -> BatchLossFn:
    """Build the loss function of the from scratch Transformer.

    The loss is computed here rather than inside the model so that label
    smoothing stays a training hyperparameter. The model keeps its own plain
    cross entropy, which the architecture tests rely on.

    Args:
        label_smoothing: Mass taken from the gold token and spread over the
            vocabulary.

    Returns:
        A function mapping a model and a batch to a scalar loss.
    """

    def batch_loss(model: nn.Module, batch: EncodedBatch) -> torch.Tensor:
        output = model(batch.input_ids, target_ids=batch.target_ids)
        logits = cast(torch.Tensor, output.logits)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch.labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            label_smoothing=label_smoothing,
        )

    return batch_loss


def make_seq2seq_batch_loss(label_smoothing: float = 0.0) -> BatchLossFn:
    """Build the loss function of a sequence-to-sequence Transformers model.

    The loss is computed here so label smoothing remains a training
    hyperparameter shared by both T5 initialisations.

    Args:
        label_smoothing: Mass taken from the gold token and spread over the
            vocabulary.

    Returns:
        A function mapping a model and a batch to a scalar loss.
    """

    def batch_loss(model: nn.Module, batch: EncodedBatch) -> torch.Tensor:
        output = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
        )
        if label_smoothing == 0.0:
            return cast(torch.Tensor, output.loss)
        logits = cast(torch.Tensor, output.logits)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch.labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            label_smoothing=label_smoothing,
        )

    return batch_loss


def count_target_tokens(batch: EncodedBatch) -> int:
    """Return the number of label positions the loss actually covers.

    Args:
        batch: The encoded batch.

    Returns:
        The number of positions that are not padding.
    """
    return int((batch.labels != IGNORE_INDEX).sum().item())


class Trainer:
    """Train a sequence to sequence model on the frozen working corpus."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        *,
        train_loader: DataLoader[Example],
        validation_loader: DataLoader[Example],
        output_dir: Path,
        batch_loss: BatchLossFn | None = None,
        callbacks: Iterable[Callback] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Build the trainer.

        Args:
            model: The model to train.
            config: Hyperparameters of the run.
            train_loader: Loader over the training split.
            validation_loader: Loader over the validation split.
            output_dir: Directory receiving the checkpoints.
            batch_loss: Function computing the loss of one batch. Defaults to
                the shared sequence-to-sequence loss.
            callbacks: Observers of the run. Defaults to a logging callback and
                a history callback.
            metadata: Free form mapping stored in every checkpoint, typically
                the dataset version and the experiment name.

        Raises:
            ValueError: If the training loader is empty.
        """
        if len(train_loader) == 0:
            raise ValueError("The training loader yields no batch.")

        self.config = config
        self.device = resolve_device(config.device)
        self.model = model.to(self.device)
        self.output_dir = output_dir
        self.metadata = metadata or {}

        self._train_loader = train_loader
        self._validation_loader = validation_loader
        self._batch_loss = batch_loss or make_seq2seq_batch_loss(config.label_smoothing)
        self._callbacks = CallbackList(
            callbacks
            if callbacks is not None
            else default_callbacks(log_every_steps=config.log_every_steps)
        )

        self.steps_per_epoch = self._steps_per_epoch()
        self.total_steps = self._total_steps()

        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_scheduler(
            self.optimizer,
            config.scheduler,
            warmup_steps=config.warmup_steps(self.total_steps),
            total_steps=self.total_steps,
        )
        self.use_amp = config.mixed_precision and supports_mixed_precision(self.device)
        self.scaler = GradScaler(device=self.device.type, enabled=self.use_amp)
        self.early_stopping = EarlyStopping(
            config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode="min",
        )
        self.checkpoints = CheckpointManager(output_dir, keep_last=config.keep_last_checkpoints)
        self.state = TrainingState(epochs=config.epochs, total_steps=self.total_steps)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _steps_per_epoch(self) -> int:
        """Return the number of optimisation steps in one epoch.

        Returns:
            The batch count divided by the accumulation depth, rounded up: the
            trailing incomplete accumulation window is still applied.
        """
        batches = len(self._train_loader)
        accumulation = self.config.gradient_accumulation_steps
        return (batches + accumulation - 1) // accumulation

    def _total_steps(self) -> int:
        """Return the number of optimisation steps the whole run plans.

        Returns:
            The planned step count, capped by ``max_steps`` when it is set.
        """
        planned = self.steps_per_epoch * self.config.epochs
        if self.config.max_steps is not None:
            return min(planned, self.config.max_steps)
        return planned

    def _set_epoch(self, epoch: int) -> None:
        """Tell a length grouped sampler which epoch it is replaying.

        Args:
            epoch: Zero based epoch index.
        """
        sampler = self._train_loader.batch_sampler
        if isinstance(sampler, LengthGroupedSampler):
            sampler.set_epoch(epoch)

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def train(self, *, resume_from: Path | None = None) -> TrainingResult:
        """Run the training loop.

        Args:
            resume_from: Checkpoint to continue from. When it is a directory,
                the most recent checkpoint it holds is used.

        Returns:
            What the run produced.
        """
        start_epoch = 0
        if resume_from is not None:
            start_epoch = self._resume(resume_from)

        self._callbacks.on_train_begin(self.state)
        started = time.perf_counter()
        history: list[EpochMetrics] = []
        best_checkpoint: Path | None = None

        for epoch in range(start_epoch, self.config.epochs):
            self.state.epoch = epoch
            self._callbacks.on_epoch_begin(self.state)

            epoch_started = time.perf_counter()
            steps_before = self.state.global_step
            train_loss = self._run_epoch()
            validation_loss = self.evaluate(self._validation_loader)

            improved = self.early_stopping.update(validation_loss, step=epoch)
            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                learning_rate=self.state.learning_rate,
                steps=self.state.global_step - steps_before,
                duration_seconds=time.perf_counter() - epoch_started,
                improved=improved,
            )
            history.append(metrics)

            self.checkpoints.save(
                self._payload(epoch=epoch),
                step=self.state.global_step,
                is_best=improved,
            )
            if improved:
                best_checkpoint = self.checkpoints.best_path

            self._callbacks.on_epoch_end(self.state, metrics)

            if self.early_stopping.should_stop:
                self.state.stopped_early = True
                break
            if self._step_budget_exhausted():
                break

        result = TrainingResult(
            epochs=tuple(history),
            best_epoch=self.early_stopping.best_step,
            best_validation_loss=self.early_stopping.best,
            stopped_early=self.state.stopped_early,
            global_step=self.state.global_step,
            duration_seconds=time.perf_counter() - started,
            best_checkpoint=best_checkpoint,
        )
        self._callbacks.on_train_end(self.state, result)
        return result

    def _run_epoch(self) -> float:
        """Run one pass over the training split.

        Returns:
            The token weighted mean training loss of the epoch.
        """
        self.model.train()
        self._set_epoch(self.state.epoch)
        self.optimizer.zero_grad(set_to_none=True)

        loss_sum = 0.0
        token_count = 0
        accumulated = 0
        batches = len(self._train_loader)
        accumulation = self.config.gradient_accumulation_steps

        for index, raw_batch in enumerate(self._train_loader):
            batch = cast(EncodedBatch, raw_batch).to(self.device)

            with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                loss = self._batch_loss(self.model, batch)

            # The gradient of a mean over an accumulation window is the mean of
            # the gradients, so the loss is divided before the backward pass.
            torch.autograd.backward(self.scaler.scale(loss / accumulation))
            accumulated += 1

            tokens = count_target_tokens(batch)
            loss_sum += float(loss.detach().item()) * tokens
            token_count += tokens

            is_last_batch = index == batches - 1
            if accumulated == accumulation or is_last_batch:
                self._optimiser_step()
                accumulated = 0
                self.state.last_loss = float(loss.detach().item())
                self.state.running_loss = loss_sum / max(1, token_count)
                self._callbacks.on_step_end(self.state)

                if self._step_budget_exhausted():
                    break

        return loss_sum / max(1, token_count)

    def _optimiser_step(self) -> None:
        """Clip the gradients, step the optimiser and advance the schedule."""
        # Unscaling first is what makes the clipping threshold mean the same
        # thing with and without mixed precision.
        self.scaler.unscale_(self.optimizer)
        norm = clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Read the rate before advancing the schedule. Reading it after would
        # report the rate of the next step, and the last record of a decaying
        # schedule would be the zero the run never actually used.
        self.state.learning_rate = float(self.optimizer.param_groups[0]["lr"])
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        self.state.global_step += 1
        self.state.grad_norm = float(norm)

    def _step_budget_exhausted(self) -> bool:
        """Return whether the optional step cap has been reached.

        Returns:
            ``True`` when ``max_steps`` is set and already reached.
        """
        return self.config.max_steps is not None and self.state.global_step >= self.config.max_steps

    @torch.no_grad()
    def evaluate(self, loader: DataLoader[Example]) -> float:
        """Measure the loss on a split without touching the weights.

        Args:
            loader: Loader over the split to measure.

        Returns:
            The token weighted mean loss. Zero when the loader is empty.
        """
        self.model.eval()
        loss_sum = 0.0
        token_count = 0

        for raw_batch in loader:
            batch = cast(EncodedBatch, raw_batch).to(self.device)
            with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                loss = self._batch_loss(self.model, batch)

            tokens = count_target_tokens(batch)
            loss_sum += float(loss.item()) * tokens
            token_count += tokens

        return loss_sum / token_count if token_count else 0.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _payload(self, *, epoch: int) -> dict[str, Any]:
        """Assemble the checkpoint payload of the current position.

        Args:
            epoch: Index of the epoch just completed.

        Returns:
            The payload.
        """
        metadata = dict(self.metadata)
        metadata.update(
            {
                "device": self.device.type,
                "mixed_precision": self.use_amp,
                "total_steps": self.total_steps,
            }
        )
        return build_payload(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.use_amp else None,
            early_stopping=self.early_stopping,
            epoch=epoch,
            global_step=self.state.global_step,
            best_validation_loss=self.early_stopping.best,
            metadata=metadata,
        )

    def _resume(self, source: Path) -> int:
        """Reload a checkpoint and return the epoch to start from.

        Args:
            source: Checkpoint file, or directory holding checkpoints.

        Returns:
            Index of the first epoch left to run.

        Raises:
            FileNotFoundError: If the directory holds no checkpoint.
        """
        path = source
        if source.is_dir():
            found = CheckpointManager(source, keep_last=self.config.keep_last_checkpoints).latest()
            if found is None:
                raise FileNotFoundError(f"No checkpoint to resume from in {source}.")
            path = found

        payload = load_checkpoint(path, map_location=self.device)
        resumed = restore(
            payload,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.use_amp else None,
            early_stopping=self.early_stopping,
        )
        self.state.global_step = resumed.global_step
        return resumed.epoch + 1


def train_model(
    model: nn.Module,
    config: TrainingConfig,
    *,
    train_loader: DataLoader[Example],
    validation_loader: DataLoader[Example],
    output_dir: Path,
    batch_loss: BatchLossFn | None = None,
    callbacks: Sequence[Callback] | None = None,
    metadata: dict[str, Any] | None = None,
    resume_from: Path | None = None,
) -> TrainingResult:
    """Build a trainer and run it, in one call.

    Args:
        model: The model to train.
        config: Hyperparameters of the run.
        train_loader: Loader over the training split.
        validation_loader: Loader over the validation split.
        output_dir: Directory receiving the checkpoints.
        batch_loss: Function computing the loss of one batch.
        callbacks: Observers of the run.
        metadata: Free form mapping stored in every checkpoint.
        resume_from: Checkpoint or directory to continue from.

    Returns:
        What the run produced.
    """
    trainer = Trainer(
        model,
        config,
        train_loader=train_loader,
        validation_loader=validation_loader,
        output_dir=output_dir,
        batch_loss=batch_loss,
        callbacks=callbacks,
        metadata=metadata,
    )
    return trainer.train(resume_from=resume_from)
