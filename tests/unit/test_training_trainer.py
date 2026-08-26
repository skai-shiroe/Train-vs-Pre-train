"""Unit tests for the training loop.

The loop is exercised on a tiny T5 and a fake tokenisation, so the
tests stay fast and offline. What is checked here is the mechanics: step
accounting, clipping, checkpointing, early stopping and resume. Whether the
architecture can actually learn is checked by the overfitting test of the
Transformer suite.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import T5Config, T5ForConditionalGeneration

from src.data.example import Example
from src.data.tokenize import IGNORE_INDEX, EncodedBatch
from src.training import trainer as trainer_module
from src.training.callbacks import Callback, HistoryCallback
from src.training.checkpoint import load_checkpoint
from src.training.config import TrainingConfig
from src.training.sampler import LengthGroupedSampler
from src.training.state import EpochMetrics, TrainingState
from src.training.trainer import Trainer, count_target_tokens, make_seq2seq_batch_loss, train_model
from src.utils.seed import set_seed

VOCAB_SIZE = 32

LoaderFactory = Callable[..., DataLoader[Example]]


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    set_seed(1234)


def build_model() -> T5ForConditionalGeneration:
    """Return a T5 small enough for a unit test."""
    return T5ForConditionalGeneration(
        T5Config(
            vocab_size=VOCAB_SIZE,
            d_model=16,
            d_kv=8,
            num_layers=1,
            num_decoder_layers=1,
            d_ff=32,
            num_heads=2,
            dropout_rate=0.0,
            pad_token_id=0,
            eos_token_id=1,
            decoder_start_token_id=0,
        )
    )


def build_config(**overrides: object) -> TrainingConfig:
    """Return a fast training configuration."""
    defaults: dict[str, object] = {
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "warmup_ratio": 0.0,
        "device": "cpu",
        "mixed_precision": False,
        "early_stopping_patience": 0,
        "log_every_steps": 1000,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)  # type: ignore[arg-type]


def constant_loss(values: list[float]) -> Callable[[nn.Module, EncodedBatch], torch.Tensor]:
    """Return a loss function replaying a fixed sequence of values.

    The value is attached to a parameter so that the backward pass still builds
    a graph, which keeps the optimiser and the scaler on their normal path.
    """
    calls = {"index": 0}

    def batch_loss(model: nn.Module, batch: EncodedBatch) -> torch.Tensor:
        anchor = next(model.parameters()).sum() * 0.0
        value = values[min(calls["index"], len(values) - 1)]
        calls["index"] += 1
        return anchor + value

    return batch_loss


class SpyCallback(Callback):
    """Count the events the loop emits."""

    def __init__(self) -> None:
        self.steps = 0
        self.epochs: list[EpochMetrics] = []
        self.ended = False

    def on_step_end(self, state: TrainingState) -> None:
        self.steps += 1

    def on_epoch_end(self, state: TrainingState, metrics: EpochMetrics) -> None:
        self.epochs.append(metrics)

    def on_train_end(self, state: TrainingState, result: object) -> None:
        self.ended = True


def build_trainer(
    make_loader: LoaderFactory,
    tmp_path: Path,
    *,
    examples: int = 16,
    callbacks: list[Callback] | None = None,
    **config_overrides: object,
) -> Trainer:
    """Return a trainer wired on fake batches."""
    config = build_config(**config_overrides)
    return Trainer(
        build_model(),
        config,
        train_loader=make_loader(examples, batch_size=config.batch_size, vocab_size=VOCAB_SIZE),
        validation_loader=make_loader(8, batch_size=config.batch_size, vocab_size=VOCAB_SIZE),
        output_dir=tmp_path / "runs",
        callbacks=callbacks if callbacks is not None else [],
    )


# ---------------------------------------------------------------------------
# Construction and planning
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_an_empty_training_loader_is_refused(make_loader: LoaderFactory, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="yields no batch"):
        Trainer(
            build_model(),
            build_config(),
            train_loader=make_loader(0, batch_size=4, vocab_size=VOCAB_SIZE),
            validation_loader=make_loader(4, batch_size=4, vocab_size=VOCAB_SIZE),
            output_dir=tmp_path,
        )


@pytest.mark.unit
def test_the_planned_steps_cover_every_batch_of_every_epoch(
    make_loader: LoaderFactory, tmp_path: Path
) -> None:
    trainer = build_trainer(make_loader, tmp_path, examples=16, epochs=3)

    assert trainer.steps_per_epoch == 4
    assert trainer.total_steps == 12


@pytest.mark.unit
def test_accumulation_divides_the_step_count(make_loader: LoaderFactory, tmp_path: Path) -> None:
    trainer = build_trainer(
        make_loader, tmp_path, examples=16, epochs=1, gradient_accumulation_steps=2
    )

    assert trainer.steps_per_epoch == 2


@pytest.mark.unit
def test_a_trailing_incomplete_window_still_steps(
    make_loader: LoaderFactory, tmp_path: Path
) -> None:
    # Five batches with an accumulation of two: the fifth must be applied, not
    # silently dropped along with its gradient.
    trainer = build_trainer(
        make_loader, tmp_path, examples=20, epochs=1, gradient_accumulation_steps=2
    )

    assert trainer.steps_per_epoch == 3

    trainer.train()

    assert trainer.state.global_step == 3


@pytest.mark.unit
def test_the_step_cap_shortens_the_plan(make_loader: LoaderFactory, tmp_path: Path) -> None:
    trainer = build_trainer(make_loader, tmp_path, examples=16, epochs=10, max_steps=5)

    assert trainer.total_steps == 5

    trainer.train()

    assert trainer.state.global_step == 5


@pytest.mark.unit
def test_mixed_precision_stays_off_on_cpu(make_loader: LoaderFactory, tmp_path: Path) -> None:
    trainer = build_trainer(make_loader, tmp_path, mixed_precision=True, device="cpu")

    assert trainer.use_amp is False


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_run_reports_one_record_per_epoch(make_loader: LoaderFactory, tmp_path: Path) -> None:
    spy = SpyCallback()
    trainer = build_trainer(make_loader, tmp_path, epochs=2, callbacks=[spy])

    result = trainer.train()

    assert len(result.epochs) == 2
    assert [metrics.epoch for metrics in spy.epochs] == [0, 1]
    assert spy.steps == trainer.total_steps
    assert spy.ended is True


@pytest.mark.unit
def test_training_lowers_the_loss(make_loader: LoaderFactory, tmp_path: Path) -> None:
    trainer = build_trainer(make_loader, tmp_path, epochs=6, learning_rate=3e-3)

    result = trainer.train()

    assert result.epochs[-1].train_loss < result.epochs[0].train_loss


@pytest.mark.unit
def test_the_learning_rate_is_reported(make_loader: LoaderFactory, tmp_path: Path) -> None:
    trainer = build_trainer(make_loader, tmp_path, epochs=1, learning_rate=1e-3)

    result = trainer.train()

    assert 0.0 < result.epochs[0].learning_rate <= 1e-3


@pytest.mark.unit
def test_the_gradient_norm_is_measured_before_clipping(
    make_loader: LoaderFactory, tmp_path: Path
) -> None:
    trainer = build_trainer(make_loader, tmp_path, epochs=1, max_grad_norm=1e-4)

    trainer.train()

    # A norm reported at the threshold would mean it was measured after the
    # clipping, which would hide exactly the signal it exists to give.
    assert trainer.state.grad_norm > 1e-4


@pytest.mark.unit
def test_the_gradients_are_clipped_before_the_optimiser_sees_them(
    make_loader: LoaderFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[tuple[float, float]] = []
    original = trainer_module.clip_grad_norm_

    def spy(parameters: object, max_norm: float, *args: object, **kwargs: object) -> torch.Tensor:
        collected = list(parameters)  # type: ignore[call-overload]
        norm = original(collected, max_norm, *args, **kwargs)  # type: ignore[arg-type]
        after = torch.sqrt(
            sum(
                (parameter.grad.detach() ** 2).sum()
                for parameter in collected
                if parameter.grad is not None
            )
        )
        observed.append((float(max_norm), float(after)))
        return norm

    monkeypatch.setattr(trainer_module, "clip_grad_norm_", spy)
    trainer = build_trainer(make_loader, tmp_path, epochs=1, max_grad_norm=0.5)

    trainer.train()

    assert len(observed) == trainer.total_steps
    assert all(threshold == 0.5 for threshold, _ in observed)
    assert all(norm <= 0.5 + 1e-4 for _, norm in observed)


@pytest.mark.unit
def test_evaluation_leaves_the_weights_alone(make_loader: LoaderFactory, tmp_path: Path) -> None:
    trainer = build_trainer(make_loader, tmp_path)
    before = [parameter.detach().clone() for parameter in trainer.model.parameters()]

    trainer.evaluate(trainer._validation_loader)

    assert all(
        torch.equal(before_value, after)
        for before_value, after in zip(before, trainer.model.parameters(), strict=True)
    )


@pytest.mark.unit
def test_an_empty_evaluation_split_reports_zero(make_loader: LoaderFactory, tmp_path: Path) -> None:
    trainer = build_trainer(make_loader, tmp_path)

    assert trainer.evaluate(make_loader(0, batch_size=4, vocab_size=VOCAB_SIZE)) == 0.0


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_run_that_stops_improving_ends_early(make_loader: LoaderFactory, tmp_path: Path) -> None:
    trainer = build_trainer(
        make_loader,
        tmp_path,
        epochs=10,
        early_stopping_patience=1,
    )
    trainer._batch_loss = constant_loss([1.0])

    result = trainer.train()

    # First epoch sets the best value, second does not improve, patience spent.
    assert result.stopped_early is True
    assert len(result.epochs) == 2


@pytest.mark.unit
def test_a_run_that_keeps_improving_uses_its_whole_budget(
    make_loader: LoaderFactory, tmp_path: Path
) -> None:
    trainer = build_trainer(make_loader, tmp_path, epochs=3, early_stopping_patience=1)

    result = trainer.train()

    assert result.stopped_early is False
    assert len(result.epochs) == 3


@pytest.mark.unit
def test_the_best_epoch_is_reported(make_loader: LoaderFactory, tmp_path: Path) -> None:
    trainer = build_trainer(make_loader, tmp_path, epochs=3, early_stopping_patience=0)

    result = trainer.train()

    losses = [metrics.validation_loss for metrics in result.epochs]
    assert result.best_epoch == losses.index(min(losses))
    assert result.best_validation_loss == pytest.approx(min(losses))


# ---------------------------------------------------------------------------
# Checkpoints and resume
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_each_epoch_writes_a_checkpoint(make_loader: LoaderFactory, tmp_path: Path) -> None:
    trainer = build_trainer(make_loader, tmp_path, epochs=2, keep_last_checkpoints=5)

    result = trainer.train()

    assert len(trainer.checkpoints.periodic_checkpoints()) == 2
    assert result.best_checkpoint is not None
    assert result.best_checkpoint.is_file()


@pytest.mark.unit
def test_the_checkpoint_carries_the_run_metadata(
    make_loader: LoaderFactory, tmp_path: Path
) -> None:
    config = build_config(epochs=1)
    trainer = Trainer(
        build_model(),
        config,
        train_loader=make_loader(16, batch_size=4, vocab_size=VOCAB_SIZE),
        validation_loader=make_loader(8, batch_size=4, vocab_size=VOCAB_SIZE),
        output_dir=tmp_path / "runs",
        callbacks=[],
        metadata={"dataset_version": "abc123"},
    )

    trainer.train()

    metadata = load_checkpoint(trainer.checkpoints.best_path)["metadata"]
    assert metadata["dataset_version"] == "abc123"
    assert metadata["device"] == "cpu"
    assert metadata["mixed_precision"] is False


@pytest.mark.unit
def test_a_resumed_run_picks_up_where_it_stopped(
    make_loader: LoaderFactory, tmp_path: Path
) -> None:
    first = build_trainer(make_loader, tmp_path, epochs=1, keep_last_checkpoints=5)
    first.train()

    second = build_trainer(make_loader, tmp_path, epochs=3, keep_last_checkpoints=5)
    result = second.train(resume_from=second.output_dir)

    # Two epochs left out of three, and the step counter continued.
    assert [metrics.epoch for metrics in result.epochs] == [1, 2]
    assert result.global_step == first.state.global_step * 3


@pytest.mark.unit
def test_a_resume_restores_the_weights(make_loader: LoaderFactory, tmp_path: Path) -> None:
    first = build_trainer(make_loader, tmp_path, epochs=1)
    first.train()

    second = build_trainer(make_loader, tmp_path, epochs=1)
    second.train(resume_from=second.checkpoints.best_path)

    assert all(
        torch.equal(before, after)
        for before, after in zip(first.model.parameters(), second.model.parameters(), strict=True)
    )


@pytest.mark.unit
def test_a_resume_without_any_checkpoint_is_reported(
    make_loader: LoaderFactory, tmp_path: Path
) -> None:
    trainer = build_trainer(make_loader, tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="No checkpoint to resume from"):
        trainer.train(resume_from=empty)


# ---------------------------------------------------------------------------
# Length grouping
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_sampler_is_told_which_epoch_it_replays(
    make_loader: LoaderFactory, make_examples: Callable[[int], list[Example]], tmp_path: Path
) -> None:
    seen: list[int] = []

    class RecordingSampler(LengthGroupedSampler):
        def set_epoch(self, epoch: int) -> None:
            seen.append(epoch)
            super().set_epoch(epoch)

    lengths = [4 + index % 7 for index in range(16)]
    sampler = RecordingSampler(lengths, 4, seed=42)
    config = build_config(epochs=3)
    trainer = Trainer(
        build_model(),
        config,
        train_loader=make_loader(16, vocab_size=VOCAB_SIZE, batch_sampler=sampler),
        validation_loader=make_loader(8, batch_size=4, vocab_size=VOCAB_SIZE),
        output_dir=tmp_path / "runs",
        callbacks=[],
    )

    trainer.train()

    # Without this, every epoch would replay the exact same batches.
    assert seen == [0, 1, 2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_padding_positions_are_left_out_of_the_token_count() -> None:
    labels = torch.tensor([[5, 6, IGNORE_INDEX], [7, IGNORE_INDEX, IGNORE_INDEX]])
    batch = EncodedBatch(
        input_ids=torch.zeros(2, 3, dtype=torch.long),
        attention_mask=torch.ones(2, 3, dtype=torch.long),
        labels=labels,
        target_ids=torch.zeros(2, 3, dtype=torch.long),
    )

    assert count_target_tokens(batch) == 3


@pytest.mark.unit
def test_label_smoothing_raises_the_loss_of_a_confident_model() -> None:
    model = build_model()
    batch = EncodedBatch(
        input_ids=torch.randint(2, VOCAB_SIZE, (2, 6)),
        attention_mask=torch.ones(2, 6, dtype=torch.long),
        labels=torch.randint(2, VOCAB_SIZE, (2, 4)),
        target_ids=torch.randint(2, VOCAB_SIZE, (2, 4)),
    )

    plain = make_seq2seq_batch_loss(0.0)(model, batch).detach()
    smoothed = make_seq2seq_batch_loss(0.2)(model, batch).detach()

    assert float(smoothed) != pytest.approx(float(plain))


@pytest.mark.unit
def test_the_helper_builds_and_runs_a_trainer(make_loader: LoaderFactory, tmp_path: Path) -> None:
    history = HistoryCallback()

    result = train_model(
        build_model(),
        build_config(epochs=1),
        train_loader=make_loader(16, batch_size=4, vocab_size=VOCAB_SIZE),
        validation_loader=make_loader(8, batch_size=4, vocab_size=VOCAB_SIZE),
        output_dir=tmp_path / "runs",
        callbacks=[history],
    )

    assert result.global_step == 4
    assert len(history.step_losses) == 4
