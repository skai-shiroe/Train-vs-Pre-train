"""Unit tests for checkpoint writing, rotation and reloading."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch
from torch import nn, optim

from src.training.checkpoint import (
    BEST_NAME,
    PAYLOAD_VERSION,
    CheckpointManager,
    build_payload,
    load_checkpoint,
    restore,
    save_checkpoint,
)
from src.training.early_stopping import EarlyStopping
from src.training.scheduler import build_scheduler


def build_run() -> tuple[nn.Module, optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Return a tiny model with its optimiser and scheduler."""
    model = nn.Linear(4, 4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimizer, "linear", warmup_steps=2, total_steps=10)
    return model, optimizer, scheduler


def payload_for(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    epoch: int = 0,
    step: int = 1,
    stopper: EarlyStopping | None = None,
) -> dict[str, object]:
    """Return a payload for a run at a given position."""
    return build_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=stopper,
        epoch=epoch,
        global_step=step,
        best_validation_loss=1.5,
        metadata={"experiment": "unit"},
    )


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_weights_survive_a_round_trip(tmp_path: Path) -> None:
    model, optimizer, scheduler = build_run()
    save_checkpoint(tmp_path / "c.pt", payload_for(model, optimizer, scheduler))

    reloaded = nn.Linear(4, 4)
    restore(load_checkpoint(tmp_path / "c.pt"), model=reloaded)

    assert torch.equal(reloaded.weight, model.weight)


@pytest.mark.unit
def test_the_optimiser_moments_survive_a_round_trip(tmp_path: Path) -> None:
    model, optimizer, scheduler = build_run()
    model(torch.randn(2, 4)).sum().backward()
    optimizer.step()
    save_checkpoint(tmp_path / "c.pt", payload_for(model, optimizer, scheduler))

    fresh_model = nn.Linear(4, 4)
    fresh_optimizer = optim.AdamW(fresh_model.parameters(), lr=1e-3)
    restore(load_checkpoint(tmp_path / "c.pt"), model=fresh_model, optimizer=fresh_optimizer)

    stored = optimizer.state_dict()["state"]
    loaded = fresh_optimizer.state_dict()["state"]
    assert torch.equal(stored[0]["exp_avg"], loaded[0]["exp_avg"])


@pytest.mark.unit
def test_the_schedule_position_survives_a_round_trip(tmp_path: Path) -> None:
    model, optimizer, scheduler = build_run()
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    save_checkpoint(tmp_path / "c.pt", payload_for(model, optimizer, scheduler))

    fresh_model, fresh_optimizer, fresh_scheduler = build_run()
    restore(
        load_checkpoint(tmp_path / "c.pt"),
        model=fresh_model,
        optimizer=fresh_optimizer,
        scheduler=fresh_scheduler,
    )

    assert fresh_scheduler.get_last_lr() == pytest.approx(scheduler.get_last_lr())


@pytest.mark.unit
def test_the_position_of_the_run_is_reported(tmp_path: Path) -> None:
    model, optimizer, scheduler = build_run()
    save_checkpoint(tmp_path / "c.pt", payload_for(model, optimizer, scheduler, epoch=3, step=120))

    resumed = restore(load_checkpoint(tmp_path / "c.pt"), model=nn.Linear(4, 4))

    assert resumed.epoch == 3
    assert resumed.global_step == 120
    assert resumed.best_validation_loss == 1.5


@pytest.mark.unit
def test_the_early_stopping_counters_survive_a_round_trip(tmp_path: Path) -> None:
    model, optimizer, scheduler = build_run()
    stopper = EarlyStopping(3)
    stopper.update(1.0, step=0)
    stopper.update(1.4, step=1)
    save_checkpoint(tmp_path / "c.pt", payload_for(model, optimizer, scheduler, stopper=stopper))

    reloaded = EarlyStopping(3)
    restore(load_checkpoint(tmp_path / "c.pt"), model=nn.Linear(4, 4), early_stopping=reloaded)

    assert reloaded.best == 1.0
    assert reloaded.num_bad_evaluations == 1


@pytest.mark.unit
def test_the_metadata_is_carried_along(tmp_path: Path) -> None:
    model, optimizer, scheduler = build_run()
    save_checkpoint(tmp_path / "c.pt", payload_for(model, optimizer, scheduler))

    assert load_checkpoint(tmp_path / "c.pt")["metadata"] == {"experiment": "unit"}


# ---------------------------------------------------------------------------
# Random generators
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_generators_are_restored_so_a_resume_continues(tmp_path: Path) -> None:
    # Restoring the weights alone would replay a different dropout mask and a
    # different shuffling than an uninterrupted run.
    model, optimizer, scheduler = build_run()
    save_checkpoint(tmp_path / "c.pt", payload_for(model, optimizer, scheduler))
    expected = (random.random(), torch.randn(3).tolist())

    random.seed(999)
    torch.manual_seed(999)
    restore(load_checkpoint(tmp_path / "c.pt"), model=nn.Linear(4, 4))

    assert (random.random(), torch.randn(3).tolist()) == expected


@pytest.mark.unit
def test_the_generators_stay_untouched_when_loading_for_inference(tmp_path: Path) -> None:
    model, optimizer, scheduler = build_run()
    save_checkpoint(tmp_path / "c.pt", payload_for(model, optimizer, scheduler))

    random.seed(999)
    expected = random.Random(999).random()
    restore(load_checkpoint(tmp_path / "c.pt"), model=nn.Linear(4, 4), restore_rng=False)

    assert random.random() == expected


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_missing_checkpoint_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint(tmp_path / "absent.pt")


@pytest.mark.unit
def test_a_foreign_payload_version_is_refused(tmp_path: Path) -> None:
    model, optimizer, scheduler = build_run()
    payload = payload_for(model, optimizer, scheduler)
    payload["version"] = PAYLOAD_VERSION + 1
    save_checkpoint(tmp_path / "c.pt", payload)

    with pytest.raises(ValueError, match="payload version"):
        load_checkpoint(tmp_path / "c.pt")


@pytest.mark.unit
def test_no_temporary_file_survives_a_write(tmp_path: Path) -> None:
    model, optimizer, scheduler = build_run()

    save_checkpoint(tmp_path / "c.pt", payload_for(model, optimizer, scheduler))

    assert [path.name for path in tmp_path.iterdir()] == ["c.pt"]


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_non_positive_retention_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keep_last must be strictly positive"):
        CheckpointManager(tmp_path, keep_last=0)


@pytest.mark.unit
def test_only_the_last_checkpoints_are_kept(tmp_path: Path) -> None:
    model, optimizer, scheduler = build_run()
    manager = CheckpointManager(tmp_path, keep_last=2)

    for step in (10, 20, 30, 40):
        manager.save(payload_for(model, optimizer, scheduler, step=step), step=step)

    assert [path.name for path in manager.periodic_checkpoints()] == [
        "checkpoint-step-00000030.pt",
        "checkpoint-step-00000040.pt",
    ]


@pytest.mark.unit
def test_the_best_checkpoint_escapes_the_rotation(tmp_path: Path) -> None:
    model, optimizer, scheduler = build_run()
    manager = CheckpointManager(tmp_path, keep_last=1)

    manager.save(payload_for(model, optimizer, scheduler, step=10), step=10, is_best=True)
    for step in (20, 30):
        manager.save(payload_for(model, optimizer, scheduler, step=step), step=step)

    assert manager.best_path.is_file()
    assert load_checkpoint(manager.best_path)["global_step"] == 10


@pytest.mark.unit
def test_the_checkpoints_sort_by_step_not_by_string(tmp_path: Path) -> None:
    model, optimizer, scheduler = build_run()
    manager = CheckpointManager(tmp_path, keep_last=10)

    for step in (9, 100, 11):
        manager.save(payload_for(model, optimizer, scheduler, step=step), step=step)

    assert [load_checkpoint(path)["global_step"] for path in manager.periodic_checkpoints()] == [
        9,
        11,
        100,
    ]


@pytest.mark.unit
def test_the_resume_point_is_the_most_recent_checkpoint(tmp_path: Path) -> None:
    model, optimizer, scheduler = build_run()
    manager = CheckpointManager(tmp_path, keep_last=3)

    for step in (10, 20):
        manager.save(payload_for(model, optimizer, scheduler, step=step), step=step)

    latest = manager.latest()
    assert latest is not None
    assert latest.name == "checkpoint-step-00000020.pt"


@pytest.mark.unit
def test_the_resume_point_falls_back_on_the_best_checkpoint(tmp_path: Path) -> None:
    model, optimizer, scheduler = build_run()
    manager = CheckpointManager(tmp_path, keep_last=1)
    save_checkpoint(manager.best_path, payload_for(model, optimizer, scheduler))

    latest = manager.latest()
    assert latest is not None
    assert latest.name == BEST_NAME


@pytest.mark.unit
def test_an_empty_directory_offers_no_resume_point(tmp_path: Path) -> None:
    assert CheckpointManager(tmp_path / "absent").latest() is None
    assert CheckpointManager(tmp_path / "absent").periodic_checkpoints() == []


@pytest.mark.unit
def test_unrelated_files_are_left_alone(tmp_path: Path) -> None:
    model, optimizer, scheduler = build_run()
    (tmp_path / "notes.txt").write_text("keep me", encoding="utf-8")
    manager = CheckpointManager(tmp_path, keep_last=1)

    for step in (10, 20):
        manager.save(payload_for(model, optimizer, scheduler, step=step), step=step)

    assert (tmp_path / "notes.txt").is_file()
