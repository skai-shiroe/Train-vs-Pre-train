"""Unit tests of the join between the declared experiments and the runs.

The property under test is the one a table depends on: every declared
experiment produces exactly one row, whether or not it ever ran.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.experiments.record import (
    STATUS_NOT_RUN,
    STATUS_OK,
    STATUS_PARTIAL,
    STATUS_STALE_CONFIG,
    RunRecord,
    run_directory,
    write_record,
)
from src.experiments.registry import collect, rows_for_study

pytestmark = pytest.mark.unit


def declare(directory: Path, name: str, *, studies: list[str], percentage: int | None = 10) -> Path:
    """Write one experiment file and return its path."""
    payload: dict[str, Any] = {
        "experiment": {"name": name, "seed": 42, "studies": studies},
        "model": {"type": "random_init", "baseline": "t5"},
    }
    if percentage is None:
        payload["model"] = {"type": "pretrained", "mode": "zero_shot"}
    else:
        payload["dataset"] = {"percentage": percentage}
        payload["training"] = {"epochs": 1}

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def record(
    name: str,
    status: str = STATUS_OK,
    score: float = 0.3,
    *,
    config: dict[str, Any] | None = None,
) -> RunRecord:
    """Return a record carrying one ROUGE-L score."""
    return RunRecord(
        experiment=name,
        status=status,
        config=config or {},
        dataset={"train_examples": 2000},
        model={"model": "scratch", "mode": "trained"},
        hardware={},
        evaluation={
            "split": "test",
            "split_size": 100,
            "generation": {"num_beams": 4},
            "rouge_config": {"seed": 42},
            "report": {
                "size": 100,
                "rouge": {
                    "scores": {"rougeL": {"precision": 0.3, "recall": 0.3, "fmeasure": score}},
                    "intervals": {"rougeL": {"low": score - 0.01, "high": score + 0.01}},
                },
                "empty_predictions": 0,
            },
        },
    )


def test_every_declared_experiment_becomes_one_row(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "b_run", studies=["dataset_size"])
    declare(configs, "a_run", studies=["dataset_size"])

    rows = collect(configs, results)

    assert [row.name for row in rows] == ["a_run", "b_run"]


def test_an_experiment_that_never_ran_keeps_its_row(tmp_path: Path) -> None:
    # A shorter table would read as a finished study.
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "scratch_10", studies=["dataset_size"])

    row = collect(configs, results)[0]

    assert row.status == STATUS_NOT_RUN
    assert row.measured is False
    assert row.rouge("rougeL") is None
    assert row.interval("rougeL") is None


def test_a_measured_run_is_joined_to_its_declaration(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "scratch_10", studies=["dataset_size"])
    write_record(record("scratch_10"), run_directory(results, "scratch_10"))

    row = collect(configs, results)[0]

    assert row.status == STATUS_OK
    assert row.measured is True
    assert row.rouge("rougeL") == pytest.approx(0.3)
    assert row.config.dataset.percentage == 10


def test_a_record_from_an_obsolete_configuration_cannot_supply_a_score(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "scratch_10", studies=["dataset_size"])
    write_record(
        record("scratch_10", config={"model": {"type": "obsolete"}}),
        run_directory(results, "scratch_10"),
    )

    row = collect(configs, results)[0]

    assert row.status == STATUS_STALE_CONFIG
    assert row.measured is False
    assert row.rouge("rougeL") is None
    assert row.interval("rougeL") is None


def test_a_partial_run_is_visible_and_unusable(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "scratch_10", studies=["dataset_size"])
    write_record(
        record("scratch_10", STATUS_PARTIAL),
        run_directory(results, "scratch_10", partial=True),
    )

    row = collect(configs, results)[0]

    assert row.status == STATUS_PARTIAL
    assert row.measured is False
    assert row.rouge("rougeL") is None


def test_a_run_without_a_declaration_is_ignored(tmp_path: Path) -> None:
    # The plan is what the tables describe. A directory left behind by a
    # deleted experiment file is not part of it.
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "scratch_10", studies=["dataset_size"])
    write_record(record("orphan"), run_directory(results, "orphan"))

    assert [row.name for row in collect(configs, results)] == ["scratch_10"]


def test_a_study_selects_the_experiments_that_declare_it(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "scratch_100", studies=["dataset_size"])
    declare(configs, "scratch_10", studies=["dataset_size"])

    rows = collect(configs, results)

    assert [row.name for row in rows_for_study(rows, "dataset_size")] == [
        "scratch_10",
        "scratch_100",
    ]
