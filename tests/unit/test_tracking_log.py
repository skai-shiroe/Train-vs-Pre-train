"""Unit tests of the command that pushes existing records to the store.

The property under test is that the command reports the declared plan rather
than only what it managed to send: an experiment that never ran is named, and a
push that failed exits non zero instead of leaving a green command behind.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.experiments.config import load_experiment_config
from src.experiments.record import STATUS_OK, RunRecord, run_directory, write_record
from src.experiments.registry import ExperimentRow, collect
from src.tracking import log as log_module
from src.tracking.log import main, push
from src.tracking.payload import TrackedRun

pytestmark = pytest.mark.unit


class RecordingTracker:
    """Tracker keeping what it was handed."""

    def __init__(self) -> None:
        self.seen: list[TrackedRun] = []

    def log(self, payload: TrackedRun) -> str | None:
        self.seen.append(payload)
        return f"run-{len(self.seen)}"


class BrokenTracker:
    """Tracker standing in for an unreachable store."""

    def log(self, payload: TrackedRun) -> str | None:
        raise ConnectionError("the store is unreachable")


def declare(configs: Path, name: str, percentage: int) -> Path:
    """Write one experiment file and return its path."""
    payload: dict[str, Any] = {
        "experiment": {"name": name, "seed": 42},
        "dataset": {"percentage": percentage},
        "model": {"type": "random_init", "baseline": "t5"},
        "training": {"epochs": 1},
    }
    configs.mkdir(parents=True, exist_ok=True)
    path = configs / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def record(name: str, declaration: Path) -> RunRecord:
    """Return a measured record, carrying the declaration that produced it.

    The configuration is read back from the file rather than repeated here:
    :class:`src.experiments.registry.ExperimentRow` refuses to report a record
    whose model block disagrees with the declaration, and a hand written copy
    would drift out of that agreement at the first default that changes.
    """
    return RunRecord(
        experiment=name,
        status=STATUS_OK,
        config=load_experiment_config(declaration).to_dict(),
        dataset={"config": "configs/data/cnn_dailymail.yaml", "version": "abc"},
        model={"baseline": "t5", "initialization": "random", "mode": "trained"},
        hardware={},
        provenance={"git_commit": "deadbeef"},
        evaluation={
            "split": "test",
            "split_size": 20,
            "generation": {},
            "rouge_config": {},
            "duration_seconds": 1.0,
            "report": {
                "size": 20,
                "empty_predictions": 0,
                "rouge": {"scores": {"rougeL": {"fmeasure": 0.15}}, "intervals": {}},
            },
        },
    )


@pytest.fixture()
def repository(tmp_path: Path) -> dict[str, Path]:
    """Lay out two declared experiments, one of which ran."""
    configs, results = tmp_path / "configs", tmp_path / "results"
    declaration = declare(configs, "scratch_100", 100)
    declare(configs, "scratch_10", 10)
    write_record(record("scratch_100", declaration), run_directory(results, "scratch_100"))
    return {"configs": configs, "results": results}


def arguments(repository: dict[str, Path], *extra: str) -> list[str]:
    """Return the common command line, plus what a test adds."""
    return [
        "--experiments-dir",
        str(repository["configs"]),
        "--results",
        str(repository["results"]),
        *extra,
    ]


def test_an_experiment_that_never_ran_sends_nothing(repository: dict[str, Path]) -> None:
    row = next(
        item
        for item in collect(repository["configs"], repository["results"])
        if item.name == "scratch_10"
    )

    assert isinstance(row, ExperimentRow)
    assert push(row, RecordingTracker(), repository["results"]) is None


def test_the_plan_is_reported_and_only_the_runs_are_sent(
    repository: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tracker = RecordingTracker()
    monkeypatch.setattr(log_module, "build_tracker", lambda **kwargs: tracker)

    code = main(arguments(repository, "--all"))

    output = capsys.readouterr().out
    assert code == 0
    assert [payload.name for payload in tracker.seen] == ["scratch_100"]
    assert "scratch_10               NOT_RUN  nothing to send" in output
    assert "scratch_100              OK       run-1" in output


def test_one_experiment_can_be_selected(
    repository: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = RecordingTracker()
    monkeypatch.setattr(log_module, "build_tracker", lambda **kwargs: tracker)

    code = main(arguments(repository, "--experiment", "scratch_100"))

    assert code == 0
    assert [payload.name for payload in tracker.seen] == ["scratch_100"]


def test_an_undeclared_experiment_is_an_error(
    repository: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(arguments(repository, "--experiment", "scratch_50"))

    assert code == 1
    assert "No experiment named 'scratch_50' is declared." in capsys.readouterr().err


def test_a_store_that_refuses_makes_the_command_fail(
    repository: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The runner may degrade silently, this command may not: pushing is the
    # only thing it does.
    monkeypatch.setattr(log_module, "build_tracker", lambda **kwargs: BrokenTracker())

    code = main(arguments(repository, "--all"))

    captured = capsys.readouterr()
    assert code == 1
    assert "scratch_100              OK       not sent" in captured.out
    assert "1 record(s) could not be sent." in captured.err
