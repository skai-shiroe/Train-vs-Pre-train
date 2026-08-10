"""Unit tests of the run record.

The record is the only thing standing between a run and a published table, so
what is checked here is mostly what it refuses to hand over: a score from a run
that was partial, failed, or never happened.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.experiments.record import (
    PARTIAL_SUFFIX,
    RUN_RECORD_FILE,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_PARTIAL,
    RunRecord,
    find_record,
    read_record,
    run_directory,
    write_record,
)

pytestmark = pytest.mark.unit


def evaluation_block(**overrides: Any) -> dict[str, Any]:
    """Return an evaluation block shaped like the evaluator writes it."""
    block: dict[str, Any] = {
        "split": "test",
        "split_size": 100,
        "partial": False,
        "generation": {"max_new_tokens": 64, "num_beams": 4},
        "rouge_config": {"use_stemmer": True, "seed": 42},
        "report": {
            "size": 100,
            "rouge": {
                "scores": {
                    "rouge1": {"precision": 0.4, "recall": 0.3, "fmeasure": 0.34},
                    "rougeL": {"precision": 0.3, "recall": 0.2, "fmeasure": 0.25},
                },
                "intervals": {"rougeL": {"low": 0.23, "high": 0.27, "confidence": 0.95}},
            },
            "empty_predictions": 2,
        },
        "duration_seconds": 12.5,
    }
    block.update(overrides)
    return block


def make_record(status: str = STATUS_OK, **overrides: Any) -> RunRecord:
    """Return a record with an evaluation, unless overridden."""
    fields: dict[str, Any] = {
        "experiment": "scratch_100",
        "status": status,
        "config": {"experiment": {"name": "scratch_100"}},
        "dataset": {"percentage": 100, "train_examples": 20000},
        "model": {"model": "scratch", "mode": "trained"},
        "hardware": {"torch_version": "2.7.0"},
        "evaluation": evaluation_block(),
    }
    fields.update(overrides)
    return RunRecord(**fields)


# ---------------------------------------------------------------------------
# What a record hands over
# ---------------------------------------------------------------------------


def test_a_complete_run_reports_its_scores() -> None:
    record = make_record()

    assert record.measured is True
    assert record.rouge("rougeL") == pytest.approx(0.25)
    assert record.interval("rougeL") == (pytest.approx(0.23), pytest.approx(0.27))


@pytest.mark.parametrize("status", [STATUS_PARTIAL, STATUS_FAILED])
def test_a_run_that_is_not_complete_reports_nothing(status: str) -> None:
    # None rather than zero: a zero is a measurement, and a run that did not
    # finish did not score zero.
    record = make_record(status)

    assert record.measured is False
    assert record.rouge("rougeL") is None
    assert record.interval("rougeL") is None


def test_a_failed_run_carries_its_error() -> None:
    record = make_record(STATUS_FAILED, evaluation=None, error="RuntimeError: out of memory")

    assert record.report == {}
    assert record.error == "RuntimeError: out of memory"


def test_an_unmeasured_variant_reports_nothing() -> None:
    record = make_record()

    assert record.rouge("rouge2") is None
    assert record.interval("rouge1") is None


# ---------------------------------------------------------------------------
# Comparability
# ---------------------------------------------------------------------------


def test_two_runs_measured_alike_share_a_key() -> None:
    assert make_record().measurement_key == make_record().measurement_key


def test_a_different_beam_width_changes_the_key() -> None:
    other = make_record(
        evaluation=evaluation_block(generation={"max_new_tokens": 64, "num_beams": 1})
    )

    assert other.measurement_key != make_record().measurement_key


def test_a_different_split_changes_the_key() -> None:
    other = make_record(evaluation=evaluation_block(split="validation"))

    assert other.measurement_key != make_record().measurement_key


def test_the_key_does_not_depend_on_the_key_order() -> None:
    # The key is rendered as sorted JSON, so a record written by an older
    # version with the same values still compares equal.
    reordered = make_record(
        evaluation=evaluation_block(generation={"num_beams": 4, "max_new_tokens": 64})
    )

    assert reordered.measurement_key == make_record().measurement_key


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_a_record_survives_a_round_trip(tmp_path: Path) -> None:
    written = write_record(make_record(), tmp_path / "run")

    reloaded = read_record(tmp_path / "run")

    assert written.name == RUN_RECORD_FILE
    assert reloaded is not None
    assert reloaded.to_dict() == make_record().to_dict()


def test_a_directory_without_a_record_reads_as_nothing(tmp_path: Path) -> None:
    assert read_record(tmp_path) is None


def test_a_truncated_record_is_a_defect_not_a_missing_run(tmp_path: Path) -> None:
    (tmp_path / RUN_RECORD_FILE).write_text(json.dumps({"status": "OK"}), encoding="utf-8")

    with pytest.raises(KeyError):
        read_record(tmp_path)


def test_a_partial_run_writes_beside_the_complete_slot(tmp_path: Path) -> None:
    complete = run_directory(tmp_path, "scratch_100")
    partial = run_directory(tmp_path, "scratch_100", partial=True)

    assert complete.name == "scratch_100"
    assert partial.name == f"scratch_100{PARTIAL_SUFFIX}"
    assert complete != partial


def test_a_complete_run_is_preferred_over_a_partial_one(tmp_path: Path) -> None:
    write_record(make_record(STATUS_PARTIAL), run_directory(tmp_path, "scratch_100", partial=True))
    write_record(make_record(), run_directory(tmp_path, "scratch_100"))

    found = find_record(tmp_path, "scratch_100")

    assert found is not None
    assert found.status == STATUS_OK


def test_a_partial_run_is_still_found_when_it_is_all_there_is(tmp_path: Path) -> None:
    # It must not read as NOT_RUN: it ran, it is simply not a result.
    write_record(make_record(STATUS_PARTIAL), run_directory(tmp_path, "scratch_100", partial=True))

    found = find_record(tmp_path, "scratch_100")

    assert found is not None
    assert found.status == STATUS_PARTIAL


def test_an_experiment_never_run_is_found_nowhere(tmp_path: Path) -> None:
    assert find_record(tmp_path, "scratch_100") is None
