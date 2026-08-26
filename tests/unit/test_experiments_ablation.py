"""Unit tests of the ablation tables.

Three properties are worth more than the rest, because a table that gets them
wrong still looks like a result: an unrun experiment must keep its row, an
empty cell must not read as a zero, and runs measured differently must not end
up in one table.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.evaluation.evaluator import QUALITATIVE_FILE
from src.experiments.ablation import (
    DATASET_SIZE_CSV,
    EXPERIMENTS_CSV,
    QUALITATIVE_JSON,
    aggregate,
    cell,
    check_comparable,
    collect_qualitative,
    format_status,
    main,
    sort_dataset_size,
)
from src.experiments.config import ExperimentConfig
from src.experiments.record import (
    STATUS_FAILED,
    STATUS_NOT_RUN,
    STATUS_OK,
    STATUS_PARTIAL,
    RunRecord,
    run_directory,
    write_record,
)
from src.experiments.registry import ExperimentRow, collect

pytestmark = pytest.mark.unit


def write_payload(directory: Path, name: str, payload: dict[str, Any]) -> Path:
    """Write one experiment file verbatim and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def declare(directory: Path, name: str, **overrides: Any) -> Path:
    """Write one from scratch experiment file and return its path."""
    payload: dict[str, Any] = {
        "experiment": {"name": name, "seed": 42, "studies": ["dataset_size"]},
        "dataset": {"percentage": 10},
        "model": {"type": "random_init", "baseline": "t5"},
        "training": {"epochs": 1},
    }
    payload.update(overrides)
    return write_payload(directory, name, payload)


def declare_zero_shot(directory: Path, name: str) -> Path:
    """Write one zero shot experiment file and return its path."""
    return write_payload(
        directory,
        name,
        {
            "experiment": {"name": name, "seed": 42, "studies": ["dataset_size"]},
            "model": {"type": "pretrained", "mode": "zero_shot"},
        },
    )


def record(
    name: str,
    *,
    status: str = STATUS_OK,
    score: float = 0.3,
    beams: int = 4,
    split: str = "test",
) -> RunRecord:
    """Return a record carrying one ROUGE-L score."""
    return RunRecord(
        experiment=name,
        status=status,
        config={},
        dataset={"train_examples": 2000},
        model={"model": "scratch", "mode": "trained", "parameters": "1234"},
        hardware={},
        training={
            "best_epoch": 2,
            "best_validation_loss": 3.5,
            "global_step": 100,
            "duration_seconds": 60.0,
        },
        evaluation={
            "split": split,
            "split_size": 100,
            "generation": {"num_beams": beams},
            "rouge_config": {"seed": 42},
            "duration_seconds": 5.0,
            "report": {
                "size": 100,
                "rouge": {
                    "scores": {
                        "rouge1": {"precision": 0.4, "recall": 0.4, "fmeasure": score + 0.1},
                        "rougeL": {"precision": 0.3, "recall": 0.3, "fmeasure": score},
                    },
                    "intervals": {"rougeL": {"low": score - 0.01, "high": score + 0.01}},
                },
                "empty_predictions": 1,
                "prediction_words_mean": 18.0,
                "reference_words_mean": 21.0,
            },
        },
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a written table back."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


def test_a_missing_measurement_is_empty_not_zero() -> None:
    # A zero is a measurement. Rendering an absent one as zero would put a
    # model that never ran on the curve, at the bottom.
    assert cell(None) == ""
    assert cell(0.0) == "0.0"


def test_a_score_keeps_enough_decimals_to_draw() -> None:
    assert cell(0.123456789) == "0.123457"
    assert cell(12) == "12"
    assert cell("OK") == "OK"


# ---------------------------------------------------------------------------
# Comparability
# ---------------------------------------------------------------------------


def test_runs_measured_alike_pass(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "a_run")
    declare(configs, "b_run")
    write_record(record("a_run"), run_directory(results, "a_run"))
    write_record(record("b_run"), run_directory(results, "b_run"))

    check_comparable(collect(configs, results), "dataset_size")


def test_a_different_beam_width_stops_the_table(tmp_path: Path) -> None:
    # The gap between the two would report the decoding, and nothing on the
    # page would say so.
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "a_run")
    declare(configs, "b_run")
    write_record(record("a_run", beams=4), run_directory(results, "a_run"))
    write_record(record("b_run", beams=1), run_directory(results, "b_run"))

    with pytest.raises(ValueError, match="not measured the same way"):
        check_comparable(collect(configs, results), "dataset_size")


def test_an_unmeasured_run_never_blocks_a_table(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "a_run")
    declare(configs, "b_run")
    write_record(record("a_run"), run_directory(results, "a_run"))
    write_record(
        record("b_run", status=STATUS_PARTIAL, beams=1),
        run_directory(results, "b_run", partial=True),
    )

    check_comparable(collect(configs, results), "dataset_size")


# ---------------------------------------------------------------------------
# The registry table
# ---------------------------------------------------------------------------


def test_an_unrun_experiment_keeps_its_row(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "scratch_10")

    aggregate(
        experiments_dir=configs, results_dir=results, output_dir=results, studies=["dataset_size"]
    )
    rows = read_csv(results / EXPERIMENTS_CSV)

    assert len(rows) == 1
    assert rows[0]["experiment"] == "scratch_10"
    assert rows[0]["status"] == STATUS_NOT_RUN
    assert rows[0]["rougeL_f"] == ""
    assert rows[0]["dataset_percentage"] == "10"


def test_a_measured_run_fills_its_columns(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "scratch_10")
    write_record(record("scratch_10"), run_directory(results, "scratch_10"))

    aggregate(
        experiments_dir=configs, results_dir=results, output_dir=results, studies=["dataset_size"]
    )
    row = read_csv(results / EXPERIMENTS_CSV)[0]

    assert row["status"] == STATUS_OK
    assert row["rougeL_f"] == "0.3"
    assert row["rougeL_low"] == "0.29"
    assert row["rouge1_f"] == "0.4"
    assert row["train_examples"] == "2000"
    assert row["parameters"] == "1234"
    assert row["best_validation_loss"] == "3.5"
    assert row["empty_predictions"] == "1"


def test_a_partial_run_keeps_its_row_without_its_score(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "scratch_10")
    write_record(
        record("scratch_10", status=STATUS_PARTIAL),
        run_directory(results, "scratch_10", partial=True),
    )

    aggregate(
        experiments_dir=configs, results_dir=results, output_dir=results, studies=["dataset_size"]
    )
    row = read_csv(results / EXPERIMENTS_CSV)[0]

    assert row["status"] == STATUS_PARTIAL
    assert row["rougeL_f"] == ""
    assert row["empty_predictions"] == ""


def test_a_failed_run_keeps_its_row(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "scratch_10")
    write_record(
        RunRecord(
            experiment="scratch_10",
            status=STATUS_FAILED,
            config={},
            dataset={},
            model={},
            hardware={},
            error="RuntimeError: out of memory",
        ),
        run_directory(results, "scratch_10"),
    )

    aggregate(
        experiments_dir=configs, results_dir=results, output_dir=results, studies=["dataset_size"]
    )
    row = read_csv(results / EXPERIMENTS_CSV)[0]

    assert row["status"] == STATUS_FAILED
    assert row["rougeL_f"] == ""


# ---------------------------------------------------------------------------
# The corpus size ablation
# ---------------------------------------------------------------------------


def test_the_zero_shot_reference_has_no_proportion(tmp_path: Path) -> None:
    # It was not trained on nothing, it was not trained. An empty cell says
    # that; a zero would put it at the origin of the curve.
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "scratch_10")
    declare_zero_shot(configs, "zero_shot")

    aggregate(
        experiments_dir=configs, results_dir=results, output_dir=results, studies=["dataset_size"]
    )
    rows = read_csv(results / DATASET_SIZE_CSV)

    zero_shot = [row for row in rows if row["experiment"] == "zero_shot"][0]
    assert zero_shot["dataset_percentage"] == ""
    assert rows[-1]["experiment"] == "zero_shot"


def test_the_curve_is_ordered_by_proportion(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    for name, percentage in (("scratch_100", 100), ("scratch_10", 10), ("scratch_50", 50)):
        declare(configs, name, dataset={"percentage": percentage})

    aggregate(
        experiments_dir=configs, results_dir=results, output_dir=results, studies=["dataset_size"]
    )
    rows = read_csv(results / DATASET_SIZE_CSV)

    assert [row["dataset_percentage"] for row in rows] == ["10", "50", "100"]


def test_the_zero_shot_row_sorts_last_whatever_its_name() -> None:
    # It is a horizontal reference line under the curve, not a point on it.
    payloads = (
        {
            "experiment": {"name": "aaa_zero", "studies": ["dataset_size"]},
            "model": {"type": "pretrained", "mode": "zero_shot"},
        },
        {
            "experiment": {"name": "zzz_scratch", "studies": ["dataset_size"]},
            "dataset": {"percentage": 10},
            "model": {"type": "random_init", "baseline": "t5"},
            "training": {"epochs": 1},
        },
    )
    rows = [
        ExperimentRow(config=ExperimentConfig.model_validate(payload), record=None)
        for payload in payloads
    ]

    assert [row.name for row in sort_dataset_size(rows)] == ["zzz_scratch", "aaa_zero"]


# ---------------------------------------------------------------------------
# Qualitative examples
# ---------------------------------------------------------------------------


def test_the_qualitative_selections_are_gathered_per_experiment(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "scratch_10")
    directory = run_directory(results, "scratch_10")
    write_record(record("scratch_10"), directory)
    (directory / QUALITATIVE_FILE).write_text(
        json.dumps({"best": [{"example_id": "id-1"}]}), encoding="utf-8"
    )

    gathered = collect_qualitative(collect(configs, results), results)

    assert gathered == {"scratch_10": {"best": [{"example_id": "id-1"}]}}


def test_an_unmeasured_run_contributes_no_example(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "scratch_10")

    assert collect_qualitative(collect(configs, results), results) == {}


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_the_command_writes_the_three_result_files(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "scratch_10")

    code = main(["--experiments", str(configs), "--results", str(results), "--study", "all"])

    assert code == 0
    for name in (EXPERIMENTS_CSV, DATASET_SIZE_CSV, QUALITATIVE_JSON):
        assert (results / name).is_file()


def test_one_study_still_refreshes_the_registry(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "scratch_10")

    main(["--experiments", str(configs), "--results", str(results), "--study", "dataset_size"])

    assert (results / EXPERIMENTS_CSV).is_file()
    assert (results / DATASET_SIZE_CSV).is_file()


def test_the_output_directory_can_be_separated(tmp_path: Path) -> None:
    configs, results, output = tmp_path / "configs", tmp_path / "results", tmp_path / "tables"
    declare(configs, "scratch_10")

    main(
        [
            "--experiments",
            str(configs),
            "--results",
            str(results),
            "--output",
            str(output),
            "--study",
            "dataset_size",
        ]
    )

    assert (output / EXPERIMENTS_CSV).is_file()
    assert not (results / EXPERIMENTS_CSV).exists()


def test_the_summary_says_when_nothing_was_measured(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "scratch_10")

    summary = format_status(collect(configs, results))

    assert "measured         0" in summary
    assert "No experiment has been run" in summary


def test_the_summary_reports_the_measured_score(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "scratch_10")
    write_record(record("scratch_10"), run_directory(results, "scratch_10"))

    summary = format_status(collect(configs, results))

    assert "measured         1" in summary
    assert "0.3000" in summary
    assert "No experiment has been run" not in summary
