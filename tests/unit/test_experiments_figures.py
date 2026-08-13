"""Unit tests of the figures of section 18.

The drawing itself is thin; what is worth testing is what reaches the canvas.
A figure is the most quoted and least verifiable output of the campaign, so the
properties below are the ones that would let an image claim something the
records do not say: an unmeasured experiment becoming a point, an interval
turning into a bound, the zero shot baseline landing on the curve, and runs
scored differently ending up on one axis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.experiments.config import PRETRAINED_FINE_TUNED, SCRATCH
from src.experiments.figures import (
    MODEL_COMPARISON_FIGURE,
    PERFORMANCE_FIGURE,
    TRAINING_LOSS_FIGURE,
    VALIDATION_LOSS_FIGURE,
    build,
    curve,
    epoch_series,
    main,
    measured,
    missing_note,
    run_styles,
    zero_shot,
)
from src.experiments.record import (
    STATUS_FAILED,
    STATUS_NOT_RUN,
    STATUS_OK,
    STATUS_PARTIAL,
    RunRecord,
    run_directory,
    write_record,
)
from src.experiments.registry import collect

pytestmark = pytest.mark.unit


def write_payload(directory: Path, name: str, payload: dict[str, Any]) -> Path:
    """Write one experiment file verbatim and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def declare(directory: Path, name: str, *, percentage: int = 10, **overrides: Any) -> Path:
    """Write one from scratch experiment file and return its path."""
    payload: dict[str, Any] = {
        "experiment": {"name": name, "seed": 42, "studies": ["dataset_size"]},
        "dataset": {"percentage": percentage},
        "model": {"type": "scratch", "d_model": 32, "num_heads": 2},
        "training": {"epochs": 1},
    }
    payload.update(overrides)
    return write_payload(directory, name, payload)


def declare_fine_tuned(directory: Path, name: str, *, percentage: int = 10) -> Path:
    """Write one fine tuned baseline experiment file and return its path."""
    return write_payload(
        directory,
        name,
        {
            "experiment": {"name": name, "seed": 42, "studies": ["dataset_size"]},
            "dataset": {"percentage": percentage},
            "model": {"type": "pretrained", "mode": "fine_tuned"},
            "training": {"epochs": 1},
        },
    )


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
    interval: bool = True,
    epochs: int = 2,
) -> RunRecord:
    """Return a record carrying one score per ROUGE variant."""
    scores = {
        variant: {"precision": 0.3, "recall": 0.3, "fmeasure": score + offset}
        for variant, offset in (("rouge1", 0.1), ("rouge2", -0.2), ("rougeL", 0.0))
    }
    intervals = (
        {
            variant: {"low": score + offset - 0.01, "high": score + offset + 0.02}
            for variant, offset in (("rouge1", 0.1), ("rouge2", -0.2), ("rougeL", 0.0))
        }
        if interval
        else {}
    )
    return RunRecord(
        experiment=name,
        status=status,
        config={},
        dataset={"train_examples": 2000},
        model={"model": "scratch", "mode": "trained", "parameters": "1234"},
        hardware={},
        training={
            "best_epoch": 1,
            "best_validation_loss": 3.5,
            "global_step": 100,
            "duration_seconds": 60.0,
            "epochs": [
                {
                    "epoch": index,
                    "train_loss": 5.0 - index,
                    "validation_loss": 6.0 - index,
                    "learning_rate": 0.0003,
                    "duration_seconds": 30.0,
                }
                for index in range(epochs)
            ],
        },
        evaluation={
            "split": "test",
            "split_size": 100,
            "generation": {"num_beams": beams},
            "rouge_config": {"seed": 42},
            "duration_seconds": 5.0,
            "report": {
                "size": 100,
                "rouge": {"scores": scores, "intervals": intervals},
                "empty_predictions": 0,
                "prediction_words_mean": 18.0,
                "reference_words_mean": 21.0,
            },
        },
    )


def zero_shot_record(name: str, *, score: float = 0.2) -> RunRecord:
    """Return a record for a run that never trained."""
    built = record(name, score=score)
    return RunRecord(
        experiment=built.experiment,
        status=built.status,
        config=built.config,
        dataset={},
        model=built.model,
        hardware={},
        training=None,
        evaluation=built.evaluation,
    )


# ---------------------------------------------------------------------------
# What becomes a point
# ---------------------------------------------------------------------------


def test_only_a_complete_run_becomes_a_point(tmp_path: Path) -> None:
    # A partial run carries no reportable score. Drawing it next to complete
    # runs would put a rehearsal on the curve with no way to see it.
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "full_run", percentage=50)
    declare(configs, "half_run", percentage=100)
    write_record(record("full_run"), run_directory(results, "full_run"))
    write_record(
        record("half_run", status=STATUS_PARTIAL),
        run_directory(results, "half_run", partial=True),
    )

    percentages, scores, _ = curve(collect(configs, results), SCRATCH)

    assert percentages == [50]
    assert len(scores) == 1


def test_an_unrun_experiment_is_named_rather_than_dropped(tmp_path: Path) -> None:
    # An image gets separated from its table the moment it is pasted into a
    # slide. One that hides what is missing reads as a finished study.
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "measured_run")
    declare(configs, "never_ran")
    declare(configs, "broke")
    write_record(record("measured_run"), run_directory(results, "measured_run"))
    write_record(record("broke", status=STATUS_FAILED), run_directory(results, "broke"))

    note = missing_note(collect(configs, results))

    assert note is not None
    assert f"never_ran ({STATUS_NOT_RUN})" in note
    assert f"broke ({STATUS_FAILED})" in note
    assert "measured_run" not in note


def test_a_fully_measured_campaign_carries_no_footnote(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "one_run")
    write_record(record("one_run"), run_directory(results, "one_run"))

    assert missing_note(collect(configs, results)) is None


def test_the_zero_shot_baseline_is_never_a_point_on_the_curve(tmp_path: Path) -> None:
    # Its weights do not depend on the training corpus, so any proportion it
    # was drawn at would claim a measurement nobody took.
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "trained_run", percentage=100)
    declare_zero_shot(configs, "baseline_run")
    write_record(record("trained_run"), run_directory(results, "trained_run"))
    write_record(zero_shot_record("baseline_run"), run_directory(results, "baseline_run"))

    rows = collect(configs, results)
    percentages, _, _ = curve(rows, SCRATCH)
    reference = zero_shot(rows)

    assert percentages == [100]
    assert reference is not None
    assert reference[0] == pytest.approx(0.2)


def test_the_two_families_are_read_apart(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "scratch_run", percentage=10)
    declare_fine_tuned(configs, "tuned_run", percentage=50)
    write_record(record("scratch_run", score=0.1), run_directory(results, "scratch_run"))
    write_record(record("tuned_run", score=0.4), run_directory(results, "tuned_run"))

    rows = collect(configs, results)

    assert curve(rows, SCRATCH)[0] == [10]
    assert curve(rows, PRETRAINED_FINE_TUNED)[0] == [50]


def test_the_points_are_ordered_by_corpus_size(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    for name, percentage in (("c_run", 100), ("a_run", 10), ("b_run", 50)):
        declare(configs, name, percentage=percentage)
        write_record(record(name), run_directory(results, name))

    assert curve(collect(configs, results), SCRATCH)[0] == [10, 50, 100]


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


def test_the_error_bars_are_distances_not_bounds(tmp_path: Path) -> None:
    # matplotlib reads yerr as lengths. Passing the bounds themselves would
    # draw a bar running from zero to the upper bound, which is a much smaller
    # looking uncertainty than the one measured.
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "one_run", percentage=10)
    write_record(record("one_run", score=0.3), run_directory(results, "one_run"))

    _, scores, errors = curve(collect(configs, results), SCRATCH)

    assert scores == [pytest.approx(0.3)]
    assert errors[0] == [pytest.approx(0.01)]
    assert errors[1] == [pytest.approx(0.02)]


def test_a_score_without_an_interval_keeps_its_point(tmp_path: Path) -> None:
    # The score is measured; the interval is what is missing. Dropping the
    # point would hide a measurement because of an absent annotation.
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "one_run", percentage=10)
    write_record(record("one_run", interval=False), run_directory(results, "one_run"))

    percentages, _, errors = curve(collect(configs, results), SCRATCH)

    assert percentages == [10]
    assert errors == [[0.0], [0.0]]


# ---------------------------------------------------------------------------
# Loss curves
# ---------------------------------------------------------------------------


def test_a_run_that_never_trained_has_no_loss_curve(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare_zero_shot(configs, "baseline_run")
    write_record(zero_shot_record("baseline_run"), run_directory(results, "baseline_run"))

    (row,) = collect(configs, results)

    assert epoch_series(row, "train_loss") == ([], [])
    assert epoch_series(row, "validation_loss") == ([], [])


def test_the_epochs_are_counted_from_one(tmp_path: Path) -> None:
    # The record counts them from zero, a reader of a figure does not.
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "one_run")
    write_record(record("one_run", epochs=3), run_directory(results, "one_run"))

    (row,) = collect(configs, results)
    epochs, values = epoch_series(row, "validation_loss")

    assert epochs == [1, 2, 3]
    assert values == [6.0, 5.0, 4.0]


def test_two_runs_of_one_family_are_told_apart(tmp_path: Path) -> None:
    # One colour per family would render four from scratch runs as a single
    # orange band with a legend nobody can use.
    configs, results = tmp_path / "configs", tmp_path / "results"
    for name in ("a_run", "b_run", "c_run"):
        declare(configs, name)
        write_record(record(name), run_directory(results, name))

    styles = run_styles(collect(configs, results))

    assert len({str(colour) for colour, _ in styles.values()}) == 3
    assert len({marker for _, marker in styles.values()}) == 3


def test_the_same_records_draw_the_same_styles(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    for name in ("a_run", "b_run"):
        declare(configs, name)
        write_record(record(name), run_directory(results, name))

    rows = collect(configs, results)

    assert run_styles(rows) == run_styles(list(reversed(rows)))


# ---------------------------------------------------------------------------
# Building the four files
# ---------------------------------------------------------------------------


def test_the_four_figures_are_written(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "scratch_run", percentage=100)
    declare_fine_tuned(configs, "tuned_run", percentage=100)
    declare_zero_shot(configs, "baseline_run")
    write_record(record("scratch_run", score=0.16), run_directory(results, "scratch_run"))
    write_record(record("tuned_run", score=0.23), run_directory(results, "tuned_run"))
    write_record(zero_shot_record("baseline_run"), run_directory(results, "baseline_run"))

    written = build(experiments_dir=configs, results_dir=results, output_dir=tmp_path / "figures")

    assert [path.name for path in written] == [
        PERFORMANCE_FIGURE,
        TRAINING_LOSS_FIGURE,
        VALIDATION_LOSS_FIGURE,
        MODEL_COMPARISON_FIGURE,
    ]
    for path in written:
        assert path.stat().st_size > 0


def test_the_figures_are_written_when_nothing_was_measured(tmp_path: Path) -> None:
    # An empty pair of axes under a title reads as a measurement of zero. The
    # files are still written, and they say on themselves why they are empty.
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "never_ran")
    results.mkdir(parents=True, exist_ok=True)

    written = build(experiments_dir=configs, results_dir=results, output_dir=tmp_path / "figures")

    assert len(written) == 4
    for path in written:
        assert path.stat().st_size > 0


def test_runs_measured_differently_are_refused(tmp_path: Path) -> None:
    # The rule the tables apply. A curve makes a gap look like a result even
    # harder than a column does.
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "a_run", percentage=10)
    declare(configs, "b_run", percentage=50)
    write_record(record("a_run", beams=4), run_directory(results, "a_run"))
    write_record(record("b_run", beams=1), run_directory(results, "b_run"))

    with pytest.raises(ValueError, match="not measured the same way"):
        build(experiments_dir=configs, results_dir=results, output_dir=tmp_path / "figures")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def test_main_writes_and_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "one_run")
    declare(configs, "never_ran")
    write_record(record("one_run"), run_directory(results, "one_run"))

    code = main(
        [
            "--experiments",
            str(configs),
            "--results",
            str(results),
            "--output",
            str(tmp_path / "figures"),
        ]
    )

    captured = capsys.readouterr().out
    assert code == 0
    assert "measured         1 of 2" in captured
    assert "never_ran" in captured
    assert len(list((tmp_path / "figures").glob("*.png"))) == 4


def test_main_reports_a_refusal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "a_run", percentage=10)
    declare(configs, "b_run", percentage=50)
    write_record(record("a_run", beams=4), run_directory(results, "a_run"))
    write_record(record("b_run", beams=1), run_directory(results, "b_run"))

    code = main(
        [
            "--experiments",
            str(configs),
            "--results",
            str(results),
            "--output",
            str(tmp_path / "figures"),
        ]
    )

    assert code == 1
    assert "not measured the same way" in capsys.readouterr().err


def test_measured_keeps_only_complete_runs(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "ok_run")
    declare(configs, "partial_run")
    write_record(record("ok_run"), run_directory(results, "ok_run"))
    write_record(
        record("partial_run", status=STATUS_PARTIAL),
        run_directory(results, "partial_run", partial=True),
    )

    assert [row.name for row in measured(collect(configs, results))] == ["ok_run"]
