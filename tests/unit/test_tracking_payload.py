"""Unit tests of what section 19 sends to the tracking store.

Two properties carry the weight here. Every field section 19 lists must reach
the payload, and no run that is not a complete measurement may contribute a
score. The second is the one a tracking user interface would otherwise break:
its default view is a table sorted by metric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.experiments.record import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_PARTIAL,
    RunRecord,
    run_directory,
    write_record,
)
from src.tracking.payload import (
    TRACKED_ARTIFACTS,
    build_metrics,
    build_params,
    build_payload,
    build_tags,
    collect_artifacts,
    flatten,
    run_name,
)

pytestmark = pytest.mark.unit


def evaluation_block(score: float = 0.15) -> dict[str, Any]:
    """Return the evaluation block of a measured run."""
    return {
        "split": "test",
        "split_size": 1000,
        "partial": False,
        "generation": {"num_beams": 4, "max_new_tokens": 64},
        "rouge_config": {"seed": 42, "bootstrap_samples": 1000},
        "duration_seconds": 2.0,
        "report": {
            "size": 1000,
            "empty_predictions": 3,
            "prediction_words": {"mean": 12.0},
            "reference_words": {"mean": 20.0},
            "rouge": {
                "scores": {
                    "rouge1": {"fmeasure": 0.30},
                    "rouge2": {"fmeasure": 0.08},
                    "rougeL": {"fmeasure": score},
                },
                "intervals": {"rougeL": {"low": score - 0.01, "high": score + 0.01}},
            },
        },
    }


def training_block() -> dict[str, Any]:
    """Return the training block of a run that optimised something."""
    return {
        "best_epoch": 1,
        "best_validation_loss": 3.2,
        "stopped_early": False,
        "global_step": 250,
        "duration_seconds": 600.0,
        "best_checkpoint": "runs/scratch_100/best.pt",
        "epochs": [
            {
                "epoch": 0,
                "train_loss": 4.0,
                "validation_loss": 3.5,
                "learning_rate": 0.0003,
                "steps": 125,
                "duration_seconds": 300.0,
                "improved": True,
            },
            {
                "epoch": 1,
                "train_loss": 3.4,
                "validation_loss": 3.2,
                "learning_rate": 0.0001,
                "steps": 125,
                "duration_seconds": 300.0,
                "improved": True,
            },
        ],
        "history": {},
    }


def record(
    *,
    status: str = STATUS_OK,
    zero_shot: bool = False,
    trains: bool = True,
    measured: bool = True,
) -> RunRecord:
    """Return a run record shaped like the ones the runner writes."""
    model_block: dict[str, Any] = (
        {"type": "pretrained", "baseline": "t5", "mode": "zero_shot", "revision": "abc123"}
        if zero_shot
        else {"type": "scratch", "d_model": 256, "encoder_layers": 4}
    )
    return RunRecord(
        experiment="scratch_100",
        status=status,
        config={
            "experiment": {"name": "scratch_100", "seed": 42, "studies": ["dataset_size"]},
            "dataset": {
                "config": "configs/data/cnn_dailymail.yaml",
                "percentage": None if zero_shot else 100,
            },
            "model": model_block,
            "training": None if not trains else {"epochs": 3, "max_steps": None},
            "evaluation": {"split": "test", "num_beams": 4},
        },
        dataset={
            "config": "configs/data/cnn_dailymail.yaml",
            "name": "cnn_dailymail",
            "version": "259d8397ce78",  # pragma: allowlist secret
            "percentage": None if zero_shot else 100,
            "train_examples": 0 if zero_shot else 20000,
        },
        model={"model": "scratch", "mode": "trained", "parameters": "44000000"},
        hardware={"torch_version": "2.7.0", "cuda_available": "True", "gpu_name": "RTX"},
        provenance={"git_commit": "deadbeef", "git_branch": "master", "git_dirty": "true"},
        training=training_block() if trains else None,
        evaluation=evaluation_block() if measured else None,
        duration_seconds=650.0,
    )


# ---------------------------------------------------------------------------
# flatten
# ---------------------------------------------------------------------------


def test_a_nested_block_becomes_dotted_names() -> None:
    assert flatten({"a": {"b": 1}}) == {"a.b": "1"}


def test_an_absent_value_is_dropped_rather_than_rendered() -> None:
    # "None" as a parameter value reads as a setting somebody chose.
    assert flatten({"max_steps": None, "epochs": 3}) == {"epochs": "3"}


def test_a_list_is_rendered_on_one_line() -> None:
    assert flatten({"studies": ["a", "b"]}) == {"studies": "a b"}


# ---------------------------------------------------------------------------
# Section 19 coverage
# ---------------------------------------------------------------------------


def test_every_field_of_section_19_reaches_the_payload() -> None:
    payload = build_payload(record())

    assert payload.params["experiment"] == "scratch_100"
    assert payload.params["experiment.seed"] == "42"
    assert payload.params["dataset.version"] == "259d8397ce78"  # pragma: allowlist secret
    assert payload.params["dataset.percentage"] == "100"
    assert payload.params["model.type"] == "scratch"
    assert payload.params["training.epochs"] == "3"
    assert payload.params["checkpoint"] == "runs/scratch_100/best.pt"
    assert payload.metrics["training_duration_seconds"] == pytest.approx(600.0)
    assert payload.metrics["rougeL_f"] == pytest.approx(0.15)
    assert payload.tags["syntra.git_commit"] == "deadbeef"
    assert payload.tags["syntra.hardware.gpu_name"] == "RTX"


def test_no_bleu_is_invented() -> None:
    # Section 2.1 puts BLEU out of scope. A zero would be a fabricated measure.
    payload = build_payload(record())
    everything = {
        **payload.params,
        **payload.tags,
        **{k: str(v) for k, v in payload.metrics.items()},
    }

    assert not [key for key in everything if "bleu" in key.lower()]


def test_the_commit_comes_from_the_record_not_from_the_machine() -> None:
    # A record pushed next week must carry the commit that produced it.
    payload = build_payload(record())

    assert payload.tags["syntra.git_commit"] == "deadbeef"
    assert payload.tags["syntra.git_dirty"] == "true"


# ---------------------------------------------------------------------------
# What a run that is not a measurement may log
# ---------------------------------------------------------------------------


def test_a_partial_run_logs_no_score() -> None:
    metrics = build_metrics(record(status=STATUS_PARTIAL))

    assert "rougeL_f" not in metrics
    assert "scored_examples" not in metrics
    assert metrics["run_duration_seconds"] == pytest.approx(650.0)


def test_a_partial_run_still_logs_what_it_trained() -> None:
    # It did train. Hiding the loss would lose a real observation.
    metrics = build_metrics(record(status=STATUS_PARTIAL))

    assert metrics["best_validation_loss"] == pytest.approx(3.2)


def test_a_failed_run_logs_its_error_and_no_metric_of_quality() -> None:
    failed = RunRecord(
        experiment="scratch_100",
        status=STATUS_FAILED,
        config={"model": {"type": "scratch"}},
        dataset={"config": "configs/data/cnn_dailymail.yaml"},
        model={},
        hardware={},
        provenance={"git_commit": "deadbeef"},
        error="RuntimeError: out of memory",
        duration_seconds=12.0,
    )

    payload = build_payload(failed)

    assert payload.tags["syntra.status"] == STATUS_FAILED
    assert payload.tags["syntra.error"] == "RuntimeError: out of memory"
    assert payload.tags["syntra.measured"] == "false"
    assert set(payload.metrics) == {"run_duration_seconds"}
    assert payload.epochs == ()


def test_a_run_that_is_not_complete_carries_its_status_in_its_name() -> None:
    assert run_name(record()) == "scratch_100"
    assert run_name(record(status=STATUS_PARTIAL)) == "scratch_100:partial"
    assert run_name(record(status=STATUS_FAILED)) == "scratch_100:failed"


# ---------------------------------------------------------------------------
# The zero shot baseline
# ---------------------------------------------------------------------------


def test_the_zero_shot_run_declares_no_corpus_proportion() -> None:
    # Absent, not zero: the model was not trained on nothing, it was not
    # trained. A zero would put it at the origin of the ablation curve.
    params = build_params(record(zero_shot=True, trains=False))

    assert "dataset.percentage" not in params
    assert "config.dataset.percentage" not in params


def test_a_run_that_did_not_train_logs_no_training_metric() -> None:
    metrics = build_metrics(record(zero_shot=True, trains=False))

    assert "training_duration_seconds" not in metrics
    assert "best_validation_loss" not in metrics
    assert metrics["rougeL_f"] == pytest.approx(0.15)


def test_the_mode_of_the_baseline_is_a_tag() -> None:
    tags = build_tags(record(zero_shot=True, trains=False))

    assert tags["syntra.model_type"] == "pretrained"
    assert tags["syntra.model_mode"] == "zero_shot"


# ---------------------------------------------------------------------------
# Curves and artefacts
# ---------------------------------------------------------------------------


def test_the_training_curves_are_logged_epoch_by_epoch() -> None:
    payload = build_payload(record())

    assert len(payload.epochs) == 2
    assert payload.epochs[0]["train_loss"] == pytest.approx(4.0)
    assert payload.epochs[1]["validation_loss"] == pytest.approx(3.2)


def test_only_the_files_that_exist_are_copied(tmp_path: Path) -> None:
    directory = run_directory(tmp_path, "scratch_100")
    write_record(record(), directory)
    (directory / "metrics.json").write_text("{}", encoding="utf-8")

    collected = [path.name for path in collect_artifacts(directory)]

    assert collected == ["run.json", "metrics.json"]


def test_the_predictions_are_never_copied() -> None:
    # One line per scored document, already published as a pipeline artefact,
    # and read by nothing in the tracking store.
    assert "predictions.jsonl" not in TRACKED_ARTIFACTS


def test_a_caller_without_a_directory_copies_nothing() -> None:
    assert collect_artifacts(None) == ()


# ---------------------------------------------------------------------------
# Records that are shaped differently from the ones the runner writes
# ---------------------------------------------------------------------------


def test_a_variant_that_was_not_measured_contributes_no_metric() -> None:
    incomplete = record()
    assert incomplete.evaluation is not None
    del incomplete.evaluation["report"]["rouge"]["scores"]["rouge2"]

    metrics = build_metrics(incomplete)

    assert "rouge2_f" not in metrics
    assert metrics["rougeL_f"] == pytest.approx(0.15)


def test_absent_length_distributions_are_skipped() -> None:
    incomplete = record()
    assert incomplete.evaluation is not None
    incomplete.evaluation["report"]["prediction_words"] = {}
    del incomplete.evaluation["report"]["reference_words"]

    metrics = build_metrics(incomplete)

    assert "prediction_words_mean" not in metrics
    assert "reference_words_mean" not in metrics


def test_a_training_block_without_curves_logs_none() -> None:
    malformed = record()
    assert malformed.training is not None
    malformed.training["epochs"] = "not a list"

    assert build_payload(malformed).epochs == ()
