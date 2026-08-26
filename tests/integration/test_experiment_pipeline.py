"""Integration tests of the experiment chain.

The link under test is the one section 35 cares about here: a configuration
file goes in, and a row of ``reports/results/experiments.csv`` comes out,
through the real tokeniser, the real trainer, the real evaluator and the real
aggregation. The model is tiny and the corpus is synthetic, so nothing measured
here is a score; what is checked is that the chain holds and that a number
reaches the table it belongs to.

The ``slow`` test at the end runs the zero shot baseline on the real
``t5-small`` weights, which is the only way to know that the pretrained branch
of the runner builds something that actually summarises.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.data.config import DataPipelineConfig
from src.experiments.ablation import DATASET_SIZE_CSV, EXPERIMENTS_CSV, aggregate
from src.experiments.config import ExperimentConfig, load_experiment_config
from src.experiments.record import STATUS_NOT_RUN, STATUS_OK, run_directory
from src.experiments.run import execute
from src.metrics.rouge import REPORTED_VARIANT
from src.models.pretrained.t5 import T5Summarizer

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _offline_random_t5(monkeypatch: pytest.MonkeyPatch, tiny_t5: Any) -> None:
    """Keep the random-initialisation branch offline and small."""
    monkeypatch.setattr(T5Summarizer, "load_random_model", lambda config: tiny_t5(32100))


def experiment_payload(data_config: Path, name: str, **overrides: Any) -> dict[str, Any]:
    """Return a from scratch experiment small enough to train in a test."""
    payload: dict[str, Any] = {
        "experiment": {"name": name, "seed": 42, "studies": ["dataset_size"]},
        "dataset": {"config": str(data_config), "percentage": 100},
        "model": {
            "type": "random_init",
            "baseline": "t5",
            "revision": "pinned",
        },
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "warmup_ratio": 0.1,
            "mixed_precision": False,
            "early_stopping_patience": 0,
            "log_every_steps": 1000,
            "device": "cpu",
        },
        "evaluation": {
            "split": "test",
            "batch_size": 4,
            "max_new_tokens": 8,
            "num_beams": 1,
            "no_repeat_ngram_size": 0,
            "bootstrap_samples": 20,
        },
    }
    payload.update(overrides)
    return payload


def write_experiment(directory: Path, payload: dict[str, Any]) -> Path:
    """Write an experiment file and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{payload['experiment']['name']}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a written table back."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Configuration file to run directory
# ---------------------------------------------------------------------------


def test_a_configuration_file_produces_a_measured_run(
    data_config_file: Path, tmp_path: Path
) -> None:
    path = write_experiment(
        tmp_path / "configs", experiment_payload(data_config_file, "scratch_tiny")
    )

    record = execute(
        load_experiment_config(path),
        results_dir=tmp_path / "results",
        runs_dir=tmp_path / "runs",
    )

    assert record.status == STATUS_OK
    assert record.rouge(REPORTED_VARIANT) is not None
    # The architecture is read from the same identifier the corpus was
    # encoded with, and the weights of that identifier are not loaded.
    assert record.model["hf_id"] == "t5-small"
    assert record.model["initialization"] == "random"


def test_the_real_tokeniser_drives_the_embedding_table(
    data_config_file: Path, frozen_corpus: DataPipelineConfig, tmp_path: Path
) -> None:
    # A vocabulary one row short raises on the first document that uses the
    # missing identifier, which is the kind of failure that only shows up on a
    # real tokeniser.
    config = ExperimentConfig.model_validate(experiment_payload(data_config_file, "scratch_tiny"))

    record = execute(config, results_dir=tmp_path / "results", runs_dir=tmp_path / "runs")

    assert record.dataset["train_examples"] == frozen_corpus.working_corpus.train_size
    assert record.evaluation is not None
    assert record.evaluation["report"]["size"] == frozen_corpus.working_corpus.test_size


# ---------------------------------------------------------------------------
# Run directory to table
# ---------------------------------------------------------------------------


def test_a_measured_run_reaches_the_tables(data_config_file: Path, tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    write_experiment(configs, experiment_payload(data_config_file, "scratch_tiny"))
    write_experiment(
        configs,
        experiment_payload(
            data_config_file,
            "scratch_never_run",
            dataset={"config": str(data_config_file), "percentage": 50},
        ),
    )

    record = execute(
        load_experiment_config(configs / "scratch_tiny.yaml"),
        results_dir=results,
        runs_dir=tmp_path / "runs",
    )
    aggregate(
        experiments_dir=configs, results_dir=results, output_dir=results, studies=["dataset_size"]
    )

    rows = {row["experiment"]: row for row in read_csv(results / EXPERIMENTS_CSV)}
    assert rows["scratch_tiny"]["status"] == STATUS_OK
    assert float(rows["scratch_tiny"]["rougeL_f"]) == pytest.approx(record.rouge(REPORTED_VARIANT))
    assert rows["scratch_never_run"]["status"] == STATUS_NOT_RUN
    assert rows["scratch_never_run"]["rougeL_f"] == ""


def test_the_curve_keeps_a_row_for_the_point_that_was_not_run(
    data_config_file: Path, tmp_path: Path
) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    write_experiment(configs, experiment_payload(data_config_file, "scratch_tiny"))
    write_experiment(
        configs,
        experiment_payload(
            data_config_file,
            "scratch_never_run",
            dataset={"config": str(data_config_file), "percentage": 50},
        ),
    )
    execute(
        load_experiment_config(configs / "scratch_tiny.yaml"),
        results_dir=results,
        runs_dir=tmp_path / "runs",
    )

    aggregate(
        experiments_dir=configs, results_dir=results, output_dir=results, studies=["dataset_size"]
    )

    curve = read_csv(results / DATASET_SIZE_CSV)
    assert [row["dataset_percentage"] for row in curve] == ["50", "100"]
    assert [row["status"] for row in curve] == [STATUS_NOT_RUN, STATUS_OK]


def test_a_resumed_run_continues_rather_than_restarts(
    data_config_file: Path, tmp_path: Path
) -> None:
    payload = experiment_payload(data_config_file, "scratch_tiny")
    payload["training"] = {**payload["training"], "epochs": 2}
    config = ExperimentConfig.model_validate(payload)
    results, runs = tmp_path / "results", tmp_path / "runs"

    first = execute(config, results_dir=results, runs_dir=runs)
    resumed = execute(config, results_dir=results, runs_dir=runs, resume=True)

    assert first.training is not None
    assert resumed.training is not None
    # Every epoch was already run, so the resumed run has nothing left to do.
    assert resumed.training["epochs"] == []


# ---------------------------------------------------------------------------
# The pretrained branch, on real weights
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_zero_shot_baseline_runs_from_a_configuration_file(
    data_config_file: Path, tmp_path: Path
) -> None:
    config = ExperimentConfig.model_validate(
        {
            "experiment": {"name": "zero_shot_tiny", "seed": 42, "studies": ["dataset_size"]},
            "dataset": {"config": str(data_config_file)},
            "model": {"type": "pretrained", "baseline": "t5", "mode": "zero_shot"},
            "evaluation": {
                "split": "test",
                "batch_size": 2,
                "max_new_tokens": 16,
                "num_beams": 1,
                "bootstrap_samples": 20,
            },
        }
    )

    record = execute(config, results_dir=tmp_path / "results", runs_dir=tmp_path / "runs")

    assert record.status == STATUS_OK
    assert record.model["mode"] == "zero_shot"
    assert record.model["hf_id"] == "t5-small"
    assert record.training is None
    assert record.dataset["train_examples"] == 0
    assert (run_directory(tmp_path / "results", "zero_shot_tiny")).is_dir()


@pytest.mark.slow
def test_two_zero_shot_runs_give_the_same_score(data_config_file: Path, tmp_path: Path) -> None:
    # A measurement that moved between two identical runs could not support a
    # comparison with anything.
    config = ExperimentConfig.model_validate(
        {
            "experiment": {"name": "zero_shot_tiny", "seed": 42, "studies": ["dataset_size"]},
            "dataset": {"config": str(data_config_file)},
            "model": {"type": "pretrained", "baseline": "t5", "mode": "zero_shot"},
            "evaluation": {"split": "test", "batch_size": 2, "max_new_tokens": 16, "num_beams": 1},
        }
    )

    first = execute(config, results_dir=tmp_path / "a", runs_dir=tmp_path / "runs")
    second = execute(config, results_dir=tmp_path / "b", runs_dir=tmp_path / "runs")

    assert first.rouge(REPORTED_VARIANT) == second.rouge(REPORTED_VARIANT)
    assert first.interval(REPORTED_VARIANT) == second.interval(REPORTED_VARIANT)
