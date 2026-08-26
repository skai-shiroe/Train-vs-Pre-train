"""Integration tests between a run record and a real MLflow store.

The unit tests assert what the payload holds. This one asserts that MLflow
accepts it: parameter names, metric names, tag names and artefact paths all have
rules of their own, and a payload that a fake tracker swallows can still be
rejected by the real one.

The store is a SQLite database inside the test directory. MLflow 3 refuses the
filesystem tracking backend, so a local store is a database, and that is worth
pinning here rather than discovering on the first traced experiment.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from src.experiments.config import RandomInitModelConfig
from src.experiments.record import STATUS_OK, STATUS_PARTIAL, RunRecord, run_directory, write_record
from src.tracking.client import MlflowTracker
from src.tracking.log import main as log_main
from src.tracking.payload import build_payload

pytestmark = pytest.mark.integration


def record(name: str = "scratch_100", status: str = STATUS_OK) -> RunRecord:
    """Return a record shaped like the ones the runner writes."""
    report: dict[str, Any] = {
        "size": 1000,
        "empty_predictions": 2,
        "prediction_words": {"mean": 12.0},
        "reference_words": {"mean": 20.0},
        "rouge": {
            "scores": {
                "rouge1": {"fmeasure": 0.30},
                "rouge2": {"fmeasure": 0.08},
                "rougeL": {"fmeasure": 0.15},
            },
            "intervals": {"rougeL": {"low": 0.14, "high": 0.16}},
        },
    }
    return RunRecord(
        experiment=name,
        status=status,
        config={
            "experiment": {"name": name, "seed": 42, "studies": ["dataset_size"]},
            "dataset": {"config": "configs/data/cnn_dailymail.yaml", "percentage": 100},
            # Dumped from the model rather than spelled out: the row that
            # joins this record to its declaration compares the two blocks.
            "model": RandomInitModelConfig(type="random_init", baseline="t5").model_dump(
                mode="json"
            ),
            "training": {"epochs": 3, "learning_rate": 0.0003, "max_steps": None},
            "evaluation": {"split": "test", "num_beams": 4},
        },
        dataset={
            "config": "configs/data/cnn_dailymail.yaml",
            "name": "cnn_dailymail",
            "version": "259d8397ce78",  # pragma: allowlist secret
            "percentage": 100,
            "train_examples": 20000,
        },
        model={
            "baseline": "t5",
            "initialization": "random",
            "mode": "trained",
            "parameters": "44000000",
        },
        hardware={"torch_version": "2.7.0", "cuda_available": "True", "gpu_name": "RTX 5070"},
        provenance={"git_commit": "deadbeef", "git_branch": "master", "git_dirty": "false"},
        training={
            "best_epoch": 1,
            "best_validation_loss": 3.2,
            "stopped_early": False,
            "global_step": 250,
            "duration_seconds": 600.0,
            "best_checkpoint": "runs/scratch_100/best.pt",
            "epochs": [
                {
                    "epoch": index,
                    "train_loss": 4.0 - index,
                    "validation_loss": 3.5 - index * 0.3,
                    "learning_rate": 0.0003,
                    "steps": 125,
                    "duration_seconds": 300.0,
                    "improved": True,
                }
                for index in range(2)
            ],
            "history": {},
        },
        evaluation={
            "split": "test",
            "split_size": 1000,
            "partial": status != STATUS_OK,
            "generation": {"num_beams": 4, "max_new_tokens": 64},
            "rouge_config": {"seed": 42, "bootstrap_samples": 1000},
            "duration_seconds": 120.0,
            "report": report,
        },
        duration_seconds=730.0,
    )


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Return the URI of a tracking store local to the test directory.

    The database is named by an absolute path and the working directory is
    moved, because MLflow resolves both the store and the artifact root
    relative to the process. A relative URI left in MLflow's global state would
    land a database in the repository the moment a later test touched it.

    An active run is closed on the way out for the same reason. MLflow keeps
    the active run in process global state while each test gets its own
    database, so a test that leaves one open hands the next one a run that
    belongs to a store it no longer points at.
    """
    import mlflow

    monkeypatch.chdir(tmp_path)
    previous = mlflow.get_tracking_uri()
    try:
        yield f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous)


def test_a_complete_run_reaches_the_store_whole(tmp_path: Path, store: str) -> None:
    import mlflow

    directory = run_directory(tmp_path / "results", "scratch_100")
    write_record(record(), directory)
    (directory / "metrics.json").write_text("{}", encoding="utf-8")

    tracker = MlflowTracker(tracking_uri=store, experiment="syntra-test")
    run_id = tracker.log(build_payload(record(), directory=directory))

    assert run_id is not None
    stored = mlflow.get_run(run_id)

    assert stored.data.tags["mlflow.runName"] == "scratch_100"
    assert stored.data.params["dataset.version"] == "259d8397ce78"  # pragma: allowlist secret
    assert stored.data.params["training.epochs"] == "3"
    assert stored.data.metrics["rougeL_f"] == pytest.approx(0.15)
    assert stored.data.metrics["training_duration_seconds"] == pytest.approx(600.0)
    assert stored.data.tags["syntra.git_commit"] == "deadbeef"
    assert stored.data.tags["syntra.hardware.gpu_name"] == "RTX 5070"

    artifacts = {item.path for item in mlflow.artifacts.list_artifacts(run_id=run_id)}
    assert artifacts == {"run.json", "metrics.json"}


def test_the_training_curve_is_readable_step_by_step(tmp_path: Path, store: str) -> None:
    import mlflow

    tracker = MlflowTracker(tracking_uri=store, experiment="syntra-test")
    run_id = tracker.log(build_payload(record()))

    assert run_id is not None
    history = mlflow.MlflowClient(tracking_uri=store).get_metric_history(run_id, "train_loss")

    assert [(point.step, point.value) for point in history] == [(0, 4.0), (1, 3.0)]


def test_a_run_with_nothing_to_measure_still_lands(tmp_path: Path, store: str) -> None:
    # The store must hold the run even when there is no number to attach: the
    # question a reader asks is which experiments exist, not which scored.
    import mlflow

    from src.tracking.payload import TrackedRun

    tracker = MlflowTracker(tracking_uri=store, experiment="syntra-test")
    run_id = tracker.log(TrackedRun(name="scratch_100:failed", tags={"syntra.status": "FAILED"}))

    assert run_id is not None
    assert mlflow.get_run(run_id).data.metrics == {}


def test_a_partial_run_lands_without_a_score(tmp_path: Path, store: str) -> None:
    import mlflow

    tracker = MlflowTracker(tracking_uri=store, experiment="syntra-test")
    run_id = tracker.log(build_payload(record(status=STATUS_PARTIAL)))

    assert run_id is not None
    stored = mlflow.get_run(run_id)

    assert stored.data.tags["mlflow.runName"] == "scratch_100:partial"
    assert stored.data.tags["syntra.status"] == STATUS_PARTIAL
    assert "rougeL_f" not in stored.data.metrics
    assert stored.data.metrics["best_validation_loss"] == pytest.approx(3.2)


def test_the_command_sends_the_records_of_the_declared_experiments(
    tmp_path: Path, store: str, capsys: pytest.CaptureFixture[str]
) -> None:
    import yaml

    configs = tmp_path / "configs"
    configs.mkdir(parents=True)
    for name in ("scratch_100", "scratch_10"):
        (configs / f"{name}.yaml").write_text(
            yaml.safe_dump(
                {
                    "experiment": {"name": name, "seed": 42},
                    "dataset": {"percentage": 100 if name == "scratch_100" else 10},
                    "model": {"type": "random_init", "baseline": "t5"},
                    "training": {"epochs": 1},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    results = tmp_path / "results"
    write_record(record(), run_directory(results, "scratch_100"))

    code = log_main(
        [
            "--all",
            "--experiments-dir",
            str(configs),
            "--results",
            str(results),
            "--tracking-uri",
            store,
            "--tracking-experiment",
            "syntra-test",
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    # The experiment that never ran is reported and produces no run in the
    # store: a row there for something nobody executed would be a fabrication.
    assert "scratch_10               NOT_RUN  nothing to send" in output
    assert "scratch_100              OK" in output


def test_a_run_is_readable_in_the_store_before_it_ends(tmp_path: Path, store: str) -> None:
    # The point of the live half: what the loop measures is queryable while the
    # training is still going, so following a six hour run does not mean
    # watching checkpoint timestamps.
    import mlflow

    from src.tracking.live import LiveMetricsCallback
    from src.training.state import EpochMetrics, TrainingState

    tracker = MlflowTracker(tracking_uri=store, experiment="syntra-test")
    live = tracker.open("scratch_100")
    callback = LiveMetricsCallback(live, every_steps=1)

    callback.on_step_end(
        TrainingState(epoch=0, epochs=3, global_step=1, total_steps=100, last_loss=2.5)
    )
    callback.on_epoch_end(
        TrainingState(epoch=0, epochs=3, global_step=1, total_steps=100),
        EpochMetrics(
            epoch=0,
            train_loss=4.0,
            validation_loss=3.5,
            learning_rate=0.0002,
            steps=1,
            duration_seconds=1.0,
            improved=True,
        ),
    )

    client = mlflow.MlflowClient(tracking_uri=store)
    running = client.get_run(live.run_id)

    assert running.info.status == "RUNNING"
    assert running.data.metrics["step_loss"] == pytest.approx(2.5)
    assert running.data.metrics["epoch_validation_loss"] == pytest.approx(3.5)
    # No score yet, and that is the point: an open run carries the observation
    # of the loop and cannot be read as a measurement.
    assert "rougeL_f" not in running.data.metrics


def test_the_finished_record_closes_the_run_it_opened(tmp_path: Path, store: str) -> None:
    # One run per experiment. An open run beside a finished mirror would leave
    # a reader asking which of the two is the result.
    import mlflow

    tracker = MlflowTracker(tracking_uri=store, experiment="syntra-test")
    live = tracker.open("scratch_100")
    live.log_metrics({"step_loss": 2.5}, step=1)
    run_id = live.finish(build_payload(record()))

    assert run_id == live.run_id
    client = mlflow.MlflowClient(tracking_uri=store)
    stored = client.get_run(run_id)

    assert stored.info.status == "FINISHED"
    assert stored.data.tags["mlflow.runName"] == "scratch_100"
    # The stream and the record curves coexist without merging: one is indexed
    # by optimisation step, the other by epoch index.
    assert stored.data.metrics["step_loss"] == pytest.approx(2.5)
    assert stored.data.metrics["rougeL_f"] == pytest.approx(0.15)
    assert [point.step for point in client.get_metric_history(run_id, "train_loss")] == [0, 1]

    experiment = client.get_experiment_by_name("syntra-test")
    assert experiment is not None
    assert len(client.search_runs([experiment.experiment_id])) == 1


def test_a_run_that_failed_is_renamed_when_it_closes(tmp_path: Path, store: str) -> None:
    # The status is not known while the run trains, so the name it opened under
    # is not the name it must keep.
    import mlflow

    tracker = MlflowTracker(tracking_uri=store, experiment="syntra-test")
    live = tracker.open("scratch_100")
    run_id = live.finish(build_payload(record(status=STATUS_PARTIAL)))

    assert run_id is not None
    stored = mlflow.MlflowClient(tracking_uri=store).get_run(run_id)

    assert stored.data.tags["mlflow.runName"] == "scratch_100:partial"
    assert stored.info.status == "FINISHED"
