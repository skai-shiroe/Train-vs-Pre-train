"""Unit tests of the experiment runner.

The tokeniser is faked and the model is tiny: what is checked is the path a run
takes from a configuration file to the files on disk, never what any model
learns. Two of these tests matter more than the others.

``test_the_evaluated_weights_come_from_the_best_checkpoint`` pins the decision
that separates a reported score from the last epoch's. The others would all
still pass if the runner measured the weights left in memory.

``test_a_crash_is_recorded_rather_than_swallowed`` pins section 44: an
experiment that fails has to leave a row, not a gap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from src.data.config import DataPipelineConfig
from src.evaluation.evaluator import METRICS_FILE, PREDICTIONS_FILE, QUALITATIVE_FILE
from src.experiments import run as run_module
from src.experiments.config import ExperimentConfig
from src.experiments.record import (
    RUN_RECORD_FILE,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_PARTIAL,
    read_record,
    run_directory,
)
from src.experiments.run import (
    HISTORY_FILE,
    _configure_logging,
    build_experiment_summarizer,
    execute,
    failure_record,
    format_record,
    load_corpus,
    main,
    run_one,
)
from src.models.pretrained.base import BaselineConfig
from src.models.pretrained.t5 import T5Summarizer
from src.models.scratch.summarizer import ScratchSummarizer
from src.tracking.payload import TrackedRun
from src.training.state import TrainingResult

pytestmark = pytest.mark.unit


class RecordingTracker:
    """Tracker keeping what the runner handed it."""

    def __init__(self) -> None:
        self.seen: list[TrackedRun] = []

    def log(self, payload: TrackedRun) -> str | None:
        self.seen.append(payload)
        return "run-1"


class BrokenTracker:
    """Tracker standing in for an unreachable store."""

    def log(self, payload: TrackedRun) -> str | None:
        raise ConnectionError("the store is unreachable")


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch: pytest.MonkeyPatch, fake_tokenizer: Any) -> Any:
    """Replace the shared tokeniser with one that needs no network."""
    monkeypatch.setattr(run_module, "build_tokenizer", lambda hf_id: fake_tokenizer)
    return fake_tokenizer


def scratch_payload(data_config: Path, **overrides: Any) -> dict[str, Any]:
    """Return a from scratch experiment small enough for a unit test."""
    payload: dict[str, Any] = {
        "experiment": {"name": "scratch_test", "seed": 42, "studies": ["dataset_size"]},
        "dataset": {"config": str(data_config), "percentage": 100},
        "model": {
            "type": "scratch",
            "d_model": 32,
            "num_heads": 2,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "d_ff": 64,
            "dropout": 0.0,
            "max_position": 512,
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
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = {**payload[key], **value}
        else:
            payload[key] = value
    return payload


@pytest.fixture
def scratch_experiment(data_config_file: Path) -> ExperimentConfig:
    """Return a validated from scratch experiment over the synthetic corpus."""
    return ExperimentConfig.model_validate(scratch_payload(data_config_file))


def write_experiment(directory: Path, payload: dict[str, Any]) -> Path:
    """Write an experiment file and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{payload['experiment']['name']}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Loading the corpus
# ---------------------------------------------------------------------------


def test_a_trained_run_loads_its_proportion(
    scratch_experiment: ExperimentConfig, frozen_corpus: DataPipelineConfig
) -> None:
    corpus = load_corpus(scratch_experiment)

    assert len(corpus.train) == frozen_corpus.working_corpus.train_size
    assert len(corpus.validation) == frozen_corpus.working_corpus.validation_size
    assert len(corpus.evaluation) == frozen_corpus.working_corpus.test_size


def test_a_smaller_proportion_loads_a_prefix(data_config_file: Path) -> None:
    config = ExperimentConfig.model_validate(
        scratch_payload(data_config_file, dataset={"percentage": 50})
    )

    corpus = load_corpus(config)

    assert len(corpus.train) == 10
    assert (
        corpus.train
        == load_corpus(ExperimentConfig.model_validate(scratch_payload(data_config_file))).train[
            :10
        ]
    )


def test_a_zero_shot_run_loads_no_training_data(data_config_file: Path) -> None:
    # Loading a corpus it will not read would be harmless and misleading: the
    # record reports the counts it loaded.
    config = ExperimentConfig.model_validate(
        {
            "experiment": {"name": "zero_shot_test", "studies": ["dataset_size"]},
            "dataset": {"config": str(data_config_file)},
            "model": {"type": "pretrained", "mode": "zero_shot"},
        }
    )

    corpus = load_corpus(config)

    assert corpus.train == ()
    assert corpus.validation == ()
    assert len(corpus.evaluation) == 5


def test_the_vocabulary_covers_the_added_tokens(
    scratch_experiment: ExperimentConfig, fake_tokenizer: Any
) -> None:
    corpus = load_corpus(scratch_experiment)

    assert corpus.vocab_size == len(fake_tokenizer)


def test_an_unknown_proportion_is_reported(data_config_file: Path) -> None:
    config = ExperimentConfig.model_validate(
        scratch_payload(data_config_file, dataset={"percentage": 25})
    )

    with pytest.raises(KeyError, match="not in the manifest"):
        load_corpus(config)


# ---------------------------------------------------------------------------
# Building the summariser
# ---------------------------------------------------------------------------


def test_an_untrained_model_says_so(scratch_experiment: ExperimentConfig) -> None:
    # A randomly initialised Transformer produces summaries too. Filing its
    # score under the trained model would be a fabricated result.
    summarizer = build_experiment_summarizer(scratch_experiment, load_corpus(scratch_experiment))

    assert isinstance(summarizer, ScratchSummarizer)
    assert summarizer.trained is False
    assert summarizer.describe()["mode"] == "untrained"


def test_weights_from_a_checkpoint_mark_the_model_as_trained(
    scratch_experiment: ExperimentConfig,
) -> None:
    corpus = load_corpus(scratch_experiment)
    state = build_experiment_summarizer(scratch_experiment, corpus).model.state_dict()

    summarizer = build_experiment_summarizer(scratch_experiment, corpus, state_dict=state)

    assert summarizer.describe()["mode"] == "trained"
    assert summarizer.describe()["encoder_layers"] == "1"


def test_the_baseline_is_built_from_the_corpus_tokeniser_settings(
    monkeypatch: pytest.MonkeyPatch,
    data_config_file: Path,
    tiny_t5: Any,
    fake_tokenizer: Any,
) -> None:
    seen: list[BaselineConfig] = []

    def fake_build(name: str, config: BaselineConfig, *, fine_tuned: bool) -> T5Summarizer:
        seen.append(config)
        return T5Summarizer(config, tiny_t5(), fake_tokenizer, fine_tuned=fine_tuned)

    monkeypatch.setattr(run_module, "build_summarizer", fake_build)
    config = ExperimentConfig.model_validate(
        {
            "experiment": {"name": "zero_shot_test", "studies": ["dataset_size"]},
            "dataset": {"config": str(data_config_file)},
            "model": {"type": "pretrained", "mode": "zero_shot"},
        }
    )

    summarizer = build_experiment_summarizer(config, load_corpus(config))

    assert summarizer.describe()["mode"] == "zero_shot"
    assert seen[0].source_prefix == "summarize: "
    assert seen[0].max_source_tokens == 512
    assert seen[0].hf_id == "t5-small"


def test_a_fine_tuned_baseline_is_rebuilt_from_its_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    data_config_file: Path,
    tiny_t5: Any,
    fake_tokenizer: Any,
) -> None:
    # The same weights measured zero shot and fine tuned are two results. The
    # mode comes from where the weights were loaded, never from a guess.
    monkeypatch.setattr(
        T5Summarizer,
        "from_pretrained",
        classmethod(
            lambda cls, config=None, *, fine_tuned=False: cls(
                config, tiny_t5(), fake_tokenizer, fine_tuned=fine_tuned
            )
        ),
    )
    config = ExperimentConfig.model_validate(
        {
            "experiment": {"name": "ft_test", "studies": ["dataset_size"]},
            "dataset": {"config": str(data_config_file), "percentage": 100},
            "model": {"type": "pretrained", "mode": "fine_tuned"},
            "training": {"epochs": 1, "batch_size": 4, "device": "cpu"},
        }
    )

    summarizer = build_experiment_summarizer(
        config, load_corpus(config), state_dict=tiny_t5().state_dict()
    )

    assert summarizer.describe()["mode"] == "fine_tuned"


def test_training_a_run_without_a_training_block_is_refused(
    scratch_experiment: ExperimentConfig, data_config_file: Path, tmp_path: Path
) -> None:
    # The loader already refuses such a file, so reaching this is a programming
    # error. It still fails loudly rather than training on a default.
    corpus = load_corpus(scratch_experiment)
    zero_shot = ExperimentConfig.model_validate(
        {
            "experiment": {"name": "zero_shot_test", "studies": ["dataset_size"]},
            "dataset": {"config": str(data_config_file)},
            "model": {"type": "pretrained", "mode": "zero_shot"},
        }
    )

    with pytest.raises(ValueError, match="no training block"):
        run_module.train_experiment(
            zero_shot,
            corpus,
            build_experiment_summarizer(scratch_experiment, corpus),
            output_dir=tmp_path / "runs",
            history_path=tmp_path / "history.json",
        )


# ---------------------------------------------------------------------------
# A whole run
# ---------------------------------------------------------------------------


def test_a_run_writes_every_artefact(scratch_experiment: ExperimentConfig, tmp_path: Path) -> None:
    results = tmp_path / "results"

    record = execute(scratch_experiment, results_dir=results, runs_dir=tmp_path / "runs")

    directory = run_directory(results, "scratch_test")
    assert record.status == STATUS_OK
    assert sorted(path.name for path in directory.iterdir()) == sorted(
        [HISTORY_FILE, METRICS_FILE, PREDICTIONS_FILE, QUALITATIVE_FILE, RUN_RECORD_FILE]
    )


def test_a_run_records_what_produced_the_score(
    scratch_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    record = execute(
        scratch_experiment, results_dir=tmp_path / "results", runs_dir=tmp_path / "runs"
    )

    assert record.dataset["percentage"] == 100
    assert record.dataset["train_examples"] == 20
    assert record.dataset["version"] == "0" * 16
    assert record.model["mode"] == "trained"
    assert record.hardware["torch_version"] == torch.__version__
    assert record.training is not None
    assert record.evaluation is not None
    assert record.evaluation["generation"]["num_beams"] == 1
    assert record.rouge("rougeL") is not None


def test_a_run_records_the_commit_it_was_started_from(
    scratch_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    # Section 19. Stored with the run rather than recomputed later, so a record
    # traced or published next week still names the code that produced it.
    record = execute(
        scratch_experiment, results_dir=tmp_path / "results", runs_dir=tmp_path / "runs"
    )

    assert set(record.provenance) == {"git_commit", "git_branch", "git_dirty"}


def test_a_finished_run_is_traced(scratch_experiment: ExperimentConfig, tmp_path: Path) -> None:
    tracker = RecordingTracker()

    execute(
        scratch_experiment,
        results_dir=tmp_path / "results",
        runs_dir=tmp_path / "runs",
        tracker=tracker,
    )

    assert [payload.name for payload in tracker.seen] == ["scratch_test"]
    assert tracker.seen[0].metrics["rougeL_f"] >= 0.0
    assert [path.name for path in tracker.seen[0].artifacts][0] == RUN_RECORD_FILE


def test_a_run_without_a_tracker_reaches_no_store(
    scratch_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    # The default, so a test that runs an experiment cannot write to a store.
    record = execute(
        scratch_experiment, results_dir=tmp_path / "results", runs_dir=tmp_path / "runs"
    )

    assert record.status == STATUS_OK


def test_a_crashed_run_is_traced_too(data_config_file: Path, tmp_path: Path) -> None:
    # An experiment that crashed is part of what the plan produced.
    tracker = RecordingTracker()
    payload = scratch_payload(data_config_file, dataset={"percentage": 7})
    config = ExperimentConfig.model_validate(payload)

    record = run_one(
        config,
        results_dir=tmp_path / "results",
        runs_dir=tmp_path / "runs",
        limit=None,
        resume=False,
        tracker=tracker,
    )

    assert record.status == STATUS_FAILED
    assert [payload.name for payload in tracker.seen] == ["scratch_test:failed"]


def test_a_store_failure_costs_the_mirror_not_the_measurement(
    scratch_experiment: ExperimentConfig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = execute(
        scratch_experiment,
        results_dir=tmp_path / "results",
        runs_dir=tmp_path / "runs",
        tracker=BrokenTracker(),
    )

    assert record.status == STATUS_OK
    assert record.rouge("rougeL") is not None
    assert "tracking failed" in capsys.readouterr().err


def test_the_evaluated_weights_come_from_the_best_checkpoint(
    scratch_experiment: ExperimentConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The weights in memory when training ends are the last epoch's, and early
    # stopping exists because that is not the epoch the run selected.
    loaded: list[Path] = []
    original = run_module.load_checkpoint

    def spy(path: Path, *, map_location: Any = "cpu") -> dict[str, Any]:
        loaded.append(Path(path))
        return original(path, map_location=map_location)

    monkeypatch.setattr(run_module, "load_checkpoint", spy)

    record = execute(
        scratch_experiment, results_dir=tmp_path / "results", runs_dir=tmp_path / "runs"
    )

    assert record.training is not None
    assert loaded == [Path(record.training["best_checkpoint"])]


def test_the_checkpoint_carries_the_corpus_it_was_trained_on(
    scratch_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    execute(scratch_experiment, results_dir=tmp_path / "results", runs_dir=tmp_path / "runs")

    payload = run_module.load_checkpoint(tmp_path / "runs" / "scratch_test" / "best.pt")

    assert payload["metadata"]["experiment"] == "scratch_test"
    assert payload["metadata"]["dataset_version"] == "0" * 16
    assert payload["metadata"]["dataset_percentage"] == 100
    assert payload["metadata"]["seed"] == 42


def test_the_written_metrics_match_the_record(
    scratch_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    results = tmp_path / "results"

    record = execute(scratch_experiment, results_dir=results, runs_dir=tmp_path / "runs")

    metrics = json.loads((run_directory(results, "scratch_test") / METRICS_FILE).read_text("utf-8"))
    assert record.evaluation is not None
    assert metrics["report"] == record.evaluation["report"]


def test_a_second_run_of_the_same_experiment_gives_the_same_score(
    scratch_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    # A score that moved between two runs of one configuration could not
    # support any comparison.
    first = execute(scratch_experiment, results_dir=tmp_path / "a", runs_dir=tmp_path / "runs_a")
    second = execute(scratch_experiment, results_dir=tmp_path / "b", runs_dir=tmp_path / "runs_b")

    assert first.rouge("rougeL") == second.rouge("rougeL")
    assert first.interval("rougeL") == second.interval("rougeL")


def test_a_finished_run_can_be_measured_again(
    scratch_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    # Resuming a training that has nothing left to do runs no epoch and so
    # reports no best checkpoint of its own. The weights are still on disk.
    results, runs = tmp_path / "results", tmp_path / "runs"
    execute(scratch_experiment, results_dir=results, runs_dir=runs)

    record = execute(scratch_experiment, results_dir=results, runs_dir=runs, resume=True)

    assert record.status == STATUS_OK
    assert record.training is not None
    assert record.training["epochs"] == []
    assert record.rouge("rougeL") is not None


def test_a_training_that_wrote_nothing_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no weights to"):
        run_module.best_weights("scratch_test", TrainingResult(), tmp_path / "runs")


# ---------------------------------------------------------------------------
# Runs that are not results
# ---------------------------------------------------------------------------


def test_a_limited_run_is_partial_and_stands_aside(
    scratch_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    results = tmp_path / "results"

    record = execute(scratch_experiment, results_dir=results, runs_dir=tmp_path / "runs", limit=2)

    assert record.status == STATUS_PARTIAL
    assert record.rouge("rougeL") is None
    assert (run_directory(results, "scratch_test", partial=True) / RUN_RECORD_FILE).is_file()
    assert not run_directory(results, "scratch_test").exists()


def test_a_capped_training_budget_is_partial_too(data_config_file: Path, tmp_path: Path) -> None:
    # A run that stopped short of its schedule exercised the pipeline; it did
    # not measure the configuration it declares.
    config = ExperimentConfig.model_validate(
        scratch_payload(data_config_file, training={"max_steps": 2})
    )

    record = execute(config, results_dir=tmp_path / "results", runs_dir=tmp_path / "runs")

    assert record.status == STATUS_PARTIAL
    assert record.evaluation is not None
    assert record.evaluation["report"]["size"] == 5


def test_a_partial_run_never_overwrites_a_complete_one(
    scratch_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    results = tmp_path / "results"
    execute(scratch_experiment, results_dir=results, runs_dir=tmp_path / "runs")

    execute(scratch_experiment, results_dir=results, runs_dir=tmp_path / "runs", limit=2)

    complete = read_record(run_directory(results, "scratch_test"))
    assert complete is not None
    assert complete.status == STATUS_OK


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def test_a_crash_is_recorded_rather_than_swallowed(data_config_file: Path, tmp_path: Path) -> None:
    config = ExperimentConfig.model_validate(
        scratch_payload(data_config_file, dataset={"config": str(tmp_path / "absent.yaml")})
    )
    results = tmp_path / "results"

    record = run_one(
        config, results_dir=results, runs_dir=tmp_path / "runs", limit=None, resume=False
    )

    assert record.status == STATUS_FAILED
    assert record.error is not None
    assert "FileNotFoundError" in record.error
    assert read_record(run_directory(results, "scratch_test")) is not None


def test_a_failure_record_carries_no_score(scratch_experiment: ExperimentConfig) -> None:
    record = failure_record(scratch_experiment, RuntimeError("out of memory"), 1.5)

    assert record.status == STATUS_FAILED
    assert record.evaluation is None
    assert record.rouge("rougeL") is None
    assert record.error == "RuntimeError: out of memory"


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_one_experiment_runs_from_its_file(data_config_file: Path, tmp_path: Path) -> None:
    path = write_experiment(tmp_path / "configs", scratch_payload(data_config_file))

    code = main(
        [
            "--config",
            str(path),
            "--results",
            str(tmp_path / "results"),
            "--runs",
            str(tmp_path / "runs"),
            # Without it the runner builds a real tracker, MLflow resolves its
            # own URI, and a unit test lands runs named scratch_test in the
            # store of the repository it is run from. What tracking does is
            # asserted above against a fake tracker.
            "--no-tracking",
        ]
    )

    assert code == 0
    assert read_record(run_directory(tmp_path / "results", "scratch_test")) is not None


def test_a_sweep_runs_every_declared_experiment(data_config_file: Path, tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    write_experiment(configs, scratch_payload(data_config_file))
    second = scratch_payload(data_config_file)
    second["experiment"] = {"name": "scratch_half", "seed": 42, "studies": ["dataset_size"]}
    second["dataset"] = {"config": str(data_config_file), "percentage": 50}
    write_experiment(configs, second)

    code = main(
        [
            "--all",
            "--experiments-dir",
            str(configs),
            "--results",
            str(tmp_path / "results"),
            "--runs",
            str(tmp_path / "runs"),
            "--no-tracking",
        ]
    )

    assert code == 0
    for name in ("scratch_test", "scratch_half"):
        assert read_record(run_directory(tmp_path / "results", name)) is not None


def test_a_sweep_survives_one_failure_and_reports_it(
    data_config_file: Path, tmp_path: Path
) -> None:
    configs = tmp_path / "configs"
    write_experiment(configs, scratch_payload(data_config_file))
    broken = scratch_payload(data_config_file)
    broken["experiment"] = {"name": "broken_run", "seed": 42, "studies": ["dataset_size"]}
    broken["dataset"] = {"config": str(tmp_path / "absent.yaml"), "percentage": 100}
    write_experiment(configs, broken)

    code = main(
        [
            "--all",
            "--experiments-dir",
            str(configs),
            "--results",
            str(tmp_path / "results"),
            "--runs",
            str(tmp_path / "runs"),
            "--no-tracking",
        ]
    )

    assert code == 1
    failed = read_record(run_directory(tmp_path / "results", "broken_run"))
    survivor = read_record(run_directory(tmp_path / "results", "scratch_test"))
    assert failed is not None and failed.status == STATUS_FAILED
    assert survivor is not None and survivor.status == STATUS_OK


def test_an_empty_experiments_directory_is_an_error(tmp_path: Path) -> None:
    assert main(["--all", "--experiments-dir", str(tmp_path)]) == 1


def test_the_two_selections_are_exclusive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--all", "--config", str(tmp_path / "x.yaml")])


def test_neither_selection_is_an_error() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_the_console_gets_the_progress_records(monkeypatch: pytest.MonkeyPatch) -> None:
    levels: list[int] = []
    monkeypatch.setattr(
        run_module.logging, "basicConfig", lambda **kwargs: levels.append(kwargs["level"])
    )

    _configure_logging(False)
    _configure_logging(True)

    assert levels == [run_module.logging.INFO, run_module.logging.DEBUG]


def test_the_logging_is_configured_before_anything_can_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The empty directory makes main fail on its first decision. Logging is
    # still configured, because a sweep that dies early has to say why.
    seen: list[bool] = []
    monkeypatch.setattr(run_module, "_configure_logging", lambda verbose: seen.append(verbose))

    assert main(["--all", "--experiments-dir", str(tmp_path)]) == 1
    assert seen == [False]


def test_verbose_reaches_the_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[bool] = []
    monkeypatch.setattr(run_module, "_configure_logging", lambda verbose: seen.append(verbose))

    main(["--all", "--experiments-dir", str(tmp_path), "--verbose"])

    assert seen == [True]


# ---------------------------------------------------------------------------
# The printed summary
# ---------------------------------------------------------------------------


def test_the_summary_of_a_complete_run_reports_the_scores(
    scratch_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    record = execute(
        scratch_experiment, results_dir=tmp_path / "results", runs_dir=tmp_path / "runs"
    )

    summary = format_record(record)

    assert "status           OK" in summary
    assert "rougeL" in summary
    assert "PARTIAL RUN" not in summary


def test_the_summary_of_a_partial_run_says_it_is_not_a_result(
    scratch_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    record = execute(
        scratch_experiment, results_dir=tmp_path / "results", runs_dir=tmp_path / "runs", limit=2
    )

    summary = format_record(record)

    assert "PARTIAL RUN" in summary
    assert "rougeL " not in summary


def test_the_summary_of_a_failed_run_is_a_status(
    scratch_experiment: ExperimentConfig,
) -> None:
    summary = format_record(failure_record(scratch_experiment, RuntimeError("boom"), 0.1))

    assert summary.splitlines()[0] == "status           FAILED"
