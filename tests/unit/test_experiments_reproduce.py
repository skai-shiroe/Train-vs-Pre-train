"""Unit tests of the reproduction chain of section 41.

The chain itself is thin: it calls five commands that are each tested where
they live. What is worth testing here is the part that is not a call, and every
property below is a way a green ``make reproduce`` would say something false.

A quick run must stay a pipeline test: capped on both sides, marked partial,
written outside the campaign, traced nowhere. A step that fails must stop the
chain rather than let the aggregation publish the previous campaign under the
current code. And the summary must not let a verification pass read as a
result.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from src.data.config import DataPipelineConfig
from src.data.example import Example
from src.experiments.config import ExperimentConfig
from src.experiments.record import STATUS_FAILED, STATUS_OK, STATUS_PARTIAL, RunRecord
from src.experiments.reproduce import (
    DONE,
    FAILED,
    MODE_FULL,
    MODE_QUICK,
    QUICK_BATCH_SIZE,
    QUICK_BOOTSTRAP_SAMPLES,
    QUICK_EVALUATION_EXAMPLES,
    QUICK_MAX_NEW_TOKENS,
    QUICK_MAX_STEPS,
    QUICK_ROOT,
    QUICK_TRAIN_EXAMPLES,
    QUICK_VALIDATION_EXAMPLES,
    SKIPPED,
    Destinations,
    Settings,
    StepResult,
    destinations_for,
    format_summary,
    main,
    prepare_corpus,
    prepare_page,
    quick_corpus,
    quick_experiment,
    reproduce,
    run_experiments,
    run_step,
)
from src.experiments.run import Corpus

pytestmark = pytest.mark.unit


def scratch_payload(**overrides: Any) -> dict[str, Any]:
    """Return a from scratch experiment declaration."""
    payload: dict[str, Any] = {
        "experiment": {"name": "scratch_10", "seed": 7, "studies": ["dataset_size"]},
        "dataset": {"percentage": 10},
        "model": {"type": "scratch", "d_model": 32, "num_heads": 2},
        "training": {"epochs": 10, "batch_size": 16, "gradient_accumulation_steps": 4},
        "evaluation": {"split": "test", "num_beams": 4, "max_new_tokens": 64},
    }
    payload.update(overrides)
    return payload


def scratch_config(**overrides: Any) -> ExperimentConfig:
    """Return a validated from scratch experiment."""
    return ExperimentConfig.model_validate(scratch_payload(**overrides))


def zero_shot_config() -> ExperimentConfig:
    """Return the zero shot baseline, which has no training block to cap."""
    return ExperimentConfig.model_validate(
        {
            "experiment": {"name": "pretrained_zero_shot", "seed": 42},
            "model": {"type": "pretrained", "mode": "zero_shot"},
        }
    )


def corpus(
    pipeline_config: DataPipelineConfig,
    make_examples: Callable[[int], list[Example]],
) -> Corpus:
    """Return a loaded corpus large enough to be truncated."""
    return Corpus(
        pipeline=pipeline_config,
        tokenizer=object(),
        dataset_version="0" * 16,
        train=tuple(make_examples(200)),
        validation=tuple(make_examples(100)),
        evaluation=tuple(make_examples(50)),
    )


def record(name: str, status: str) -> RunRecord:
    """Return a run record carrying a status and nothing else."""
    return RunRecord(experiment=name, status=status, config={}, dataset={}, model={}, hardware={})


# ---------------------------------------------------------------------------
# What quick mode cuts, and what it must not touch
# ---------------------------------------------------------------------------


def test_quick_mode_caps_the_training_and_the_decoding() -> None:
    quick = quick_experiment(scratch_config())

    assert quick.training is not None
    assert quick.training.max_steps == QUICK_MAX_STEPS
    assert quick.training.epochs == 1
    assert quick.training.batch_size == QUICK_BATCH_SIZE
    assert quick.evaluation.num_beams == 1
    assert quick.evaluation.max_new_tokens == QUICK_MAX_NEW_TOKENS
    assert quick.evaluation.bootstrap_samples == QUICK_BOOTSTRAP_SAMPLES


def test_a_capped_run_is_partial_by_construction() -> None:
    assert not scratch_config().capped
    assert quick_experiment(scratch_config()).capped


def test_quick_mode_leaves_the_identity_of_the_experiment_alone() -> None:
    declared = scratch_config()
    quick = quick_experiment(declared)

    assert quick.name == declared.name
    assert quick.experiment.seed == declared.experiment.seed
    assert quick.model == declared.model
    assert quick.dataset.percentage == declared.dataset.percentage
    assert quick.evaluation.split == declared.evaluation.split


def test_the_zero_shot_baseline_keeps_no_training_block() -> None:
    quick = quick_experiment(zero_shot_config())

    assert quick.training is None
    assert not quick.capped
    assert quick.evaluation.num_beams == 1


def test_quick_mode_truncates_training_and_validation_only(
    pipeline_config: DataPipelineConfig,
    make_examples: Callable[[int], list[Example]],
) -> None:
    loaded = corpus(pipeline_config, make_examples)

    cut = quick_corpus(loaded)

    assert len(cut.train) == QUICK_TRAIN_EXAMPLES
    assert len(cut.validation) == QUICK_VALIDATION_EXAMPLES
    # The evaluation split keeps its size: the run is scored on the first
    # documents through the limit, so the record says eight of fifty rather
    # than eight of eight.
    assert cut.evaluation == loaded.evaluation
    assert cut.dataset_version == loaded.dataset_version


# ---------------------------------------------------------------------------
# Where a quick run is allowed to write
# ---------------------------------------------------------------------------


def test_quick_mode_writes_nothing_the_campaign_reads() -> None:
    quick = destinations_for(MODE_QUICK)
    full = destinations_for(MODE_FULL)

    assert quick.results.is_relative_to(QUICK_ROOT)
    assert quick.fragments.is_relative_to(QUICK_ROOT)
    assert quick.readme.is_relative_to(QUICK_ROOT)
    assert all(directory.is_relative_to(QUICK_ROOT) for directory in quick.figures)
    assert quick.runs != full.runs
    assert not any(directory in full.figures for directory in quick.figures)


def test_full_mode_draws_the_figures_where_the_report_and_the_site_read_them() -> None:
    full = destinations_for(MODE_FULL)

    assert len(full.figures) == 2
    assert full.readme == Path("README.md")


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown mode"):
        destinations_for("almost")


@pytest.mark.parametrize("name", ["README.md", "RAPPORT.md"])
def test_full_mode_injects_into_the_page_of_the_repository(name: str) -> None:
    page = Path(name)
    before = page.read_bytes()

    target = prepare_page(Settings(mode=MODE_FULL), page, page)

    assert target == page
    # Choosing the page writes nothing: the injection belongs to the step.
    assert page.read_bytes() == before


@pytest.mark.parametrize("name", ["README.md", "RAPPORT.md"])
def test_quick_mode_injects_into_a_copy(tmp_path: Path, name: str) -> None:
    destinations = Destinations(
        results=tmp_path / "results",
        runs=tmp_path / "runs",
        figures=(tmp_path / "figures",),
        fragments=tmp_path / "_generated",
        readme=tmp_path / "README.md",
        report=tmp_path / "RAPPORT.md",
    )
    source = Path(name)

    target = prepare_page(
        Settings(mode=MODE_QUICK, destinations=destinations), source, tmp_path / name
    )

    assert target == tmp_path / name
    assert target.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_a_quick_run_never_traces() -> None:
    assert not Settings(mode=MODE_QUICK, tracking=True).traces
    assert Settings(mode=MODE_FULL, tracking=True).traces
    assert not Settings(mode=MODE_FULL, tracking=False).traces


# ---------------------------------------------------------------------------
# The corpus step
# ---------------------------------------------------------------------------


def test_an_existing_corpus_is_reported_rather_than_rebuilt(
    data_config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(_: DataPipelineConfig) -> dict[str, Any]:
        raise AssertionError("the corpus must not be rebuilt when it is already there")

    monkeypatch.setattr("src.experiments.reproduce.build_corpus", refuse)

    status, detail = prepare_corpus(data_config_file, rebuild=False)

    assert status == SKIPPED
    assert "0000" in detail


def test_the_rebuild_flag_rebuilds_and_reports_the_version(
    data_config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.experiments.reproduce.build_corpus",
        lambda _: {"dataset_version": "abcdef123456789"},  # pragma: allowlist secret
    )

    status, detail = prepare_corpus(data_config_file, rebuild=True)

    assert status == DONE
    assert "abcdef123456" in detail  # pragma: allowlist secret


def test_a_missing_corpus_is_built(
    data_config_file: Path,
    frozen_corpus: DataPipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (frozen_corpus.paths.processed / "manifest.json").unlink()
    monkeypatch.setattr(
        "src.experiments.reproduce.build_corpus", lambda _: {"dataset_version": "112233445566"}
    )

    status, _ = prepare_corpus(data_config_file, rebuild=False)

    assert status == DONE


# ---------------------------------------------------------------------------
# The experiment step
# ---------------------------------------------------------------------------


class Sweep:
    """Record what the chain asked of the runner, without running anything."""

    def __init__(self, status: str = STATUS_PARTIAL) -> None:
        self.status = status
        self.calls: list[dict[str, Any]] = []

    def __call__(self, config: ExperimentConfig, **kwargs: Any) -> RunRecord:
        self.calls.append({"config": config, **kwargs})
        return record(config.name, self.status)


@pytest.fixture
def offline_sweep(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_config: DataPipelineConfig,
    make_examples: Callable[[int], list[Example]],
) -> Callable[..., Sweep]:
    """Return a factory replacing the runner, the loader and the tracker."""

    def factory(*, configs: list[ExperimentConfig], status: str = STATUS_PARTIAL) -> Sweep:
        sweep = Sweep(status)
        monkeypatch.setattr("src.experiments.reproduce.discover_experiments", lambda _: configs)
        monkeypatch.setattr("src.experiments.reproduce.run_one", sweep)
        monkeypatch.setattr(
            "src.experiments.reproduce.load_corpus",
            lambda _: corpus(pipeline_config, make_examples),
        )
        return sweep

    return factory


def test_a_quick_sweep_limits_the_evaluation_and_hands_over_a_cut_corpus(
    offline_sweep: Callable[..., Sweep],
) -> None:
    sweep = offline_sweep(configs=[scratch_config()])
    settings = Settings(mode=MODE_QUICK, destinations=destinations_for(MODE_QUICK))

    status, _ = run_experiments(settings)
    call = sweep.calls[0]

    assert status == DONE
    assert call["limit"] == QUICK_EVALUATION_EXAMPLES
    assert call["corpus"] is not None
    assert len(call["corpus"].train) == QUICK_TRAIN_EXAMPLES
    assert call["config"].capped
    assert not call["resume"]
    assert call["results_dir"].is_relative_to(QUICK_ROOT)
    assert call["runs_dir"] != Path("runs")


def test_a_full_sweep_runs_the_declared_experiment_as_declared(
    offline_sweep: Callable[..., Sweep],
) -> None:
    declared = scratch_config()
    sweep = offline_sweep(configs=[declared], status=STATUS_OK)

    status, detail = run_experiments(Settings(mode=MODE_FULL))
    call = sweep.calls[0]

    assert status == DONE
    assert "1 experiments run" in detail
    assert call["limit"] is None
    assert call["corpus"] is None
    assert call["config"] == declared


def test_a_failed_experiment_fails_the_step_and_names_itself(
    offline_sweep: Callable[..., Sweep],
) -> None:
    offline_sweep(configs=[scratch_config()], status=STATUS_FAILED)

    status, detail = run_experiments(Settings(mode=MODE_FULL))

    assert status == FAILED
    assert "scratch_10" in detail


def test_an_empty_experiments_directory_is_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("src.experiments.reproduce.discover_experiments", lambda _: [])

    status, detail = run_experiments(Settings(mode=MODE_FULL, experiments_dir=tmp_path))

    assert status == FAILED
    assert "no experiment file" in detail


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------


def test_a_step_that_raises_is_recorded_rather_than_propagated() -> None:
    def boom() -> tuple[str, str]:
        raise RuntimeError("the corpus is not where the configuration says")

    result = run_step("data", boom)

    assert result.failed
    assert result.detail.startswith("RuntimeError: ")
    assert result.duration_seconds >= 0


def test_the_chain_stops_at_the_first_failing_step(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []

    def plan(_: Settings) -> tuple[tuple[str, Any], ...]:
        def step(name: str, status: str) -> tuple[str, Any]:
            def action() -> tuple[str, str]:
                ran.append(name)
                return status, name

            return name, action

        return (step("data", DONE), step("experiments", FAILED), step("tables", DONE))

    monkeypatch.setattr("src.experiments.reproduce.steps", plan)

    results = reproduce(Settings(mode=MODE_QUICK))

    assert ran == ["data", "experiments"]
    assert [result.status for result in results] == [DONE, FAILED]


def test_the_five_steps_of_section_41_are_chained_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.experiments.reproduce.prepare_corpus", lambda *_, **__: (SKIPPED, "corpus")
    )
    for step in (
        "run_experiments",
        "aggregate_tables",
        "draw_figures",
        "synchronise_documentation",
    ):
        monkeypatch.setattr(f"src.experiments.reproduce.{step}", lambda _: (DONE, "ok"))

    results = reproduce(Settings(mode=MODE_QUICK))

    assert [result.name for result in results] == [
        "data",
        "experiments",
        "tables",
        "figures",
        "documentation",
    ]


# ---------------------------------------------------------------------------
# What the summary says
# ---------------------------------------------------------------------------


def done(name: str) -> StepResult:
    """Return a step that succeeded in no time at all."""
    return StepResult(name, DONE, "ok", 0.5)


def test_a_green_quick_run_says_it_measured_nothing() -> None:
    summary = format_summary(Settings(mode=MODE_QUICK), [done("data"), done("experiments")])

    assert "QUICK RUN" in summary
    assert "PARTIAL" in summary
    assert "MODE=full" in summary


def test_a_green_full_run_claims_nothing_of_the_sort() -> None:
    summary = format_summary(Settings(mode=MODE_FULL), [done("data")])

    assert "QUICK RUN" not in summary
    assert "mode             full" in summary


def test_the_summary_names_the_step_that_stopped_the_chain() -> None:
    results = [done("data"), StepResult("experiments", FAILED, "boom", 1.0)]

    summary = format_summary(Settings(mode=MODE_FULL), results)

    assert "stopped on step 'experiments'" in summary
    assert "QUICK RUN" not in summary


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_the_command_line_defaults_to_the_mode_that_produces_no_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Settings] = []

    def capture(settings: Settings) -> list[StepResult]:
        seen.append(settings)
        return [done("data")]

    monkeypatch.setattr("src.experiments.reproduce.reproduce", capture)

    assert main([]) == 0
    assert seen[0].mode == MODE_QUICK
    assert seen[0].destinations == destinations_for(MODE_QUICK)


def test_a_failed_chain_exits_non_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.experiments.reproduce.reproduce",
        lambda _: [done("data"), StepResult("experiments", FAILED, "boom", 1.0)],
    )

    assert main(["--mode", "full"]) == 1


def test_the_no_tracking_flag_reaches_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Settings] = []
    monkeypatch.setattr(
        "src.experiments.reproduce.reproduce",
        lambda settings: (seen.append(settings), [done("data")])[1],
    )

    assert main(["--mode", "full", "--no-tracking", "--rebuild-data"]) == 0
    assert not seen[0].tracking
    assert seen[0].rebuild_data
    assert seen[0].destinations == destinations_for(MODE_FULL)


def test_an_unknown_mode_is_refused_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        main(["--mode", "almost"])
