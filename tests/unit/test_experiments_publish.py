"""Unit tests of the publication of a run into the model registry.

The property under test is that only a complete measurement reaches the
registry, and that what reaches it carries the run it came from. Everything the
API will later answer about a served model is decided here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from backend.app.registry.local import (
    WEIGHTS_PAYLOAD_KEY,
    WEIGHTS_PAYLOAD_VERSION,
    LocalModelRegistry,
)
from backend.app.registry.models import PRETRAINED, SCRATCH, ModelDraft, ModelSource
from src.experiments.publish import (
    default_name,
    format_version,
    main,
    publish_experiment,
    registry_metrics,
    strip_checkpoint,
)
from src.experiments.record import STATUS_FAILED, STATUS_OK, STATUS_PARTIAL, RunRecord, write_record
from src.experiments.record import run_directory as results_directory
from src.training.checkpoint import BEST_NAME, build_payload, save_checkpoint

pytestmark = pytest.mark.unit


PIPELINE: dict[str, Any] = {
    "name": "xsum",
    "dataset": {
        "hf_id": "EdinburghNLP/xsum",
        "hf_config": None,
        "revision": None,
        "source_column": "document",
        "target_column": "summary",
        "id_column": "id",
    },
    "working_corpus": {
        "seed": 42,
        "train_size": 200,
        "validation_size": 20,
        "test_size": 20,
    },
    "proportions": [10, 50, 100],
    "preprocess": {
        "min_source_chars": 10,
        "max_source_chars": 2000,
        "min_target_chars": 5,
        "max_target_chars": 200,
        "normalise_unicode": True,
        "collapse_whitespace": True,
    },
    "tokenizer": {
        "hf_id": "t5-small",
        "source_prefix": "summarize: ",
        "max_source_tokens": 512,
        "max_target_tokens": 64,
    },
    "paths": {"raw": "data/raw/xsum", "processed": "data/processed/xsum"},
}


class Tiny(torch.nn.Module):
    """A model small enough to checkpoint in a unit test."""

    def __init__(self) -> None:
        """Build the module."""
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)


def write_pipeline(tmp_path: Path) -> Path:
    """Write a data pipeline configuration and return its path."""
    path = tmp_path / "xsum.yaml"
    path.write_text(yaml.safe_dump(PIPELINE, sort_keys=False), encoding="utf-8")
    return path


def declare(
    configs: Path, name: str, *, zero_shot: bool = False, revision: str | None = None
) -> Path:
    """Write one experiment file and return its path."""
    payload: dict[str, Any] = {
        "experiment": {"name": name, "seed": 42, "description": "un essai"},
    }
    if zero_shot:
        payload["model"] = {"type": "pretrained", "mode": "zero_shot", "revision": revision}
    else:
        payload["model"] = {"type": "scratch", "d_model": 32, "num_heads": 2}
        payload["dataset"] = {"percentage": 10}
        payload["training"] = {"epochs": 1}

    configs.mkdir(parents=True, exist_ok=True)
    path = configs / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def record(
    name: str,
    pipeline: Path,
    *,
    status: str = STATUS_OK,
    zero_shot: bool = False,
    revision: str | None = None,
    measured: bool = True,
) -> RunRecord:
    """Return a record shaped like the ones the runner writes."""
    model_block: dict[str, Any] = (
        {"type": "pretrained", "baseline": "t5", "mode": "zero_shot", "revision": revision}
        if zero_shot
        else {"type": "scratch", "d_model": 32, "num_heads": 2}
    )
    evaluation = {
        "split": "test",
        "split_size": 20,
        "partial": False,
        "generation": {"num_beams": 4},
        "rouge_config": {"seed": 42},
        "duration_seconds": 1.0,
        "report": {
            "size": 20,
            "empty_predictions": 0,
            "rouge": {
                "scores": {
                    "rouge1": {"fmeasure": 0.30},
                    "rouge2": {"fmeasure": 0.08},
                    "rougeL": {"fmeasure": 0.15},
                },
                "intervals": {"rougeL": {"low": 0.14, "high": 0.16}},
            },
        },
    }
    return RunRecord(
        experiment=name,
        status=status,
        config={
            "experiment": {"name": name, "seed": 42, "description": "un essai"},
            "dataset": {"config": str(pipeline), "percentage": None if zero_shot else 10},
            "model": model_block,
        },
        dataset={
            "config": str(pipeline),
            "name": "xsum",
            "version": "259d8397ce78",  # pragma: allowlist secret
            "percentage": None if zero_shot else 10,
        },
        model={"model": "scratch", "mode": "trained"},
        hardware={"torch_version": "2.7.0"},
        provenance={"git_commit": "deadbeef", "git_branch": "master", "git_dirty": "false"},
        evaluation=evaluation if measured else None,
        duration_seconds=5.0,
    )


def checkpoint(runs: Path, name: str) -> Path:
    """Write a best checkpoint for one experiment and return its path."""
    model = Tiny()
    payload = build_payload(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters()),
        epoch=0,
        global_step=1,
        best_validation_loss=3.2,
        metadata={"experiment": name},
    )
    return save_checkpoint(runs / name / BEST_NAME, payload)


def prepare(tmp_path: Path, name: str = "scratch_100", **kwargs: Any) -> dict[str, Path]:
    """Lay out a repository holding one declared and measured experiment."""
    configs, results, runs = tmp_path / "configs", tmp_path / "results", tmp_path / "runs"
    pipeline = write_pipeline(tmp_path)
    declare(
        configs, name, zero_shot=kwargs.get("zero_shot", False), revision=kwargs.get("revision")
    )
    written = record(name, pipeline, **kwargs)
    write_record(
        written, results_directory(results, name, partial=written.status == STATUS_PARTIAL)
    )
    if not kwargs.get("zero_shot", False):
        checkpoint(runs, name)
    return {"configs": configs, "results": results, "runs": runs, "registry": tmp_path / "registry"}


# ---------------------------------------------------------------------------
# What may be published
# ---------------------------------------------------------------------------


def test_a_complete_run_is_published_with_its_measurement(tmp_path: Path) -> None:
    paths = prepare(tmp_path)
    registry = LocalModelRegistry(paths["registry"])

    published = publish_experiment(
        "scratch_100",
        registry=registry,
        experiments_dir=paths["configs"],
        results_dir=paths["results"],
        runs_dir=paths["runs"],
    )

    assert published.label == "scratch:v1"
    assert published.experiment == "scratch_100"
    assert published.git_commit == "deadbeef"
    assert published.dataset_version == "259d8397ce78"  # pragma: allowlist secret
    assert published.metrics["rougeL_f"] == pytest.approx(0.15)
    assert published.architecture["d_model"] == 32
    assert published.tokenizer["hf_id"] == "t5-small"
    assert published.description == "un essai"


def test_an_experiment_that_never_ran_is_refused(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    declare(configs, "scratch_100")

    with pytest.raises(ValueError, match="never run"):
        publish_experiment(
            "scratch_100",
            registry=LocalModelRegistry(tmp_path / "registry"),
            experiments_dir=configs,
            results_dir=tmp_path / "results",
            runs_dir=tmp_path / "runs",
        )


def test_an_undeclared_experiment_is_refused(tmp_path: Path) -> None:
    paths = prepare(tmp_path)

    with pytest.raises(ValueError, match="No experiment named"):
        publish_experiment(
            "scratch_10",
            registry=LocalModelRegistry(paths["registry"]),
            experiments_dir=paths["configs"],
            results_dir=paths["results"],
            runs_dir=paths["runs"],
        )


@pytest.mark.parametrize("status", [STATUS_PARTIAL, STATUS_FAILED])
def test_a_run_that_is_not_a_measurement_is_refused(tmp_path: Path, status: str) -> None:
    # The registry answers what a served model scored. A version whose answer
    # is PARTIAL has no answer.
    paths = prepare(tmp_path, status=status)

    with pytest.raises(ValueError, match="carries no measurement"):
        publish_experiment(
            "scratch_100",
            registry=LocalModelRegistry(paths["registry"]),
            experiments_dir=paths["configs"],
            results_dir=paths["results"],
            runs_dir=paths["runs"],
        )


def test_a_measured_run_whose_checkpoint_is_gone_is_refused(tmp_path: Path) -> None:
    paths = prepare(tmp_path)
    (paths["runs"] / "scratch_100" / BEST_NAME).unlink()

    with pytest.raises(FileNotFoundError, match="checkpoint is gone"):
        publish_experiment(
            "scratch_100",
            registry=LocalModelRegistry(paths["registry"]),
            experiments_dir=paths["configs"],
            results_dir=paths["results"],
            runs_dir=paths["runs"],
        )


# ---------------------------------------------------------------------------
# The baseline and its revision
# ---------------------------------------------------------------------------


def test_the_baseline_cannot_be_published_without_a_pinned_revision(tmp_path: Path) -> None:
    # Section 14. This is the rule that keeps an unpinned t5-small out of the
    # thing the API serves.
    paths = prepare(tmp_path, name="pretrained_zero_shot", zero_shot=True, revision=None)

    with pytest.raises(ValueError, match="without pinning a revision"):
        publish_experiment(
            "pretrained_zero_shot",
            registry=LocalModelRegistry(paths["registry"]),
            experiments_dir=paths["configs"],
            results_dir=paths["results"],
            runs_dir=paths["runs"],
        )


def test_the_baseline_is_published_as_a_hub_reference(tmp_path: Path) -> None:
    paths = prepare(tmp_path, name="pretrained_zero_shot", zero_shot=True, revision="df1b051c")
    registry = LocalModelRegistry(paths["registry"])

    published = publish_experiment(
        "pretrained_zero_shot",
        registry=registry,
        experiments_dir=paths["configs"],
        results_dir=paths["results"],
        runs_dir=paths["runs"],
    )

    assert published.name == PRETRAINED
    assert published.source is ModelSource.HUGGING_FACE
    assert published.hf_id == "t5-small"
    assert published.revision == "df1b051c"
    assert registry.resolve_artifact(published) is None


def test_the_side_of_the_comparison_decides_the_name(tmp_path: Path) -> None:
    pipeline = write_pipeline(tmp_path)

    assert default_name(record("scratch_10", pipeline)) == SCRATCH
    assert default_name(record("zero", pipeline, zero_shot=True, revision="a")) == PRETRAINED


# ---------------------------------------------------------------------------
# The artefact
# ---------------------------------------------------------------------------


def test_the_published_weights_carry_no_optimiser_state(tmp_path: Path) -> None:
    # A served model never resumes, and the moments triple the size.
    source = checkpoint(tmp_path / "runs", "scratch_100")
    destination = strip_checkpoint(source, tmp_path / "weights.pt")

    stored = torch.load(destination, map_location="cpu", weights_only=True)

    assert set(stored) == {"version", WEIGHTS_PAYLOAD_KEY}
    assert stored["version"] == WEIGHTS_PAYLOAD_VERSION
    assert "linear.weight" in stored[WEIGHTS_PAYLOAD_KEY]
    assert destination.stat().st_size < source.stat().st_size


def test_the_published_weights_load_into_the_model_they_came_from(tmp_path: Path) -> None:
    model = Tiny()
    payload = build_payload(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters()),
        epoch=0,
        global_step=1,
        best_validation_loss=1.0,
    )
    source = save_checkpoint(tmp_path / "best.pt", payload)

    stored = torch.load(
        strip_checkpoint(source, tmp_path / "weights.pt"), map_location="cpu", weights_only=True
    )
    rebuilt = Tiny()
    rebuilt.load_state_dict(stored[WEIGHTS_PAYLOAD_KEY])

    assert torch.equal(rebuilt.linear.weight, model.linear.weight)


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


def test_publishing_can_promote_in_one_step(tmp_path: Path) -> None:
    paths = prepare(tmp_path)
    registry = LocalModelRegistry(paths["registry"])

    publish_experiment(
        "scratch_100",
        registry=registry,
        experiments_dir=paths["configs"],
        results_dir=paths["results"],
        runs_dir=paths["runs"],
        alias="champion",
    )

    assert registry.get_model("champion").label == "scratch:v1"


def test_an_unknown_alias_is_refused_before_publishing(tmp_path: Path) -> None:
    paths = prepare(tmp_path)
    registry = LocalModelRegistry(paths["registry"])

    with pytest.raises(ValueError, match="Unknown alias"):
        publish_experiment(
            "scratch_100",
            registry=registry,
            experiments_dir=paths["configs"],
            results_dir=paths["results"],
            runs_dir=paths["runs"],
            alias="staging",
        )

    assert registry.names() == []


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def test_the_command_reports_what_it_published(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = prepare(tmp_path)

    code = main(
        [
            "--experiment",
            "scratch_100",
            "--registry",
            str(paths["registry"]),
            "--experiments-dir",
            str(paths["configs"]),
            "--results",
            str(paths["results"]),
            "--runs",
            str(paths["runs"]),
            "--alias",
            "champion",
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "published        scratch:v1" in output
    assert "rougeL           0.1500" in output
    assert "champion points at scratch:v1" in output


def test_a_variant_that_was_not_measured_is_not_published(tmp_path: Path) -> None:
    written = record("scratch_100", write_pipeline(tmp_path))
    assert written.evaluation is not None
    del written.evaluation["report"]["rouge"]["scores"]["rouge2"]

    assert set(registry_metrics(written)) == {"rouge1_f", "rougeL_f"}


def test_the_summary_shows_only_what_was_measured_and_promoted(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "registry")
    source = tmp_path / "weights.pt"
    source.write_bytes(b"weights")
    published = registry.publish(
        ModelDraft(name="scratch", source=ModelSource.CHECKPOINT, metrics={"rougeL_f": 0.15}),
        weights=source,
    )

    rendered = format_version(published, None, registry)

    assert "rougeL           0.1500" in rendered
    assert "rouge1" not in rendered
    assert "alias" not in rendered


def test_the_summary_of_a_hub_version_names_no_file(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "registry")
    published = registry.publish(
        ModelDraft(
            name="pretrained",
            source=ModelSource.HUGGING_FACE,
            hf_id="t5-small",
            revision="df1b051c",
        )
    )
    registry.promote("champion", "pretrained", "v1")

    rendered = format_version(published, "champion", registry)

    assert "artifact" not in rendered
    assert "checksum" not in rendered
    assert "champion points at pretrained:v1" in rendered


def test_the_command_publishes_without_promoting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = prepare(tmp_path)

    code = main(
        [
            "--experiment",
            "scratch_100",
            "--registry",
            str(paths["registry"]),
            "--experiments-dir",
            str(paths["configs"]),
            "--results",
            str(paths["results"]),
            "--runs",
            str(paths["runs"]),
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "serves by default v1" in output
    assert "points at" not in output


def test_a_refused_publication_exits_non_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = prepare(tmp_path, status=STATUS_PARTIAL)

    code = main(
        [
            "--experiment",
            "scratch_100",
            "--registry",
            str(paths["registry"]),
            "--experiments-dir",
            str(paths["configs"]),
            "--results",
            str(paths["results"]),
            "--runs",
            str(paths["runs"]),
        ]
    )

    assert code == 1
    assert "carries no measurement" in capsys.readouterr().err
