"""Unit tests of the generated table freshness check.

The check is the reason the fragments are worth writing: without it a stale
table is only a convention nobody enforces. What it must never do is pass. It
compares the committed fragments against the run records, and the records are
not versioned, so the one dangerous outcome is a green result on a machine that
had nothing to compare.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.check_sync import FAILED, OK, PENDING, check_report_freshness
from src.experiments.fragments import (
    HEADLINE_REGION,
    REGION_BEGIN,
    REGION_END,
    build,
    write,
)
from src.experiments.record import STATUS_OK, RunRecord, run_directory, write_record

pytestmark = pytest.mark.unit


def declare(directory: Path, name: str, **overrides: Any) -> Path:
    """Write one from scratch experiment file and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "experiment": {"name": name, "seed": 42, "studies": ["dataset_size"]},
        "dataset": {"percentage": 10},
        "model": {"type": "scratch", "d_model": 32, "num_heads": 2},
        "training": {"epochs": 1},
    }
    payload.update(overrides)
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def record(name: str, *, beams: int = 4) -> RunRecord:
    """Return a record carrying one ROUGE-L score."""
    return RunRecord(
        experiment=name,
        status=STATUS_OK,
        config={},
        dataset={"train_examples": 2000, "version": "abc123"},
        model={"parameters": "1234567"},
        hardware={"gpu_name": "Test GPU"},
        provenance={"git_commit": "deadbeef"},
        training={"duration_seconds": 60.0},
        evaluation={
            "split": "test",
            "generation": {"num_beams": beams},
            "rouge_config": {"seed": 42},
            "report": {"rouge": {"scores": {"rougeL": {"fmeasure": 0.3}}}},
        },
        duration_seconds=60.0,
    )


def readme_at(directory: Path) -> Path:
    """Write a page carrying the headline markers and return its path."""
    path = directory / "README.md"
    path.write_text(
        "# Titre\n\n"
        f"{REGION_BEGIN.format(name=HEADLINE_REGION)}\n"
        f"{REGION_END.format(name=HEADLINE_REGION)}\n",
        encoding="utf-8",
    )
    return path


def rendered_for(fragments: Path) -> dict[Path, str]:
    """Render every table of a temporary campaign, the way the check does."""
    return build(
        experiments_dir=fragments.parent / "configs",
        results_dir=fragments.parent / "results",
        output_dir=fragments,
        readme=fragments.parent / "README.md",
    )


@pytest.fixture
def campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the check at a temporary campaign and return the fragments directory."""
    configs, results, fragments = (tmp_path / part for part in ("configs", "results", "generated"))
    declare(configs, "a_run")
    write_record(record("a_run"), run_directory(results, "a_run"))

    monkeypatch.setattr("scripts.check_sync.EXPERIMENTS_DIR", configs)
    monkeypatch.setattr("scripts.check_sync.RESULTS_DIR", results)
    monkeypatch.setattr("scripts.check_sync.FRAGMENTS_DIR", fragments)
    monkeypatch.setattr("scripts.check_sync.README", readme_at(tmp_path))
    return fragments


def test_a_machine_without_records_reports_pending_never_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # reports/results is gitignored, so this is what a CI runner sees. Passing
    # here would mean the check reports a match it never made, and
    # regenerating would rewrite every table of the report as NOT_RUN.
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "a_run")
    monkeypatch.setattr("scripts.check_sync.EXPERIMENTS_DIR", configs)
    monkeypatch.setattr("scripts.check_sync.RESULTS_DIR", results)

    status, message = check_report_freshness()

    assert status == PENDING
    assert "no run record" in message


def test_fragments_matching_the_records_pass(campaign: Path) -> None:
    write(rendered_for(campaign))

    status, message = check_report_freshness()

    assert status == OK
    assert "10 generated tables" in message


def test_a_file_whose_table_was_retyped_fails(campaign: Path) -> None:
    # The README is compared like a fragment even though it is injected rather
    # than included: it is the one page nothing else would catch, because
    # MkDocs never builds it and no strict mode ever sees it.
    write(rendered_for(campaign))
    readme = campaign.parent / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8").replace("0,3000", "0,9999"), "utf-8")

    status, message = check_report_freshness()

    assert status == FAILED
    assert "README.md" in message


def test_a_hand_edited_fragment_fails_with_the_command_to_run(campaign: Path) -> None:
    rendered = rendered_for(campaign)
    write(rendered)
    (campaign / "campaign.md").write_text("| Runs complets | 99 |\n", encoding="utf-8")

    status, message = check_report_freshness()

    assert status == FAILED
    assert "campaign.md" in message
    assert "make report-sync" in message


def test_a_never_generated_report_fails_rather_than_pending(campaign: Path) -> None:
    # The records are there and the fragments are not: that is a report which
    # has not been synchronised, not a machine that cannot check.
    status, _ = check_report_freshness()

    assert status == FAILED


def test_runs_measured_differently_fail_the_check(
    campaign: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs, results = campaign.parent / "configs", campaign.parent / "results"
    declare(configs, "b_run")
    write_record(record("b_run", beams=1), run_directory(results, "b_run"))
    monkeypatch.setattr("scripts.check_sync.EXPERIMENTS_DIR", configs)

    status, message = check_report_freshness()

    assert status == FAILED
    assert "not measured the same way" in message
