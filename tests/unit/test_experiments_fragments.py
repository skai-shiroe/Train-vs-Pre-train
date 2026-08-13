"""Unit tests of the generated documentation tables.

The fragments exist so that no page can state the previous campaign. Six
properties carry that, and each of them is a way the mechanism would fail while
still producing a plausible page: a missing measurement must not render as a
number, a rendering must not depend on how the command was invoked, a hand edit
must be detected, the capitalisation rate must be counted rather than assumed,
a table placed on two pages must be one table, and the README must be refreshed
in place rather than included, because nothing builds it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.evaluation.evaluator import PREDICTIONS_FILE
from src.experiments.fragments import (
    ARCHITECTURE_FRAGMENT,
    BANNER,
    CAMPAIGN_FRAGMENT,
    CAPITALISATION_FRAGMENT,
    DATASET_SIZE_FRAGMENT,
    FAMILIES_FRAGMENT,
    HEADLINE_FRAGMENT,
    ARCHITECTURE_REGION,
    CAPITALISATION_REGION,
    FAMILIES_REGION,
    HEADLINE_REGION,
    METRICS_FRAGMENT,
    PLAN_FRAGMENT,
    PRETRAINED_FRAGMENT,
    REGION_BEGIN,
    REGION_END,
    best_name,
    build,
    capitalisation,
    capitalisation_table,
    injected,
    interval,
    main,
    number,
    relative,
    score,
    seconds,
    share,
    stale,
    write,
)
from src.experiments.record import (
    STATUS_FAILED,
    STATUS_OK,
    RunRecord,
    run_directory,
    write_record,
)
from src.experiments.registry import collect

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


def declare_zero_shot(directory: Path, name: str) -> Path:
    """Write one zero shot experiment file and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": name, "seed": 42, "studies": ["dataset_size"]},
                "model": {"type": "pretrained", "mode": "zero_shot"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def declare_fine_tuned(directory: Path, name: str, percentage: int) -> Path:
    """Write one fine tuning experiment file and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": name, "seed": 42, "studies": ["dataset_size"]},
                "dataset": {"percentage": percentage},
                "model": {"type": "pretrained", "mode": "fine_tuned"},
                "training": {"epochs": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def record(
    name: str,
    *,
    status: str = STATUS_OK,
    value: float = 0.3,
    beams: int = 4,
    variants: dict[str, float] | None = None,
) -> RunRecord:
    """Return a record carrying one ROUGE-L score, its interval, and any other variant."""
    scores: dict[str, Any] = {"rougeL": {"fmeasure": value}}
    intervals: dict[str, Any] = {"rougeL": {"low": value - 0.01, "high": value + 0.01}}
    for variant, other in (variants or {}).items():
        scores[variant] = {"fmeasure": other}
        intervals[variant] = {"low": other - 0.01, "high": other + 0.01}
    return RunRecord(
        experiment=name,
        status=status,
        config={},
        dataset={"train_examples": 2000, "version": "abc123"},
        model={"model": "scratch", "mode": "trained", "parameters": "1234567"},
        hardware={"gpu_name": "Test GPU", "torch_version": "2.13.0"},
        provenance={"git_commit": "deadbeef"},
        training={"duration_seconds": 61.4},
        evaluation={
            "split": "test",
            "split_size": 100,
            "generation": {"num_beams": beams},
            "rouge_config": {"seed": 42},
            "report": {"size": 100, "rouge": {"scores": scores, "intervals": intervals}},
        },
        duration_seconds=120.0,
    )


def write_predictions(results: Path, name: str, entries: list[dict[str, str]]) -> Path:
    """Write the predictions file of one run and return its path."""
    directory = run_directory(results, name)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PREDICTIONS_FILE
    path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )
    return path


def campaign(tmp_path: Path) -> tuple[Path, Path]:
    """Declare two measured runs and return the configuration and result roots."""
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "a_run")
    declare(configs, "b_run", dataset={"percentage": 100})
    write_record(record("a_run", value=0.20), run_directory(results, "a_run"))
    write_record(record("b_run", value=0.30), run_directory(results, "b_run"))
    return configs, results


def readme_at(directory: Path) -> Path:
    """Return a page carrying the headline markers, writing the stub if absent."""
    path = directory / "README.md"
    if not path.is_file():
        path.write_text(
            "# Titre\n\nCe qui précède le tableau.\n\n"
            f"{REGION_BEGIN.format(name=HEADLINE_REGION)}\n"
            f"{REGION_END.format(name=HEADLINE_REGION)}\n\n"
            "Ce qui le suit.\n",
            encoding="utf-8",
        )
    return path


def report_at(directory: Path) -> Path:
    """Return a page carrying the three report markers, writing the stub if absent."""
    path = directory / "RAPPORT.md"
    if not path.is_file():
        markers = "\n\n".join(
            f"{REGION_BEGIN.format(name=name)}\n{REGION_END.format(name=name)}"
            for name in (FAMILIES_REGION, ARCHITECTURE_REGION, CAPITALISATION_REGION)
        )
        path.write_text(
            f"# Rapport\n\nCe qui précède les tableaux.\n\n{markers}\n\nCe qui les suit.\n",
            encoding="utf-8",
        )
    return path


def render(tmp_path: Path, configs: Path, results: Path) -> dict[Path, str]:
    """Render every table into the temporary tree."""
    return build(
        experiments_dir=configs,
        results_dir=results,
        output_dir=tmp_path / "out",
        readme=readme_at(tmp_path),
        report=report_at(tmp_path),
    )


def arguments(tmp_path: Path, configs: Path, results: Path) -> list[str]:
    """Return the command line pointing at the temporary tree."""
    return [
        "--experiments",
        str(configs),
        "--results",
        str(results),
        "--output",
        str(tmp_path / "out"),
        "--readme",
        str(readme_at(tmp_path)),
        "--report",
        str(report_at(tmp_path)),
    ]


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


def test_numbers_are_rendered_the_way_the_report_reads_them() -> None:
    assert number(15591424) == "15 591 424"
    assert number("1234567") == "1 234 567"
    assert score(0.23104) == "0,2310"
    assert interval((0.2240, 0.2381)) == "[0,2240, 0,2381]"
    assert seconds(1346.6) == "1 347 s"
    assert share(629, 1000) == "62,9 %"


def test_an_absent_quantity_never_renders_as_a_value() -> None:
    # A dash or a zero in a score cell is a character a reader takes for a
    # measurement. Section 44 forbids exactly that.
    assert number(None) == ""
    assert score(None) == ""
    assert interval(None) == ""
    assert seconds(None) == ""
    assert share(0, 0) == ""


# ---------------------------------------------------------------------------
# What the tables carry
# ---------------------------------------------------------------------------


def test_a_finished_campaign_renders_without_a_status_column(tmp_path: Path) -> None:
    configs, results = campaign(tmp_path)

    rendered = render(tmp_path, configs, results)
    table = rendered[tmp_path / "out" / DATASET_SIZE_FRAGMENT]

    assert "Statut" not in table
    assert "| 0,2000 | [0,1900, 0,2100] |" in table


def test_an_unrun_experiment_keeps_its_row_and_raises_the_status_column(tmp_path: Path) -> None:
    configs, results = campaign(tmp_path)
    declare(configs, "c_run", dataset={"percentage": 50})

    rendered = render(tmp_path, configs, results)
    table = rendered[tmp_path / "out" / DATASET_SIZE_FRAGMENT]

    assert "Statut" in table
    # The row is present, its score cells are empty, and the status says why.
    assert "| `scratch` | 50 % |  |  |  | `NOT_RUN` |" in table


def test_a_failed_run_fills_no_score_cell(tmp_path: Path) -> None:
    configs, results = campaign(tmp_path)
    declare(configs, "c_run", dataset={"percentage": 50})
    write_record(record("c_run", status=STATUS_FAILED), run_directory(results, "c_run"))

    rendered = render(tmp_path, configs, results)
    table = rendered[tmp_path / "out" / DATASET_SIZE_FRAGMENT]

    assert "`FAILED`" in table
    assert "0,3000" not in table.split("`FAILED`")[0].splitlines()[-1]


def test_the_zero_shot_row_says_not_applicable_rather_than_zero(tmp_path: Path) -> None:
    # The baseline was not trained on nothing, it was not trained at all.
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare_zero_shot(configs, "zero_run")
    write_record(record("zero_run"), run_directory(results, "zero_run"))

    rendered = render(tmp_path, configs, results)
    table = rendered[tmp_path / "out" / DATASET_SIZE_FRAGMENT]

    assert "| `pretrained_zero_shot` | sans objet | sans objet |" in table
    assert "| 0 |" not in table


def test_the_architecture_table_reports_the_cost_beside_the_score(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(
        configs,
        "deep_run",
        experiment={"name": "deep_run", "seed": 42, "studies": ["architecture"]},
        model={"type": "scratch", "d_model": 32, "num_heads": 2, "encoder_layers": 6},
    )
    write_record(record("deep_run"), run_directory(results, "deep_run"))

    rendered = render(tmp_path, configs, results)
    table = rendered[tmp_path / "out" / ARCHITECTURE_FRAGMENT]

    assert "| `deep_run` | 6 + 4 | 1 234 567 | 0,3000 | [0,2900, 0,3100] | 61 s |" in table


def test_an_unrun_depth_keeps_its_row_in_the_architecture_table(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    architecture = {"name": "deep_run", "seed": 42, "studies": ["architecture"]}
    declare(configs, "deep_run", experiment=architecture)
    declare(
        configs,
        "deeper_run",
        experiment={**architecture, "name": "deeper_run"},
        model={"type": "scratch", "d_model": 32, "num_heads": 2, "encoder_layers": 6},
    )
    write_record(record("deep_run"), run_directory(results, "deep_run"))

    rendered = render(tmp_path, configs, results)
    table = rendered[tmp_path / "out" / ARCHITECTURE_FRAGMENT]

    assert "Statut" in table
    assert "| `deeper_run` | 6 + 4 |  |  |  |  | `NOT_RUN` |" in table


def test_the_campaign_stamp_reports_the_commit_and_the_corpus(tmp_path: Path) -> None:
    configs, results = campaign(tmp_path)

    rendered = render(tmp_path, configs, results)
    stamp = rendered[tmp_path / "out" / CAMPAIGN_FRAGMENT]

    assert "| Runs complets | 2 |" in stamp
    assert "| Statuts | `OK` 2 |" in stamp
    assert "| Calcul cumulé | 4 minutes |" in stamp
    assert "`abc123`" in stamp
    assert "`deadbeef`" in stamp


def test_the_plan_always_carries_its_statuses(tmp_path: Path) -> None:
    # The other tables raise the status column only when something is missing.
    # This one is the answer to how much of the plan is done, so a campaign
    # where everything passed still has to show what passed.
    configs, results = campaign(tmp_path)
    declare(configs, "c_run", dataset={"percentage": 50})

    rendered = render(tmp_path, configs, results)
    plan = rendered[tmp_path / "out" / PLAN_FRAGMENT]

    assert "| `a_run` | `scratch` | 10 % | taille du corpus | `OK` | 0,2000 |" in plan
    assert "| `c_run` | `scratch` | 50 % | taille du corpus | `NOT_RUN` |  |" in plan


def test_every_fragment_says_it_is_generated(tmp_path: Path) -> None:
    configs, results = campaign(tmp_path)
    readme, report = readme_at(tmp_path), report_at(tmp_path)

    rendered = render(tmp_path, configs, results)
    fragments = {path: text for path, text in rendered.items() if path not in (readme, report)}

    assert len(rendered) == 11
    assert len(fragments) == 9
    assert all(text.startswith(BANNER) and text.endswith("\n") for text in fragments.values())
    # The two pages are the targets that are not fragments. Their banner sits
    # inside each region, under the opening marker, so a page keeps its title
    # and its prose and still says which parts of it are generated.
    assert rendered[readme].startswith("# Titre")
    assert f"{REGION_BEGIN.format(name=HEADLINE_REGION)}\n{BANNER}" in rendered[readme]
    assert rendered[report].startswith("# Rapport")
    for name in (FAMILIES_REGION, ARCHITECTURE_REGION, CAPITALISATION_REGION):
        assert f"{REGION_BEGIN.format(name=name)}\n{BANNER}" in rendered[report]


# ---------------------------------------------------------------------------
# The tables the other pages carry
# ---------------------------------------------------------------------------


def test_the_metrics_table_reports_three_variants_and_one_interval(tmp_path: Path) -> None:
    # The interval belongs to the reported variant alone. Three intervals on a
    # row would read as three results rather than one score and two figures
    # supporting it.
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "a_run")
    write_record(
        record("a_run", value=0.23, variants={"rouge1": 0.29, "rouge2": 0.09}),
        run_directory(results, "a_run"),
    )

    rendered = render(tmp_path, configs, results)
    metrics = rendered[tmp_path / "out" / METRICS_FRAGMENT]

    assert "| Expérience | ROUGE-1 | ROUGE-2 | ROUGE-L | IC 95 % sur ROUGE-L |" in metrics
    assert "| `a_run` | 0,2900 | 0,0900 | **0,2300** | [0,2200, 0,2400] |" in metrics


def test_the_best_score_is_the_only_one_emphasised(tmp_path: Path) -> None:
    configs, results = campaign(tmp_path)

    rendered = render(tmp_path, configs, results)
    metrics = rendered[tmp_path / "out" / METRICS_FRAGMENT]

    assert "**0,3000**" in metrics
    assert "**0,2000**" not in metrics
    assert metrics.count("**") == 2


def test_an_unmeasured_row_is_never_the_best(tmp_path: Path) -> None:
    # max() over rows would otherwise have to compare a score with nothing, and
    # an empty cell in bold renders as four asterisks.
    configs, results = campaign(tmp_path)
    declare(configs, "c_run", dataset={"percentage": 50})
    rows = collect(configs, results)

    assert best_name(rows) == "b_run"
    assert best_name([row for row in rows if row.name == "c_run"]) is None


def test_the_baseline_table_starts_from_the_untouched_weights(tmp_path: Path) -> None:
    # The zero shot run comes first here and last in the ablation tables: there
    # it is a line under a curve, here it is what fine tuning moves away from.
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare_zero_shot(configs, "pretrained_zero_shot")
    declare_fine_tuned(configs, "pretrained_ft_100", 100)
    declare(configs, "scratch_100", dataset={"percentage": 100})
    write_record(
        record("pretrained_zero_shot", value=0.13), run_directory(results, "pretrained_zero_shot")
    )
    write_record(
        record("pretrained_ft_100", value=0.23), run_directory(results, "pretrained_ft_100")
    )
    write_record(record("scratch_100", value=0.16), run_directory(results, "scratch_100"))

    rendered = render(tmp_path, configs, results)
    baseline = rendered[tmp_path / "out" / PRETRAINED_FRAGMENT]

    assert "`scratch_100`" not in baseline
    assert baseline.index("`pretrained_zero_shot`") < baseline.index("`pretrained_ft_100`")
    assert "| `pretrained_zero_shot` | zero-shot | sans objet | 0,1300 |" in baseline
    assert "| `pretrained_ft_100` | fine-tuné | 100 % | 0,2300 |" in baseline


def test_the_families_table_writes_the_sign_of_the_gap(tmp_path: Path) -> None:
    # A column named for a gap must not let a regression read as a gain.
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "scratch_10", dataset={"percentage": 10})
    declare_fine_tuned(configs, "pretrained_ft_10", 10)
    declare(configs, "scratch_50", dataset={"percentage": 50})
    declare_fine_tuned(configs, "pretrained_ft_50", 50)
    write_record(record("scratch_10", value=0.1250), run_directory(results, "scratch_10"))
    write_record(
        record("pretrained_ft_10", value=0.1791), run_directory(results, "pretrained_ft_10")
    )
    write_record(record("scratch_50", value=0.2000), run_directory(results, "scratch_50"))
    write_record(
        record("pretrained_ft_50", value=0.1800), run_directory(results, "pretrained_ft_50")
    )

    rendered = render(tmp_path, configs, results)
    families = rendered[tmp_path / "out" / FAMILIES_FRAGMENT]

    assert "| 10 % | 2 000 | 0,1250 | 0,1791 | 0,0541 | +43 % |" in families
    assert "| 50 % | 2 000 | 0,2000 | 0,1800 | 0,0200 | -10 % |" in families


def test_the_families_table_distinguishes_an_unrun_run_from_an_undeclared_one(
    tmp_path: Path,
) -> None:
    # One is a hole in the campaign, the other a hole in the plan. Both leave
    # an empty cell, and an empty cell on its own is not a reason.
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "scratch_10", dataset={"percentage": 10})
    declare_fine_tuned(configs, "pretrained_ft_10", 10)
    declare(configs, "scratch_50", dataset={"percentage": 50})
    write_record(record("scratch_10", value=0.1250), run_directory(results, "scratch_10"))
    write_record(record("scratch_50", value=0.1550), run_directory(results, "scratch_50"))

    rendered = render(tmp_path, configs, results)
    families = rendered[tmp_path / "out" / FAMILIES_FRAGMENT]

    assert "| 10 % | 2 000 | 0,1250 |  |  |  | `pretrained_ft_10` `NOT_RUN` |" in families
    assert "| 50 % | 2 000 | 0,1550 |  |  |  | `t5-small` fine-tuné : non déclarée |" in families


def test_a_relative_gap_needs_both_sides(tmp_path: Path) -> None:
    assert relative(0.1250, 0.1791) == "+43 %"
    assert relative(None, 0.1791) == ""
    assert relative(0.1250, None) == ""
    assert relative(0.0, 0.1791) == ""


def test_the_headline_table_names_the_model_rather_than_the_experiment(tmp_path: Path) -> None:
    # A reader arriving on the index has not met the experiment names yet.
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare_zero_shot(configs, "pretrained_zero_shot")
    declare_fine_tuned(configs, "pretrained_ft_100", 100)
    declare(configs, "scratch_100", dataset={"percentage": 100})
    write_record(
        record("pretrained_zero_shot", value=0.13), run_directory(results, "pretrained_zero_shot")
    )
    write_record(
        record("pretrained_ft_100", value=0.23), run_directory(results, "pretrained_ft_100")
    )
    write_record(record("scratch_100", value=0.16), run_directory(results, "scratch_100"))

    rendered = render(tmp_path, configs, results)
    headline = rendered[tmp_path / "out" / HEADLINE_FRAGMENT]

    assert "pretrained_ft_100" not in headline
    assert "| `t5-small` fine-tuné | 100 % | **0,2300** | [0,2200, 0,2400] |" in headline
    assert "| from scratch | 100 % | 0,1600 | [0,1500, 0,1700] |" in headline
    assert "| `t5-small` zero-shot | sans objet | 0,1300 | [0,1200, 0,1400] |" in headline


def test_the_index_and_the_readme_carry_the_same_table(tmp_path: Path) -> None:
    # Two pages retyping one measurement is two chances to state the previous
    # campaign. They are rendered once and placed twice.
    configs, results = campaign(tmp_path)

    rendered = render(tmp_path, configs, results)
    headline = rendered[tmp_path / "out" / HEADLINE_FRAGMENT]

    assert headline.strip() in rendered[readme_at(tmp_path)]


# ---------------------------------------------------------------------------
# Injection, for the one page MkDocs does not build
# ---------------------------------------------------------------------------


def test_the_prose_around_an_injected_table_is_kept(tmp_path: Path) -> None:
    readme = readme_at(tmp_path)

    text = injected(readme, HEADLINE_REGION, "| Modèle |\n| --- |")

    assert text.startswith("# Titre\n\nCe qui précède le tableau.\n")
    assert text.endswith("Ce qui le suit.\n")
    assert "| Modèle |" in text


def test_injecting_twice_writes_the_same_page(tmp_path: Path) -> None:
    # The region is found by its markers, not by its length, so a second run
    # must replace the table rather than nest it under the first.
    readme = readme_at(tmp_path)

    once = injected(readme, HEADLINE_REGION, "| Un |\n| --- |")
    readme.write_text(once, encoding="utf-8")
    twice = injected(readme, HEADLINE_REGION, "| Un |\n| --- |")
    readme.write_text(twice, encoding="utf-8")

    assert twice == once
    assert twice.count(REGION_BEGIN.format(name=HEADLINE_REGION)) == 1
    assert injected(readme, HEADLINE_REGION, "| Deux |\n| --- |").count("| Un |") == 0


def test_a_page_without_markers_is_refused(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# Titre\n\nAucun marqueur.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="carries no `headline` region"):
        injected(readme, HEADLINE_REGION, "| Modèle |")


def test_markers_in_the_wrong_order_are_refused(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        f"{REGION_END.format(name=HEADLINE_REGION)}\n"
        f"{REGION_BEGIN.format(name=HEADLINE_REGION)}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="carries no `headline` region"):
        injected(readme, HEADLINE_REGION, "| Modèle |")


def test_a_missing_page_is_refused_rather_than_created(tmp_path: Path) -> None:
    # Writing the file would produce a README holding a table and nothing else.
    with pytest.raises(ValueError, match="has nowhere to go"):
        injected(tmp_path / "nowhere.md", HEADLINE_REGION, "| Modèle |")


# ---------------------------------------------------------------------------
# Comparability, inherited from the tables and the figures
# ---------------------------------------------------------------------------


def test_runs_measured_differently_are_refused(tmp_path: Path) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "a_run")
    declare(configs, "b_run")
    write_record(record("a_run", beams=4), run_directory(results, "a_run"))
    write_record(record("b_run", beams=1), run_directory(results, "b_run"))

    with pytest.raises(ValueError, match="not measured the same way"):
        render(tmp_path, configs, results)


def test_a_refused_study_leaves_no_fragment_half_written(tmp_path: Path) -> None:
    # Rendering and writing are split for this: a campaign that cannot be
    # tabulated must not leave the report holding one new table and three old.
    configs, results = tmp_path / "configs", tmp_path / "results"
    output = tmp_path / "out"
    declare(configs, "a_run")
    declare(configs, "b_run")
    write_record(record("a_run", beams=4), run_directory(results, "a_run"))
    write_record(record("b_run", beams=1), run_directory(results, "b_run"))
    readme = readme_at(tmp_path)
    untouched = readme.read_text(encoding="utf-8")

    with pytest.raises(ValueError):
        render(tmp_path, configs, results)

    assert not output.exists()
    assert readme.read_text(encoding="utf-8") == untouched


# ---------------------------------------------------------------------------
# Capitalisation, the one number recomputed here
# ---------------------------------------------------------------------------


def test_an_empty_prediction_counts_in_the_total_and_not_as_lowercase(tmp_path: Path) -> None:
    path = write_predictions(
        tmp_path,
        "a_run",
        [
            {"prediction": "people have been arrested", "reference": "Three people"},
            {"prediction": "Three people were arrested", "reference": "Three people"},
            {"prediction": "", "reference": "Three people"},
        ],
    )

    assert capitalisation(path) == (1, 0, 3)


def test_a_blank_line_is_not_a_prediction(tmp_path: Path) -> None:
    # A blank line at the end of the file would otherwise raise the total by
    # one and lower every rate on the page.
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"prediction": "people", "reference": "Three"}\n\n\n', encoding="utf-8")

    assert capitalisation(path) == (1, 0, 1)


def test_a_run_without_predictions_is_absent_rather_than_zero(tmp_path: Path) -> None:
    assert capitalisation(tmp_path / "nowhere.jsonl") is None


def test_the_capitalisation_table_carries_the_reference_control(tmp_path: Path) -> None:
    # Without the reference row the prediction rates are high against nothing.
    configs, results = campaign(tmp_path)
    write_predictions(
        results,
        "a_run",
        [
            {"prediction": "people arrested", "reference": "Three people"},
            {"prediction": "Three people", "reference": "Three people"},
        ],
    )

    table = capitalisation_table(collect(configs, results), results)

    assert "| `a_run` | 50,0 % |" in table
    assert "| Références | 0,0 % |" in table
    assert "`b_run`" not in table


def test_a_campaign_that_wrote_no_prediction_says_so(tmp_path: Path) -> None:
    configs, results = campaign(tmp_path)

    table = capitalisation_table(collect(configs, results), results)

    assert table.startswith("Aucun run complet")


# ---------------------------------------------------------------------------
# Freshness, which is what the whole mechanism is for
# ---------------------------------------------------------------------------


def test_the_rendering_does_not_depend_on_how_the_path_was_spelled(tmp_path: Path) -> None:
    # A fragment that differed between a relative and an absolute invocation
    # would be reported stale by the check on a machine where nothing changed.
    configs, results = campaign(tmp_path)

    spelled = build(
        experiments_dir=configs,
        results_dir=results,
        output_dir=tmp_path / "out",
        readme=readme_at(tmp_path),
        report=report_at(tmp_path),
    )
    resolved = build(
        experiments_dir=configs.resolve(),
        results_dir=results.resolve(),
        output_dir=tmp_path / "out",
        readme=readme_at(tmp_path).resolve(),
        report=report_at(tmp_path).resolve(),
    )

    assert list(spelled.values()) == list(resolved.values())


def test_a_freshly_written_set_is_not_stale(tmp_path: Path) -> None:
    configs, results = campaign(tmp_path)
    rendered = render(tmp_path, configs, results)

    written = write(rendered)

    assert len(written) == 11
    assert stale(rendered) == []


def test_a_hand_edited_fragment_is_reported_stale(tmp_path: Path) -> None:
    configs, results = campaign(tmp_path)
    rendered = render(tmp_path, configs, results)
    write(rendered)
    edited = tmp_path / "out" / CAPITALISATION_FRAGMENT
    edited.write_text("| Modèle | 0,9999 % |\n", encoding="utf-8")

    assert stale(rendered) == [edited]


def test_a_missing_fragment_is_stale_rather_than_absent(tmp_path: Path) -> None:
    configs, results = campaign(tmp_path)
    rendered = render(tmp_path, configs, results)

    assert len(stale(rendered)) == 11


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def test_check_mode_writes_nothing_and_fails_on_a_stale_fragment(tmp_path: Path) -> None:
    configs, results = campaign(tmp_path)
    output = tmp_path / "out"
    command = arguments(tmp_path, configs, results)
    untouched = readme_at(tmp_path).read_text(encoding="utf-8")

    assert main([*command, "--check"]) == 1
    assert not output.exists()
    assert readme_at(tmp_path).read_text(encoding="utf-8") == untouched

    assert main(command) == 0
    assert main([*command, "--check"]) == 0


def test_a_refused_study_exits_with_the_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configs, results = tmp_path / "configs", tmp_path / "results"
    declare(configs, "a_run")
    declare(configs, "b_run")
    write_record(record("a_run", beams=4), run_directory(results, "a_run"))
    write_record(record("b_run", beams=1), run_directory(results, "b_run"))

    code = main(arguments(tmp_path, configs, results))

    assert code == 1
    assert "not measured the same way" in capsys.readouterr().err


def test_a_page_without_a_region_exits_with_the_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A page that lost its markers must not be silently skipped: it would keep
    # showing the previous campaign while the check reported everything fresh.
    configs, results = campaign(tmp_path)
    command = arguments(tmp_path, configs, results)
    readme_at(tmp_path).write_text("# Titre\n\nPlus aucun marqueur.\n", encoding="utf-8")

    code = main(command)

    assert code == 1
    assert "carries no `headline` region" in capsys.readouterr().err
