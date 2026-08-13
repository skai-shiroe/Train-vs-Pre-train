"""Integration test of the chain behind ``make reproduce``.

The unit tests check what quick mode cuts and where it writes. What they cannot
check is that the five steps actually hold together: that the sweep leaves
records the aggregation can read, that the tables and the figures are produced
from those records rather than from the ones of the previous campaign, and that
the generated Markdown lands where it was aimed.

So this runs the whole chain in quick mode, on the synthetic corpus of the
fixtures and a Transformer small enough to train in a test. Nothing measured
here is a score, and the assertions say so: every record must come out
``PARTIAL`` and every score column must stay empty.

The last property is the one that matters most on a real machine: the chain
must write nothing outside the destinations it was given. A hard coded path in
any of the five steps would overwrite a campaign, and the guard below is the
only place that would catch it.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.data.config import DataPipelineConfig
from src.data.dataset import load_split
from src.data.statistics import compute_statistics, summarise_lengths
from src.experiments.ablation import DATASET_SIZE_CSV, EXPERIMENTS_CSV
from src.experiments.fragments import CAMPAIGN_FRAGMENT, DEFAULT_README, HEADLINE_FRAGMENT
from src.experiments.record import STATUS_PARTIAL, read_record, run_directory
from src.experiments.registry import DEFAULT_RESULTS_DIR
from src.experiments.reproduce import (
    DONE,
    MODE_QUICK,
    SKIPPED,
    Destinations,
    Settings,
    reproduce,
)

pytestmark = pytest.mark.integration

#: Splits the corpus record describes, in the order ``make data`` writes them.
SPLITS = ("train", "validation", "test")


def write_statistics(processed: Path) -> Path:
    """Write the statistics record ``make data`` leaves beside the corpus.

    The fixture corpus carries a manifest but no statistics, and the corpus
    page is generated from both. Building it here rather than widening the
    fixture keeps the character counts real: they are measured from the
    examples on disk, exactly as the pipeline measures them.

    Args:
        processed: Directory holding the frozen corpus.

    Returns:
        The path written.
    """
    token_lengths = asdict(summarise_lengths([12, 18, 24, 30]))
    statistics: dict[str, Any] = {
        "characters_and_words": {
            split: compute_statistics(split, load_split(processed, split)).to_dict()
            for split in SPLITS
        },
        "tokens": {
            split: {
                "source_tokens": token_lengths,
                "target_tokens": token_lengths,
                "truncated_sources": 0,
                "truncated_sources_ratio": 0.0,
                "truncated_targets": 0,
                "truncated_targets_ratio": 0.0,
            }
            for split in SPLITS
        },
    }
    path = processed / "statistics.json"
    path.write_text(json.dumps(statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def declare(directory: Path, data_config: Path, name: str) -> Path:
    """Write one experiment small enough for the chain to run in a test.

    Args:
        directory: Directory the experiment file goes to.
        data_config: The data pipeline configuration it reads its corpus from.
        name: Experiment name, which becomes its run directory.

    Returns:
        The path written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": name, "seed": 42, "studies": ["dataset_size"]},
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
                    "epochs": 5,
                    "batch_size": 8,
                    "learning_rate": 0.001,
                    "mixed_precision": False,
                    "device": "cpu",
                },
                "evaluation": {"split": "test", "no_repeat_ngram_size": 0},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def chain(
    frozen_corpus: DataPipelineConfig,
    data_config_file: Path,
    tmp_path: Path,
) -> Settings:
    """Return a quick invocation whose destinations are all under ``tmp_path``."""
    write_statistics(frozen_corpus.paths.processed)
    experiments = tmp_path / "experiments"
    declare(experiments, data_config_file, "scratch_100")

    output = tmp_path / "quick"
    return Settings(
        mode=MODE_QUICK,
        experiments_dir=experiments,
        data_config=data_config_file,
        destinations=Destinations(
            results=output / "results",
            runs=output / "runs",
            figures=(output / "figures",),
            fragments=output / "_generated",
            readme=output / "README.md",
            report=output / "RAPPORT.md",
        ),
        tracking=False,
    )


def test_the_five_steps_run_and_produce_what_the_next_one_reads(chain: Settings) -> None:
    results = reproduce(chain)

    assert [result.name for result in results] == [
        "data",
        "experiments",
        "tables",
        "figures",
        "documentation",
    ]
    assert [result.status for result in results] == [SKIPPED, DONE, DONE, DONE, DONE]


def test_every_record_of_a_quick_run_is_partial(
    chain: Settings, frozen_corpus: DataPipelineConfig
) -> None:
    reproduce(chain)

    record = read_record(run_directory(chain.destinations.results, "scratch_100", partial=True))

    assert record is not None
    assert record.status == STATUS_PARTIAL
    assert not record.measured
    assert record.evaluation is not None
    # The real size of the split is what the record carries, beside the handful
    # of documents the run was scored on. A record claiming a complete split is
    # the one thing quick mode must not write.
    assert record.evaluation["split_size"] == frozen_corpus.working_corpus.test_size


def test_the_tables_keep_the_row_and_leave_the_score_empty(chain: Settings) -> None:
    reproduce(chain)

    rows = list(
        csv.DictReader(
            (chain.destinations.results / EXPERIMENTS_CSV).read_text(encoding="utf-8").splitlines()
        )
    )

    assert [row["experiment"] for row in rows] == ["scratch_100"]
    assert rows[0]["status"] == STATUS_PARTIAL
    assert rows[0]["rougeL_f"] == ""
    assert (chain.destinations.results / DATASET_SIZE_CSV).is_file()


def test_the_four_figures_are_drawn_from_the_records_of_this_run(chain: Settings) -> None:
    reproduce(chain)

    drawn = sorted(path.name for path in chain.destinations.figures[0].glob("*.png"))

    assert len(drawn) == 4


def test_the_generated_tables_land_beside_the_run_rather_than_in_the_documentation(
    chain: Settings,
) -> None:
    reproduce(chain)

    headline = chain.destinations.fragments / HEADLINE_FRAGMENT
    campaign = chain.destinations.fragments / CAMPAIGN_FRAGMENT

    assert campaign.is_file()
    # Built from the records this chain just wrote, and saying what they are.
    assert STATUS_PARTIAL in headline.read_text(encoding="utf-8")
    assert chain.destinations.readme.is_file()


def test_the_chain_writes_nothing_outside_its_destinations(chain: Settings) -> None:
    readme = DEFAULT_README.read_bytes()
    campaign = (
        sorted(path.name for path in DEFAULT_RESULTS_DIR.iterdir())
        if DEFAULT_RESULTS_DIR.is_dir()
        else []
    )

    reproduce(chain)

    assert DEFAULT_README.read_bytes() == readme
    assert campaign == (
        sorted(path.name for path in DEFAULT_RESULTS_DIR.iterdir())
        if DEFAULT_RESULTS_DIR.is_dir()
        else []
    )
