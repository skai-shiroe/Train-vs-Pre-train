"""Unit tests for the padding measurement script.

The padding figures behind the batching decision come from this script, so its arithmetic
is checked rather than trusted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.measure_padding import (
    batches_at_cap,
    main,
    measure,
    parse_args,
    render,
    shuffled_batches,
)


@pytest.mark.unit
def test_shuffled_batches_cover_the_corpus_once() -> None:
    batches = shuffled_batches(50, 8, seed=42)

    assert sorted(index for batch in batches for index in batch) == list(range(50))
    assert len(batches) == 7


@pytest.mark.unit
def test_the_shuffle_is_seeded() -> None:
    assert shuffled_batches(50, 8, seed=1) == shuffled_batches(50, 8, seed=1)
    assert shuffled_batches(50, 8, seed=1) != shuffled_batches(50, 8, seed=2)


@pytest.mark.unit
def test_a_batch_holding_a_truncated_document_counts_as_capped() -> None:
    lengths = [512, 30, 40, 50]

    assert batches_at_cap(lengths, [[0, 1], [2, 3]], 512) == (1, 2)


@pytest.mark.unit
def test_grouping_beats_shuffling_at_every_useful_batch_size() -> None:
    lengths = [512 if index % 3 == 0 else 40 + index % 200 for index in range(400)]

    rows = measure(lengths, cap=512, batch_sizes=[4, 8, 16], seed=42, mega_batch_factor=50)

    assert len(rows) == 3
    assert all(row["grouped_waste"] < row["shuffled_waste"] for row in rows)


@pytest.mark.unit
def test_a_batch_size_of_one_wastes_nothing_either_way() -> None:
    lengths = [512 if index % 3 == 0 else 40 + index for index in range(60)]

    row = measure(lengths, cap=512, batch_sizes=[1], seed=42, mega_batch_factor=50)[0]

    assert row["shuffled_waste"] == 0.0
    assert row["grouped_waste"] == 0.0


@pytest.mark.unit
def test_the_table_reports_the_measurement() -> None:
    rows = measure([512, 100, 200, 300], cap=512, batch_sizes=[2], seed=42, mega_batch_factor=2)

    table = render(rows, cap=512, examples=4)

    assert "4 documents, truncation at 512 source tokens" in table
    assert "%" in table


@pytest.mark.unit
def test_the_defaults_point_at_the_frozen_corpus() -> None:
    args = parse_args([])

    assert args.processed_dir == Path("data/processed/cnn_dailymail")
    assert args.split == "test"
    assert args.limit == 256


@pytest.mark.unit
def test_a_missing_corpus_is_reported_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--processed-dir", str(tmp_path / "absent")])

    assert code == 1
    assert "Measurement failed" in capsys.readouterr().err
