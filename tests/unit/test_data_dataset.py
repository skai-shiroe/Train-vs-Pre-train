"""Unit tests for reading back the frozen corpus."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from src.data.dataset import (
    MANIFEST_NAME,
    SummarizationDataset,
    load_manifest,
    load_split,
    load_training_proportion,
    read_jsonl,
    write_jsonl,
)
from src.data.example import Example
from src.data.split import checksum


@pytest.fixture
def corpus_dir(tmp_path: Path, make_examples: Callable[[int], list[Example]]) -> Path:
    """Write a small frozen corpus and return its directory."""
    examples = make_examples(20)
    write_jsonl(tmp_path / "train.jsonl", examples)
    write_jsonl(tmp_path / "validation.jsonl", examples[:5])
    write_jsonl(tmp_path / "test.jsonl", examples[5:10])

    manifest = {
        "name": "fixture",
        "dataset_version": checksum(examples),
        "seed": 42,
        "splits": {"train": 20, "validation": 5, "test": 5},
        "split_checksums": {"train": checksum(examples)},
        "proportions": {"10": 2, "50": 10, "100": 20},
        "proportion_checksums": {"10": checksum(examples[:2])},
        "tokenizer": "t5-small",
        "source_dataset": "fixture/corpus",
    }
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


@pytest.mark.unit
def test_jsonl_roundtrip_preserves_order_and_content(
    tmp_path: Path, make_examples: Callable[[int], list[Example]]
) -> None:
    examples = make_examples(10)
    path = tmp_path / "split.jsonl"

    write_jsonl(path, examples)

    assert list(read_jsonl(path)) == examples


@pytest.mark.unit
def test_jsonl_survives_non_ascii(tmp_path: Path) -> None:
    example = Example("id-accent", "Le resume francais, ecrit avec des accents : ete.", "Resume.")
    path = tmp_path / "split.jsonl"

    write_jsonl(path, [example])

    assert list(read_jsonl(path)) == [example]


@pytest.mark.unit
def test_jsonl_writing_creates_missing_directories(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "split.jsonl"

    write_jsonl(path, [Example("a", "b", "c")])

    assert path.is_file()


@pytest.mark.unit
def test_reading_a_missing_file_points_at_the_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make data"):
        list(read_jsonl(tmp_path / "absent.jsonl"))


@pytest.mark.unit
def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "split.jsonl"
    path.write_text(
        json.dumps({"id": "a", "source": "s", "target": "t"}) + "\n\n\n",
        encoding="utf-8",
    )

    assert len(list(read_jsonl(path))) == 1


@pytest.mark.unit
def test_load_split_returns_the_stored_order(corpus_dir: Path) -> None:
    train = load_split(corpus_dir, "train")

    assert len(train) == 20
    assert train[0].example_id == "id-0"


@pytest.mark.unit
def test_an_unknown_split_is_refused(corpus_dir: Path) -> None:
    with pytest.raises(ValueError, match="Split must be one of"):
        load_split(corpus_dir, "dev")


@pytest.mark.unit
def test_manifest_is_loaded(corpus_dir: Path) -> None:
    manifest = load_manifest(corpus_dir)

    assert manifest.seed == 42
    assert manifest.tokenizer == "t5-small"
    assert manifest.proportions == {"10": 2, "50": 10, "100": 20}


@pytest.mark.unit
def test_a_missing_manifest_points_at_the_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make data"):
        load_manifest(tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize(("percentage", "expected"), [(10, 2), (50, 10), (100, 20)])
def test_proportions_load_the_announced_number_of_examples(
    corpus_dir: Path, percentage: int, expected: int
) -> None:
    assert len(load_training_proportion(corpus_dir, percentage)) == expected


@pytest.mark.unit
def test_loaded_proportions_are_nested(corpus_dir: Path) -> None:
    ten = load_training_proportion(corpus_dir, 10)
    fifty = load_training_proportion(corpus_dir, 50)

    assert fifty[: len(ten)] == ten


@pytest.mark.unit
def test_an_unknown_proportion_lists_the_available_ones(corpus_dir: Path) -> None:
    with pytest.raises(KeyError, match="Available"):
        load_training_proportion(corpus_dir, 25)


@pytest.mark.unit
def test_dataset_exposes_length_and_indexing(
    make_examples: Callable[[int], list[Example]],
) -> None:
    examples = make_examples(7)
    dataset = SummarizationDataset(examples)

    assert len(dataset) == 7
    assert dataset[3] == examples[3]


@pytest.mark.unit
def test_dataset_does_not_alias_the_input_list(
    make_examples: Callable[[int], list[Example]],
) -> None:
    examples = make_examples(3)
    dataset = SummarizationDataset(examples)

    examples.clear()

    assert len(dataset) == 3
