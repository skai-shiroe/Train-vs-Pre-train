"""Integration tests for the data pipeline orchestration.

The upstream download is replaced by a synthetic corpus, so these tests run
offline and in a fraction of a second. What they exercise is the wiring:
cleaning, drawing, validation, checksums, manifest and statistics.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from src.data import build as build_module
from src.data.build import build, main, parse_args
from src.data.config import DataPipelineConfig, DatasetConfig
from src.data.dataset import MANIFEST_NAME, load_manifest, load_split, load_training_proportion
from src.data.download import DownloadError, download_split
from src.data.example import Example
from src.data.validate import ValidationError

pytestmark = pytest.mark.integration


def synthetic_corpus(
    make_example: Callable[..., Example],
) -> dict[str, tuple[Example, ...]]:
    """Return a corpus large enough to draw the fixture working corpus from."""
    return {
        "train": tuple(make_example(index) for index in range(200)),
        "validation": tuple(make_example(1000 + index) for index in range(50)),
        "test": tuple(make_example(2000 + index) for index in range(50)),
    }


@pytest.fixture
def offline_build(
    monkeypatch: pytest.MonkeyPatch, make_example: Callable[..., Example]
) -> Callable[[DataPipelineConfig], dict[str, Any]]:
    """Return a build function whose download step is replaced by a fixture."""
    corpus = synthetic_corpus(make_example)
    monkeypatch.setattr(build_module, "download_corpus", lambda *_: corpus)

    def run(config: DataPipelineConfig) -> dict[str, Any]:
        return build(config, with_token_statistics=False)

    return run


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def test_build_writes_the_three_splits_and_the_manifest(
    offline_build: Callable[[DataPipelineConfig], dict[str, Any]],
    pipeline_config: DataPipelineConfig,
) -> None:
    offline_build(pipeline_config)

    processed = pipeline_config.paths.processed
    assert (processed / "train.jsonl").is_file()
    assert (processed / "validation.jsonl").is_file()
    assert (processed / "test.jsonl").is_file()
    assert (processed / MANIFEST_NAME).is_file()
    assert (processed / "statistics.json").is_file()


def test_the_splits_have_the_configured_sizes(
    offline_build: Callable[[DataPipelineConfig], dict[str, Any]],
    pipeline_config: DataPipelineConfig,
) -> None:
    manifest = offline_build(pipeline_config)

    assert manifest["splits"] == {"train": 20, "validation": 5, "test": 5}


def test_the_manifest_records_the_nested_proportions(
    offline_build: Callable[[DataPipelineConfig], dict[str, Any]],
    pipeline_config: DataPipelineConfig,
) -> None:
    manifest = offline_build(pipeline_config)

    assert manifest["proportions"] == {"10": 2, "50": 10, "100": 20}


def test_the_full_proportion_matches_the_training_split(
    offline_build: Callable[[DataPipelineConfig], dict[str, Any]],
    pipeline_config: DataPipelineConfig,
) -> None:
    manifest = offline_build(pipeline_config)

    assert manifest["proportion_checksums"]["100"] == manifest["split_checksums"]["train"]


def test_building_twice_produces_the_same_dataset_version(
    offline_build: Callable[[DataPipelineConfig], dict[str, Any]],
    pipeline_config: DataPipelineConfig,
) -> None:
    first = offline_build(pipeline_config)
    second = offline_build(pipeline_config)

    assert first["dataset_version"] == second["dataset_version"]


def test_a_different_seed_produces_a_different_corpus(
    offline_build: Callable[[DataPipelineConfig], dict[str, Any]],
    pipeline_config: DataPipelineConfig,
) -> None:
    first = offline_build(pipeline_config)

    other_seed = pipeline_config.model_copy(
        update={"working_corpus": pipeline_config.working_corpus.model_copy(update={"seed": 7})}
    )
    second = offline_build(other_seed)

    assert first["dataset_version"] != second["dataset_version"]


def test_the_written_corpus_reloads_identically(
    offline_build: Callable[[DataPipelineConfig], dict[str, Any]],
    pipeline_config: DataPipelineConfig,
) -> None:
    manifest = offline_build(pipeline_config)
    processed = pipeline_config.paths.processed

    assert len(load_split(processed, "train")) == manifest["splits"]["train"]
    assert load_manifest(processed).dataset_version == manifest["dataset_version"]


def test_loaded_proportions_are_prefixes_of_the_training_split(
    offline_build: Callable[[DataPipelineConfig], dict[str, Any]],
    pipeline_config: DataPipelineConfig,
) -> None:
    offline_build(pipeline_config)
    processed = pipeline_config.paths.processed

    train = load_split(processed, "train")
    ten = load_training_proportion(processed, 10)
    fifty = load_training_proportion(processed, 50)

    assert train[: len(ten)] == ten
    assert train[: len(fifty)] == fifty


def test_the_manifest_is_written_with_sorted_keys(
    offline_build: Callable[[DataPipelineConfig], dict[str, Any]],
    pipeline_config: DataPipelineConfig,
) -> None:
    offline_build(pipeline_config)
    raw = (pipeline_config.paths.processed / MANIFEST_NAME).read_text(encoding="utf-8")

    keys = list(json.loads(raw).keys())

    assert keys == sorted(keys)


def test_statistics_describe_every_split(
    offline_build: Callable[[DataPipelineConfig], dict[str, Any]],
    pipeline_config: DataPipelineConfig,
) -> None:
    offline_build(pipeline_config)
    raw = (pipeline_config.paths.processed / "statistics.json").read_text(encoding="utf-8")

    statistics = json.loads(raw)

    assert set(statistics["characters_and_words"]) == {"train", "validation", "test"}
    assert "tokens" not in statistics


def test_a_leaking_corpus_stops_the_build(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_config: DataPipelineConfig,
    make_example: Callable[..., Example],
) -> None:
    shared = tuple(make_example(index) for index in range(60))
    leaking = {"train": shared, "validation": shared, "test": shared}
    monkeypatch.setattr(build_module, "download_corpus", lambda *_: leaking)

    with pytest.raises(ValidationError, match="shared between"):
        build(pipeline_config, with_token_statistics=False)


def test_an_empty_summary_stops_the_build(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_config: DataPipelineConfig,
    make_example: Callable[..., Example],
) -> None:
    # Two adjustments make the failure deterministic. A zero minimum length
    # stops the cleaning stage from dropping the broken example, and a training
    # population of exactly train_size guarantees that the draw keeps it.
    permissive = pipeline_config.model_copy(
        update={"preprocess": pipeline_config.preprocess.model_copy(update={"min_target_chars": 0})}
    )
    train_size = permissive.working_corpus.train_size
    broken = (
        Example("empty-summary", "x" * 300, ""),
        *(make_example(index) for index in range(1, train_size)),
    )
    corpus = synthetic_corpus(make_example)
    monkeypatch.setattr(build_module, "download_corpus", lambda *_: {**corpus, "train": broken})

    with pytest.raises(ValidationError, match="empty summaries"):
        build(permissive, with_token_statistics=False)


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def test_the_default_configuration_is_the_cnn_dailymail_one() -> None:
    assert parse_args([]).config == Path("configs/data/cnn_dailymail.yaml")


def test_arguments_are_parsed() -> None:
    args = parse_args(["--config", "other.yaml", "--skip-token-statistics", "--verbose"])

    assert args.config == Path("other.yaml")
    assert args.skip_token_statistics is True
    assert args.verbose is True


def test_the_cli_reports_a_missing_configuration_without_crashing(tmp_path: Path) -> None:
    assert main(["--config", str(tmp_path / "absent.yaml")]) == 1


def test_the_cli_succeeds_on_a_valid_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pipeline_config: DataPipelineConfig,
    make_example: Callable[..., Example],
) -> None:
    monkeypatch.setattr(build_module, "download_corpus", lambda *_: synthetic_corpus(make_example))

    config_path = tmp_path / "pipeline.yaml"
    payload = pipeline_config.model_dump(mode="json")
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["--config", str(config_path), "--skip-token-statistics"]) == 0


# ---------------------------------------------------------------------------
# Download layer
# ---------------------------------------------------------------------------


def _dataset_config() -> DatasetConfig:
    return DatasetConfig(
        hf_id="fixture/corpus",
        source_column="document",
        target_column="summary",
        id_column="id",
    )


def test_rows_are_converted_into_examples(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rows = [
        {"id": "a", "document": "premier document", "summary": "premier resume"},
        {"id": "b", "document": "second document", "summary": "second resume"},
    ]
    monkeypatch.setattr("datasets.load_dataset", lambda *_args, **_kwargs: rows)

    examples = download_split(_dataset_config(), "train", tmp_path)

    assert [example.example_id for example in examples] == ["a", "b"]
    assert examples[0].source == "premier document"


def test_a_missing_identifier_column_falls_back_on_the_position(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows: Iterator[dict[str, str]] = iter(
        [{"document": "doc", "summary": "sum"}, {"document": "doc2", "summary": "sum2"}]
    )
    monkeypatch.setattr("datasets.load_dataset", lambda *_args, **_kwargs: rows)

    examples = download_split(_dataset_config(), "train", tmp_path)

    assert [example.example_id for example in examples] == ["train-0", "train-1"]


def test_a_renamed_column_upstream_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = [{"id": "a", "text": "doc", "summary": "sum"}]
    monkeypatch.setattr("datasets.load_dataset", lambda *_args, **_kwargs: rows)

    with pytest.raises(DownloadError, match="Available columns"):
        download_split(_dataset_config(), "train", tmp_path)


def test_a_download_failure_is_wrapped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise OSError("network unreachable")

    monkeypatch.setattr("datasets.load_dataset", explode)

    with pytest.raises(DownloadError, match="Check the network access"):
        download_split(_dataset_config(), "train", tmp_path)


def test_the_cache_directory_is_created(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("datasets.load_dataset", lambda *_args, **_kwargs: [])
    cache = tmp_path / "deep" / "cache"

    download_split(_dataset_config(), "train", cache)

    assert cache.is_dir()
