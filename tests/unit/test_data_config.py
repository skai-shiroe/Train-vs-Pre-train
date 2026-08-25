"""Unit tests for the data pipeline configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.data.config import (
    DataPipelineConfig,
    PreprocessConfig,
    load_pipeline_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_CONFIG = REPO_ROOT / "configs" / "data" / "cnn_dailymail.yaml"


@pytest.mark.unit
def test_the_shipped_cnn_dailymail_configuration_is_valid() -> None:
    config = load_pipeline_config(CORPUS_CONFIG)

    assert config.name == "cnn_dailymail"
    assert config.dataset.hf_id == "abisee/cnn_dailymail"
    assert config.tokenizer.hf_id == "t5-small"
    assert config.working_corpus.seed == 42


@pytest.mark.unit
def test_the_dataset_configuration_name_is_pinned() -> None:
    """The hub publishes three versions of the corpus under one identifier.

    Only 3.0.0 is the non anonymised version the summarisation literature
    reports against. Leaving the field empty would let the hub choose, and a
    silent switch to an anonymised variant would change every score without
    changing a line of this repository.
    """
    config = load_pipeline_config(CORPUS_CONFIG)

    assert config.dataset.hf_config == "3.0.0"
    assert config.dataset.source_column == "article"
    assert config.dataset.target_column == "highlights"


@pytest.mark.unit
def test_the_shipped_configuration_matches_the_locked_decisions() -> None:
    config = load_pipeline_config(CORPUS_CONFIG)

    assert config.working_corpus.train_size == 20000
    assert config.working_corpus.validation_size == 1000
    assert config.working_corpus.test_size == 1000
    assert config.proportions == [10, 50, 100]


@pytest.mark.unit
def test_proportion_sizes_are_nested_and_exact(pipeline_config: DataPipelineConfig) -> None:
    sizes = pipeline_config.proportion_sizes()

    assert sizes == {10: 2, 50: 10, 100: 20}
    assert list(sizes.values()) == sorted(sizes.values())


@pytest.mark.unit
def test_a_proportion_never_collapses_to_zero(pipeline_config: DataPipelineConfig) -> None:
    tiny = pipeline_config.model_copy(
        update={
            "working_corpus": pipeline_config.working_corpus.model_copy(update={"train_size": 5})
        }
    )

    assert tiny.proportion_sizes()[10] == 1


@pytest.mark.unit
def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Configuration not found"):
        load_pipeline_config(tmp_path / "absent.yaml")


@pytest.mark.unit
def test_non_mapping_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        load_pipeline_config(path)


@pytest.mark.unit
def test_unknown_key_is_rejected(pipeline_config: DataPipelineConfig) -> None:
    payload = pipeline_config.model_dump()
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        DataPipelineConfig.model_validate(payload)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("proportions", "message"),
    [
        ([], "At least one proportion"),
        ([10, 10, 100], "Duplicate proportions"),
        ([0, 100], r"must lie in \(0, 100\]"),
        ([10, 120], r"must lie in \(0, 100\]"),
        ([10, 50], "largest proportion must be 100"),
    ],
)
def test_invalid_proportions_are_rejected(
    pipeline_config: DataPipelineConfig, proportions: list[int], message: str
) -> None:
    payload = pipeline_config.model_dump()
    payload["proportions"] = proportions

    with pytest.raises(ValidationError, match=message):
        DataPipelineConfig.model_validate(payload)


@pytest.mark.unit
def test_inverted_source_bounds_are_rejected() -> None:
    with pytest.raises(ValidationError, match="min_source_chars must be lower"):
        PreprocessConfig(
            min_source_chars=100,
            max_source_chars=50,
            min_target_chars=5,
            max_target_chars=100,
        )


@pytest.mark.unit
def test_inverted_target_bounds_are_rejected() -> None:
    with pytest.raises(ValidationError, match="min_target_chars must be lower"):
        PreprocessConfig(
            min_source_chars=10,
            max_source_chars=1000,
            min_target_chars=200,
            max_target_chars=100,
        )


@pytest.mark.unit
def test_configuration_is_immutable(pipeline_config: DataPipelineConfig) -> None:
    with pytest.raises(ValidationError):
        pipeline_config.name = "other"  # type: ignore[misc]
