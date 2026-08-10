"""Typed configuration of the data pipeline.

The YAML files under ``configs/data/`` are validated by these models before any
download happens. A malformed configuration must fail immediately, with a
message naming the offending field, rather than halfway through a download.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetConfig(BaseModel):
    """Identification of the upstream corpus on the Hugging Face hub."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hf_id: str = Field(description="Dataset identifier, for example EdinburghNLP/xsum.")
    hf_config: str | None = Field(default=None, description="Optional dataset configuration name.")
    revision: str | None = Field(default=None, description="Commit or tag pinned for the download.")
    source_column: str = Field(description="Column holding the document to summarise.")
    target_column: str = Field(description="Column holding the reference summary.")
    id_column: str | None = Field(default=None, description="Column holding a stable example id.")


class WorkingCorpusConfig(BaseModel):
    """Size and seed of the frozen working corpus.

    The full upstream corpus is too large for the hardware budget of the
    project. Every reported score refers to this subset, never to the full
    dataset. Presenting one for the other would be a fabricated result.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(ge=0, description="Seed of the single draw that freezes the corpus.")
    train_size: int = Field(gt=0, description="Number of training examples kept.")
    validation_size: int = Field(gt=0, description="Number of validation examples kept.")
    test_size: int = Field(gt=0, description="Number of test examples kept.")


class PreprocessConfig(BaseModel):
    """Cleaning rules applied before any tokenisation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_source_chars: int = Field(ge=0, description="Documents shorter than this are dropped.")
    max_source_chars: int = Field(gt=0, description="Documents longer than this are dropped.")
    min_target_chars: int = Field(ge=0, description="Summaries shorter than this are dropped.")
    max_target_chars: int = Field(gt=0, description="Summaries longer than this are dropped.")
    normalise_unicode: bool = Field(default=True, description="Apply NFKC normalisation.")
    collapse_whitespace: bool = Field(default=True, description="Collapse runs of whitespace.")

    @model_validator(mode="after")
    def _check_bounds(self) -> PreprocessConfig:
        """Ensure the lower bounds stay below the upper bounds.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If a minimum is greater than or equal to its maximum.
        """
        if self.min_source_chars >= self.max_source_chars:
            raise ValueError("min_source_chars must be lower than max_source_chars.")
        if self.min_target_chars >= self.max_target_chars:
            raise ValueError("min_target_chars must be lower than max_target_chars.")
        return self


class TokenizerConfig(BaseModel):
    """Tokeniser shared by the from scratch model and the pretrained baseline.

    Sharing the tokeniser keeps the comparison clean: same vocabulary, same
    segmentation, same test set.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    hf_id: str = Field(description="Tokeniser identifier, for example t5-small.")
    source_prefix: str = Field(default="", description="Prefix prepended to every document.")
    max_source_tokens: int = Field(gt=0, description="Truncation length of the document.")
    max_target_tokens: int = Field(gt=0, description="Truncation length of the summary.")


class PathsConfig(BaseModel):
    """Filesystem layout of the pipeline outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw: Path = Field(description="Cache directory of the upstream download.")
    processed: Path = Field(description="Directory holding the frozen working corpus.")


class DataPipelineConfig(BaseModel):
    """Complete configuration of a data pipeline run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Short name of the corpus, used in file names.")
    dataset: DatasetConfig
    working_corpus: WorkingCorpusConfig
    proportions: list[int] = Field(description="Ablation proportions, in percent.")
    preprocess: PreprocessConfig
    tokenizer: TokenizerConfig
    paths: PathsConfig

    @model_validator(mode="after")
    def _check_proportions(self) -> DataPipelineConfig:
        """Validate the ablation proportions.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If the proportions are empty, out of range, duplicated,
                or if the largest one is not 100.
        """
        if not self.proportions:
            raise ValueError("At least one proportion is required.")
        if len(set(self.proportions)) != len(self.proportions):
            raise ValueError(f"Duplicate proportions: {self.proportions}.")
        if any(not 0 < value <= 100 for value in self.proportions):
            raise ValueError(f"Proportions must lie in (0, 100], got {self.proportions}.")
        if max(self.proportions) != 100:
            raise ValueError("The largest proportion must be 100, the full working corpus.")
        return self

    def proportion_sizes(self) -> dict[int, int]:
        """Return the number of training examples behind each proportion.

        Subsets are nested: the 10 percent subset is a prefix of the 50 percent
        subset, itself a prefix of the full corpus. Nesting isolates the effect
        of the corpus size from the effect of the sample composition.

        Returns:
            A mapping from percentage to example count, sorted by percentage.
        """
        total = self.working_corpus.train_size
        return {
            percentage: max(1, round(total * percentage / 100))
            for percentage in sorted(self.proportions)
        }


def load_pipeline_config(path: Path | str) -> DataPipelineConfig:
    """Load and validate a data pipeline configuration file.

    Args:
        path: Path to the YAML configuration.

    Returns:
        The validated configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the file does not contain a YAML mapping.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping.")

    return DataPipelineConfig.model_validate(raw)
