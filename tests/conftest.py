"""Shared fixtures for the test suite."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import T5Config, T5ForConditionalGeneration

from src.data.config import (
    DataPipelineConfig,
    DatasetConfig,
    PathsConfig,
    PreprocessConfig,
    TokenizerConfig,
    WorkingCorpusConfig,
)
from src.data.dataset import MANIFEST_NAME, SummarizationDataset, write_jsonl
from src.data.example import Example
from src.data.tokenize import IGNORE_INDEX, EncodedBatch

#: Vocabulary of the offline fixtures. Small enough for a unit test, large
#: enough that the identifiers derived from a text do not all collide.
FAKE_VOCAB_SIZE = 32


@pytest.fixture
def make_example() -> Callable[..., Example]:
    """Return a factory building examples with sensible defaults.

    Returns:
        A callable accepting an index and optional overrides.
    """

    def factory(index: int, *, source: str | None = None, target: str | None = None) -> Example:
        return Example(
            example_id=f"id-{index}",
            source=source if source is not None else f"Document number {index}. " * 20,
            target=target if target is not None else f"Summary number {index}.",
        )

    return factory


@pytest.fixture
def make_examples(make_example: Callable[..., Example]) -> Callable[[int], list[Example]]:
    """Return a factory building a list of distinct examples.

    Args:
        make_example: The single example factory.

    Returns:
        A callable accepting a count and returning that many examples.
    """

    def factory(count: int) -> list[Example]:
        return [make_example(index) for index in range(count)]

    return factory


class FakeCollator:
    """Turn examples into an encoded batch without downloading a tokeniser.

    The unit tests of the training loop need real ``EncodedBatch`` tensors, not
    a real vocabulary. Token identifiers are derived from the example index, so
    a batch is deterministic and the model has a learnable signal.
    """

    def __init__(self, *, vocab_size: int, max_source: int = 16, max_target: int = 6) -> None:
        self.vocab_size = vocab_size
        self.max_source = max_source
        self.max_target = max_target

    def _ids(self, example: Example, length: int, offset: int) -> list[int]:
        # A stable digest, not hash(): string hashing is randomised per process,
        # so a test asserting on the batches would not be reproducible.
        seed = sum(ord(character) for character in example.example_id) % self.vocab_size
        return [
            2 + (seed + offset + position) % (self.vocab_size - 2) for position in range(length)
        ]

    def __call__(self, batch: list[Example]) -> EncodedBatch:
        source_lengths = [
            min(self.max_source, 2 + len(example.source) % self.max_source) for example in batch
        ]
        target_lengths = [
            min(self.max_target, 2 + len(example.target) % self.max_target) for example in batch
        ]
        width = max(source_lengths)
        target_width = max(target_lengths)

        input_ids = torch.zeros(len(batch), width, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), width, dtype=torch.long)
        target_ids = torch.zeros(len(batch), target_width, dtype=torch.long)

        for row, example in enumerate(batch):
            source = self._ids(example, source_lengths[row], 0)
            target = self._ids(example, target_lengths[row], 1)
            input_ids[row, : len(source)] = torch.tensor(source)
            attention_mask[row, : len(source)] = 1
            target_ids[row, : len(target)] = torch.tensor(target)

        labels = target_ids.masked_fill(target_ids == 0, IGNORE_INDEX)
        return EncodedBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            target_ids=target_ids,
        )


@pytest.fixture
def make_loader(
    make_examples: Callable[[int], list[Example]],
) -> Callable[..., DataLoader[Example]]:
    """Return a factory building a DataLoader over fake encoded batches.

    Returns:
        A callable accepting an example count, a batch size and an optional
        batch sampler.
    """

    def factory(
        count: int,
        *,
        batch_size: int = 4,
        vocab_size: int = 32,
        batch_sampler: object | None = None,
    ) -> DataLoader[Example]:
        dataset = SummarizationDataset(make_examples(count))
        collate = FakeCollator(vocab_size=vocab_size)
        if batch_sampler is not None:
            return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate)  # type: ignore[arg-type]
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    return factory


class FakeTokenizer:
    """Encode and decode without downloading a vocabulary.

    Enough of the ``PreTrainedTokenizerBase`` surface for the pretrained
    baseline to run offline: one identifier per word, truncation, padding to the
    longest sequence of the batch, and a decoding that is readable in an
    assertion. Identifier ``0`` is padding and ``1`` ends a sequence, as in the
    T5 vocabulary.
    """

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, vocab_size: int = FAKE_VOCAB_SIZE) -> None:
        self.vocab_size = vocab_size
        self.calls: list[list[str]] = []
        self.saved_to: list[str] = []

    def __len__(self) -> int:
        # The experiment runner sizes the embedding table with len(), which
        # counts the tokens added after the vocabulary was built.
        return self.vocab_size

    def _ids(self, text: str, max_length: int) -> list[int]:
        words = text.split()
        ids = [
            2 + sum(ord(character) for character in word) % (self.vocab_size - 2) for word in words
        ]
        return [*ids[: max_length - 1], self.eos_token_id]

    def __call__(
        self,
        texts: list[str],
        *,
        max_length: int = 512,
        truncation: bool = True,
        padding: str = "longest",
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        self.calls.append(list(texts))
        rows = [self._ids(text, max_length) for text in texts]
        width = max(len(row) for row in rows)

        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        attention_mask = torch.zeros(len(rows), width, dtype=torch.long)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention_mask[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch_decode(self, sequences: torch.Tensor, skip_special_tokens: bool = True) -> list[str]:
        special = {self.pad_token_id, self.eos_token_id} if skip_special_tokens else set()
        return [
            " ".join(f"t{int(token)}" for token in row if int(token) not in special)
            for row in sequences
        ]

    def save_pretrained(self, directory: object) -> tuple[str, ...]:
        self.saved_to.append(str(directory))
        return (str(directory),)


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    """Return a tokeniser that needs no network."""
    return FakeTokenizer()


@pytest.fixture
def tiny_t5() -> Callable[..., T5ForConditionalGeneration]:
    """Return a factory building a T5 small enough for a unit test.

    The weights are random: what these tests check is the wiring around the
    model, never what it produces. Whether ``t5-small`` actually summarises is
    an integration concern.
    """

    def factory(vocab_size: int = FAKE_VOCAB_SIZE) -> T5ForConditionalGeneration:
        return T5ForConditionalGeneration(
            T5Config(
                vocab_size=vocab_size,
                d_model=16,
                d_ff=32,
                d_kv=8,
                num_layers=1,
                num_decoder_layers=1,
                num_heads=2,
                dropout_rate=0.0,
                pad_token_id=0,
                eos_token_id=1,
                decoder_start_token_id=0,
            )
        )

    return factory


@pytest.fixture
def pipeline_config(tmp_path_factory: pytest.TempPathFactory) -> DataPipelineConfig:
    """Return a small but valid pipeline configuration.

    Args:
        tmp_path_factory: Factory used to place the outputs outside the repo.

    Returns:
        A configuration usable by the offline parts of the pipeline.
    """
    root = tmp_path_factory.mktemp("corpus")
    return DataPipelineConfig(
        name="fixture",
        dataset=DatasetConfig(
            hf_id="fixture/corpus",
            source_column="document",
            target_column="summary",
            id_column="id",
        ),
        working_corpus=WorkingCorpusConfig(seed=42, train_size=20, validation_size=5, test_size=5),
        proportions=[10, 50, 100],
        preprocess=PreprocessConfig(
            min_source_chars=10,
            max_source_chars=10000,
            min_target_chars=5,
            max_target_chars=500,
        ),
        tokenizer=TokenizerConfig(
            hf_id="t5-small",
            source_prefix="summarize: ",
            max_source_tokens=512,
            max_target_tokens=64,
        ),
        paths=PathsConfig(raw=root / "raw", processed=root / "processed"),
    )


@pytest.fixture
def frozen_corpus(pipeline_config: DataPipelineConfig) -> DataPipelineConfig:
    """Write a small frozen corpus on disk and return the configuration locating it.

    The experiment runner reads the corpus through
    :mod:`src.data.dataset`, manifest included, so a test of the runner needs a
    real directory rather than a list of examples.

    Args:
        pipeline_config: The configuration whose paths the corpus is written to.

    Returns:
        The same configuration, now backed by files.
    """

    def block(count: int, offset: int) -> list[Example]:
        return [
            Example(
                example_id=f"id-{offset + index}",
                source=" ".join(f"word{(offset + index) % 5}{position}" for position in range(40)),
                target=f"Summary number {offset + index} of a short local report.",
            )
            for index in range(count)
        ]

    processed = pipeline_config.paths.processed
    sizes = {
        "train": pipeline_config.working_corpus.train_size,
        "validation": pipeline_config.working_corpus.validation_size,
        "test": pipeline_config.working_corpus.test_size,
    }
    for position, (split, count) in enumerate(sizes.items()):
        write_jsonl(processed / f"{split}.jsonl", block(count, position * 1000))

    (processed / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "name": pipeline_config.name,
                "dataset_version": "0" * 16,
                "seed": pipeline_config.working_corpus.seed,
                "splits": sizes,
                "split_checksums": dict.fromkeys(sizes, "0" * 16),
                "proportions": {
                    str(key): value for key, value in pipeline_config.proportion_sizes().items()
                },
                "proportion_checksums": {
                    str(key): "0" * 16 for key in pipeline_config.proportion_sizes()
                },
                "tokenizer": pipeline_config.tokenizer.hf_id,
                "source_dataset": pipeline_config.dataset.hf_id,
            }
        ),
        encoding="utf-8",
    )
    return pipeline_config


@pytest.fixture
def data_config_file(frozen_corpus: DataPipelineConfig, tmp_path: Path) -> Path:
    """Dump the corpus configuration to a YAML file an experiment can point at.

    Args:
        frozen_corpus: The configuration backed by files.
        tmp_path: Directory the file is written to.

    Returns:
        The path of the written configuration.
    """
    path = tmp_path / "corpus.yaml"
    path.write_text(
        yaml.safe_dump(frozen_corpus.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    return path
