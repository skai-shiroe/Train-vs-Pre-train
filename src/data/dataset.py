"""Access to the frozen working corpus.

This module is the only entry point used by the training and evaluation code.
Nothing downstream reads a JSONL file by hand, so the on disk layout can change
without touching the experiments.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import PreTrainedTokenizerBase

from src.data.config import TokenizerConfig
from src.data.example import Example
from src.data.tokenize import EncodedBatch, encode_batch

MANIFEST_NAME = "manifest.json"
SPLIT_NAMES = ("train", "validation", "test")


def write_jsonl(path: Path, examples: Sequence[Example]) -> None:
    """Write examples as JSON Lines, one object per line.

    Args:
        path: Destination file. Parent directories are created.
        examples: Examples to write, in the order they must be read back.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[Example]:
    """Read examples from a JSON Lines file.

    Args:
        path: Source file.

    Yields:
        One example per non empty line, in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {path}. Run: make data")

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield Example.from_dict(json.loads(stripped))


@dataclass(frozen=True, slots=True)
class Manifest:
    """Description of a frozen corpus, written next to the splits.

    Attributes:
        name: Short name of the corpus.
        dataset_version: Checksum of the whole working corpus, traced in MLflow.
        seed: Seed of the draw that froze the corpus.
        splits: Number of examples per split.
        split_checksums: Checksum per split.
        proportions: Number of training examples per ablation percentage.
        proportion_checksums: Checksum per ablation percentage.
        tokenizer: Identifier of the shared tokeniser.
        source_dataset: Identifier of the upstream corpus.
    """

    name: str
    dataset_version: str
    seed: int
    splits: dict[str, int]
    split_checksums: dict[str, str]
    proportions: dict[str, int]
    proportion_checksums: dict[str, str]
    tokenizer: str
    source_dataset: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Manifest:
        """Rebuild a manifest from its serialised form.

        Args:
            payload: Mapping loaded from ``manifest.json``.

        Returns:
            The reconstructed manifest.
        """
        return cls(**payload)


def load_manifest(processed_dir: Path) -> Manifest:
    """Load the manifest of a frozen corpus.

    Args:
        processed_dir: Directory holding the splits and the manifest.

    Returns:
        The manifest.

    Raises:
        FileNotFoundError: If the manifest does not exist.
    """
    path = processed_dir / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}. Run: make data")

    return Manifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_split(processed_dir: Path, split: str) -> tuple[Example, ...]:
    """Load one split of the frozen corpus.

    Args:
        processed_dir: Directory holding the splits.
        split: One of ``train``, ``validation`` or ``test``.

    Returns:
        The examples of the split, in stored order.

    Raises:
        ValueError: If the split name is unknown.
    """
    if split not in SPLIT_NAMES:
        raise ValueError(f"Split must be one of {SPLIT_NAMES}, got {split!r}.")

    return tuple(read_jsonl(processed_dir / f"{split}.jsonl"))


def load_training_proportion(processed_dir: Path, percentage: int) -> tuple[Example, ...]:
    """Load the training subset of a given ablation proportion.

    Subsets are prefixes of the stored training split, so they are nested by
    construction and no extra file is needed.

    Args:
        processed_dir: Directory holding the splits and the manifest.
        percentage: Requested proportion, in percent.

    Returns:
        The training examples of that proportion.

    Raises:
        KeyError: If the proportion is absent from the manifest.
    """
    manifest = load_manifest(processed_dir)
    key = str(percentage)
    if key not in manifest.proportions:
        available = ", ".join(sorted(manifest.proportions))
        raise KeyError(f"Proportion {percentage} is not in the manifest. Available: {available}.")

    size = manifest.proportions[key]
    return load_split(processed_dir, "train")[:size]


class SummarizationDataset(Dataset[Example]):
    """A read only view over a sequence of examples."""

    def __init__(self, examples: Sequence[Example]) -> None:
        """Wrap a sequence of examples.

        Args:
            examples: Examples to expose, in their stored order.
        """
        self._examples = tuple(examples)

    def __len__(self) -> int:
        """Return the number of examples.

        Returns:
            The dataset size.
        """
        return len(self._examples)

    def __getitem__(self, index: int) -> Example:
        """Return the example at a given position.

        Args:
            index: Zero based position.

        Returns:
            The example stored at that position.
        """
        return self._examples[index]


class Collator:
    """Turn a list of examples into a tokenised batch.

    Tokenising in the collator rather than up front keeps the memory footprint
    proportional to the batch, not to the corpus, and lets the DataLoader
    workers share the work.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, config: TokenizerConfig) -> None:
        """Build the collator.

        Args:
            tokenizer: The shared tokeniser.
            config: Prefix and truncation lengths.
        """
        self._tokenizer = tokenizer
        self._config = config

    def __call__(self, batch: list[Example]) -> EncodedBatch:
        """Encode one batch.

        Args:
            batch: Examples selected by the sampler.

        Returns:
            The encoded batch.
        """
        return encode_batch(batch, self._tokenizer, self._config)


def build_dataloader(
    examples: Sequence[Example],
    tokenizer: PreTrainedTokenizerBase,
    config: TokenizerConfig,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    batch_sampler: Sampler[list[int]] | None = None,
) -> DataLoader[Example]:
    """Build a DataLoader over a sequence of examples.

    Args:
        examples: Examples to iterate over.
        tokenizer: The shared tokeniser.
        config: Prefix and truncation lengths.
        batch_size: Number of examples per batch. Ignored when
            ``batch_sampler`` is supplied, since the sampler already emits
            whole batches.
        shuffle: Whether to reshuffle at every epoch. Always ``False`` for
            validation and test, so that batches stay comparable across runs.
            Ignored when ``batch_sampler`` is supplied.
        num_workers: Number of worker processes.
        batch_sampler: Optional sampler yielding lists of indices. Used by the
            training loop to group documents of similar length, which cuts the
            padding waste of the encoder.

    Returns:
        The configured DataLoader.
    """
    from src.utils.seed import seed_worker

    dataset = SummarizationDataset(examples)
    collate = Collator(tokenizer, config)
    worker_init = seed_worker if num_workers > 0 else None

    if batch_sampler is not None:
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate,
            worker_init_fn=worker_init,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        worker_init_fn=worker_init,
        drop_last=False,
    )
