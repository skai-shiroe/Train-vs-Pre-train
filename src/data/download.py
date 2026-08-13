"""Download of the upstream corpus.

The download is isolated in its own module so the rest of the pipeline can be
tested without any network access. Everything downstream consumes
:class:`~src.data.example.Example` objects, not Hugging Face datasets.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.data.config import DatasetConfig
from src.data.example import Example

#: Upstream split names, mapped to the names used inside the project.
UPSTREAM_SPLITS = {"train": "train", "validation": "validation", "test": "test"}


class DownloadError(Exception):
    """Raised when the upstream corpus cannot be obtained."""


def _rows_to_examples(rows: Any, config: DatasetConfig, split: str) -> Iterator[Example]:
    """Convert upstream rows into project examples.

    Args:
        rows: An iterable of mappings coming from the Hugging Face dataset.
        config: Column names of the upstream corpus.
        split: Name of the split, used to build a fallback identifier.

    Yields:
        One example per row.

    Raises:
        DownloadError: If an expected column is missing, which signals a schema
            change upstream.
    """
    for position, row in enumerate(rows):
        try:
            source = row[config.source_column]
            target = row[config.target_column]
        except KeyError as error:
            available = ", ".join(sorted(row.keys()))
            raise DownloadError(
                f"Column {error} is missing from {config.hf_id}. Available columns: {available}."
            ) from error

        if config.id_column and config.id_column in row:
            example_id = str(row[config.id_column])
        else:
            example_id = f"{split}-{position}"

        yield Example(example_id=example_id, source=str(source), target=str(target))


def download_split(
    config: DatasetConfig,
    split: str,
    cache_dir: Path,
) -> tuple[Example, ...]:
    """Download one split of the upstream corpus.

    Args:
        config: Identification of the upstream corpus.
        split: Split to download.
        cache_dir: Directory used by the Hugging Face cache.

    Returns:
        The examples of the split, in upstream order.

    Raises:
        DownloadError: If the split cannot be loaded, typically because the
            machine is offline and the cache is empty.
    """
    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset = load_dataset(
            config.hf_id,
            config.hf_config,
            split=split,
            revision=config.revision,
            cache_dir=str(cache_dir),
        )
    except Exception as error:  # noqa: BLE001 - upstream raises many unrelated types
        raise DownloadError(
            f"Could not download split {split!r} of {config.hf_id}: {error}. "
            "Check the network access and the dataset identifier."
        ) from error

    return tuple(_rows_to_examples(dataset, config, split))


def download_corpus(
    config: DatasetConfig,
    cache_dir: Path,
) -> dict[str, tuple[Example, ...]]:
    """Download the three splits of the upstream corpus.

    Args:
        config: Identification of the upstream corpus.
        cache_dir: Directory used by the Hugging Face cache.

    Returns:
        A mapping from project split name to examples.
    """
    return {
        project_split: download_split(config, upstream_split, cache_dir)
        for project_split, upstream_split in UPSTREAM_SPLITS.items()
    }
