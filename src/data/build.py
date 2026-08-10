"""Build the frozen working corpus.

Entry point behind ``make data``:

```bash
python -m src.data.build --config configs/data/xsum.yaml
```

The command is idempotent: running it twice with the same configuration
produces byte identical files, and therefore the same ``dataset_version``.
That property is what makes the ablation reproducible.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.data.config import DataPipelineConfig, load_pipeline_config
from src.data.dataset import MANIFEST_NAME, write_jsonl
from src.data.download import download_corpus
from src.data.example import Example
from src.data.preprocess import preprocess
from src.data.split import (
    build_working_corpus,
    checksum,
    proportion_subset,
)
from src.data.statistics import compute_statistics, summarise_lengths
from src.data.validate import require_clean, require_no_leakage, validate_split

LOGGER = logging.getLogger("syntra.data.build")

STATISTICS_NAME = "statistics.json"


def _configure_logging(verbose: bool) -> None:
    """Configure the module logger.

    Args:
        verbose: Whether to emit debug level records.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


def _token_statistics(
    corpus_splits: dict[str, tuple[Example, ...]],
    config: DataPipelineConfig,
) -> dict[str, Any]:
    """Measure the untruncated token lengths of every split.

    The figures justify the truncation lengths: they tell how many documents
    the configured limit actually cuts.

    Args:
        corpus_splits: The frozen splits.
        config: Provides the tokeniser identifier and the truncation lengths.

    Returns:
        A mapping from split name to token length statistics, including the
        share of examples that the configured truncation cuts.
    """
    from src.data.tokenize import build_tokenizer, token_lengths

    tokenizer = build_tokenizer(config.tokenizer.hf_id)
    report: dict[str, Any] = {}

    for split, examples in corpus_splits.items():
        source_lengths, target_lengths = token_lengths(examples, tokenizer, config.tokenizer)
        truncated_sources = sum(
            1 for length in source_lengths if length > config.tokenizer.max_source_tokens
        )
        truncated_targets = sum(
            1 for length in target_lengths if length > config.tokenizer.max_target_tokens
        )
        report[split] = {
            "source_tokens": asdict(summarise_lengths(source_lengths)),
            "target_tokens": asdict(summarise_lengths(target_lengths)),
            "truncated_sources": truncated_sources,
            "truncated_sources_ratio": round(truncated_sources / len(examples), 4),
            "truncated_targets": truncated_targets,
            "truncated_targets_ratio": round(truncated_targets / len(examples), 4),
        }

    return report


def build(config: DataPipelineConfig, *, with_token_statistics: bool = True) -> dict[str, Any]:
    """Run the whole pipeline and write the frozen corpus to disk.

    Args:
        config: Validated pipeline configuration.
        with_token_statistics: Whether to measure token lengths, which requires
            downloading the tokeniser.

    Returns:
        The manifest, as a JSON serialisable mapping.
    """
    LOGGER.info("Downloading %s into %s", config.dataset.hf_id, config.paths.raw)
    upstream = download_corpus(config.dataset, config.paths.raw)
    for split, examples in upstream.items():
        LOGGER.info("Upstream split %s: %d examples", split, len(examples))

    LOGGER.info("Cleaning and filtering")
    cleaned: dict[str, tuple[Example, ...]] = {}
    for split, examples in upstream.items():
        kept = tuple(preprocess(examples, config.preprocess))
        dropped = len(examples) - len(kept)
        LOGGER.info(
            "Split %s: %d kept, %d dropped by the length bounds (%.2f%%)",
            split,
            len(kept),
            dropped,
            100 * dropped / len(examples) if examples else 0.0,
        )
        cleaned[split] = kept

    LOGGER.info("Drawing the working corpus with seed %d", config.working_corpus.seed)
    corpus = build_working_corpus(
        cleaned["train"],
        cleaned["validation"],
        cleaned["test"],
        train_size=config.working_corpus.train_size,
        validation_size=config.working_corpus.validation_size,
        test_size=config.working_corpus.test_size,
        seed=config.working_corpus.seed,
    )

    LOGGER.info("Validating the working corpus")
    splits = corpus.splits()
    for split, examples in splits.items():
        report = validate_split(split, examples)
        LOGGER.info("%s", report.describe())
        require_clean(report)
    require_no_leakage(corpus.train, corpus.validation, corpus.test)
    LOGGER.info("No leakage between train, validation and test")

    LOGGER.info("Writing the corpus to %s", config.paths.processed)
    config.paths.processed.mkdir(parents=True, exist_ok=True)
    for split, examples in splits.items():
        write_jsonl(config.paths.processed / f"{split}.jsonl", examples)

    proportion_sizes = config.proportion_sizes()
    proportion_checksums = {
        str(percentage): checksum(proportion_subset(corpus.train, percentage))
        for percentage in proportion_sizes
    }

    manifest: dict[str, Any] = {
        "name": config.name,
        "dataset_version": checksum(corpus.train + corpus.validation + corpus.test),
        "seed": corpus.seed,
        "splits": {split: len(examples) for split, examples in splits.items()},
        "split_checksums": {split: checksum(examples) for split, examples in splits.items()},
        "proportions": {str(key): value for key, value in proportion_sizes.items()},
        "proportion_checksums": proportion_checksums,
        "tokenizer": config.tokenizer.hf_id,
        "source_dataset": config.dataset.hf_id,
    }
    _write_json(config.paths.processed / MANIFEST_NAME, manifest)

    LOGGER.info("Computing statistics")
    statistics: dict[str, Any] = {
        "characters_and_words": {
            split: compute_statistics(split, examples).to_dict()
            for split, examples in splits.items()
        }
    }
    if with_token_statistics:
        statistics["tokens"] = _token_statistics(splits, config)
    _write_json(config.paths.processed / STATISTICS_NAME, statistics)

    LOGGER.info("dataset_version = %s", manifest["dataset_version"])
    return manifest


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a mapping as indented JSON with a trailing newline.

    Sorting the keys keeps the file byte stable across runs, which is what
    makes the drift detection of section 38.4 usable.

    Args:
        path: Destination file.
        payload: Mapping to serialise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line arguments.

    Args:
        argv: Argument list, defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.data.build",
        description="Build the frozen working corpus from the upstream dataset.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/xsum.yaml"),
        help="Path to the data pipeline configuration.",
    )
    parser.add_argument(
        "--skip-token-statistics",
        action="store_true",
        help="Skip the token length measurement, which requires the tokeniser.",
    )
    parser.add_argument("--verbose", action="store_true", help="Emit debug level logs.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline from the command line.

    Args:
        argv: Argument list, defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 on a handled failure.
    """
    args = parse_args(argv)
    _configure_logging(args.verbose)

    try:
        config = load_pipeline_config(args.config)
        build(config, with_token_statistics=not args.skip_token_statistics)
    except Exception as error:  # noqa: BLE001 - the CLI reports, it does not crash
        LOGGER.error("%s", error)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
