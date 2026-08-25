"""Measure how much of the encoder computation goes into padding.

The collator pads each batch to its own longest sequence, so the padding is
already dynamic. This script answers the question that matters next: does that
actually save anything on the corpus and at the batch sizes the training uses?

It compares two batching strategies on the same documents:

```text
shuffled   what a plain shuffling sampler produces
grouped    what src.training.sampler.LengthGroupedSampler produces
```

Run it against the frozen corpus:

```bash
python -m scripts.measure_padding --processed-dir data/processed/cnn_dailymail
```

The padding figures behind the batching decision come from this command. Nothing
in the table is typed by hand.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.config import load_pipeline_config  # noqa: E402
from src.data.dataset import load_split  # noqa: E402
from src.data.tokenize import build_tokenizer, source_token_lengths  # noqa: E402
from src.training.sampler import LengthGroupedSampler, padding_waste  # noqa: E402

DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16)


def shuffled_batches(count: int, batch_size: int, *, seed: int) -> list[list[int]]:
    """Return the batches a plain shuffling sampler produces.

    Args:
        count: Number of examples.
        batch_size: Number of examples per batch.
        seed: Seed of the shuffle.

    Returns:
        One list of indices per batch.
    """
    order = list(range(count))
    random.Random(seed).shuffle(order)
    return [order[start : start + batch_size] for start in range(0, count, batch_size)]


def batches_at_cap(
    lengths: Sequence[int], batches: Sequence[Sequence[int]], cap: int
) -> tuple[int, int]:
    """Count the batches whose width reaches the truncation length.

    A batch at the cap costs exactly as much as fixed padding would: dynamic
    padding saved nothing on it.

    Args:
        lengths: Token length of every example.
        batches: The batches to measure.
        cap: The truncation length.

    Returns:
        A pair ``(at_cap, total)``.
    """
    at_cap = sum(1 for batch in batches if max(lengths[index] for index in batch) >= cap)
    return at_cap, len(batches)


def measure(
    lengths: Sequence[int],
    *,
    cap: int,
    batch_sizes: Sequence[int],
    seed: int,
    mega_batch_factor: int,
) -> list[dict[str, object]]:
    """Measure both strategies at every batch size.

    Args:
        lengths: Token length of every example.
        cap: The truncation length.
        batch_sizes: Batch sizes to measure.
        seed: Seed of the shuffling and of the sampler.
        mega_batch_factor: Number of batches per sorting window.

    Returns:
        One row per batch size.
    """
    rows: list[dict[str, object]] = []
    for batch_size in batch_sizes:
        shuffled = shuffled_batches(len(lengths), batch_size, seed=seed)
        grouped = list(
            LengthGroupedSampler(
                lengths, batch_size, seed=seed, mega_batch_factor=mega_batch_factor
            )
        )
        at_cap, total = batches_at_cap(lengths, shuffled, cap)
        rows.append(
            {
                "batch_size": batch_size,
                "batches_at_cap": at_cap,
                "batches": total,
                "shuffled_waste": padding_waste(lengths, shuffled),
                "grouped_waste": padding_waste(lengths, grouped),
            }
        )
    return rows


def render(rows: Sequence[dict[str, object]], *, cap: int, examples: int) -> str:
    """Render the measurement as a fixed width table.

    Args:
        rows: Rows produced by :func:`measure`.
        cap: The truncation length.
        examples: Number of documents measured.

    Returns:
        The table, ready to be printed or pasted into the documentation.
    """
    lines = [
        f"{examples} documents, truncation at {cap} source tokens",
        "",
        f"{'batch':>6}  {'batches at cap':>16}  {'shuffled waste':>15}  {'grouped waste':>14}",
        f"{'-' * 6}  {'-' * 16}  {'-' * 15}  {'-' * 14}",
    ]
    for row in rows:
        at_cap = f"{row['batches_at_cap']} / {row['batches']}"
        shuffled = f"{float(row['shuffled_waste']) * 100:.1f} %"  # type: ignore[arg-type]
        grouped = f"{float(row['grouped_waste']) * 100:.1f} %"  # type: ignore[arg-type]
        lines.append(f"{row['batch_size']:>6}  {at_cap:>16}  {shuffled:>15}  {grouped:>14}")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line arguments.

    Args:
        argv: Argument list, defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m scripts.measure_padding",
        description="Measure the share of encoder positions that carry padding.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/cnn_dailymail"),
        help="Directory holding the frozen corpus.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/cnn_dailymail.yaml"),
        help="Data pipeline configuration, read for the tokeniser and the truncation.",
    )
    parser.add_argument("--split", default="test", help="Split to measure.")
    parser.add_argument("--limit", type=int, default=256, help="Number of documents measured.")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the batching.")
    parser.add_argument(
        "--mega-batch-factor", type=int, default=50, help="Batches per sorting window."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the measurement from the command line.

    Args:
        argv: Argument list, defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 on a handled failure.
    """
    args = parse_args(argv)

    try:
        config = load_pipeline_config(args.config)
        examples = load_split(args.processed_dir, args.split)[: args.limit]
        tokenizer = build_tokenizer(config.tokenizer.hf_id)
        lengths = source_token_lengths(examples, tokenizer, config.tokenizer)
    except (FileNotFoundError, ValueError) as error:
        print(f"Measurement failed: {error}", file=sys.stderr)
        return 1

    rows = measure(
        lengths,
        cap=config.tokenizer.max_source_tokens,
        batch_sizes=DEFAULT_BATCH_SIZES,
        seed=args.seed,
        mega_batch_factor=args.mega_batch_factor,
    )
    print(render(rows, cap=config.tokenizer.max_source_tokens, examples=len(lengths)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
