"""Length grouped batching, and the measurement that justifies it.

## The problem

The collator pads each batch to its own longest sequence, so the padding is
already dynamic. On XSum that is almost worthless on the source side. The
source truncation is 512 tokens and 38 percent of the documents reach it, so a
single long document in a batch pushes the whole batch to 512. Measured on 256
test documents, every batch hits the cap from batch size 8 upwards, and about a
quarter of the encoder positions carry padding:

```text
 batch    batches at cap   shuffled waste   grouped waste
------  ----------------  ---------------  --------------
     1         103 / 256            0.0 %           0.0 %
     2          84 / 128           17.0 %           0.6 %
     4           56 / 64           23.5 %           1.4 %
     8           31 / 32           24.8 %           1.5 %
    16           16 / 16           24.9 %           3.2 %
```

Dynamic padding degenerates into fixed padding exactly at the batch sizes the
training uses, and grouping recovers about a quarter of the encoder
computation. The table above is the verbatim output of:

```bash
python -m scripts.measure_padding --processed-dir data/processed/xsum
```

## The fix

Batching documents of similar length together brings the waste down to a few
percent. A global sort would do it, but it would also destroy the shuffling and
therefore the reproducibility of the ablation: the model would see the corpus
in length order at every epoch.

:class:`LengthGroupedSampler` keeps both properties:

```text
1. shuffle the corpus with the run seed and the epoch number
2. cut the shuffled order into windows of batch_size * mega_batch_factor
3. sort each window by length, longest first
4. cut each window into batches
5. shuffle the order of the batches, then move the widest batch back to the front
```

Randomness survives because the window content changes at every epoch. The
schedule stays reproducible because every draw comes from a seed derived from
the run seed and the epoch, never from the global generator.

Step 5 puts the widest batch first on purpose: the peak memory of the epoch is
paid on the first step, so a configuration that does not fit in the 8 GB budget
fails immediately instead of after twenty minutes of training.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Sequence

from torch.utils.data import DataLoader, Sampler
from transformers import PreTrainedTokenizerBase

from src.data.config import TokenizerConfig
from src.data.dataset import build_dataloader
from src.data.example import Example
from src.data.tokenize import source_token_lengths


class LengthGroupedSampler(Sampler[list[int]]):
    """Yield batches of indices whose documents have similar lengths."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        seed: int,
        mega_batch_factor: int = 50,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        """Build the sampler.

        Args:
            lengths: Token length of every example, in dataset order.
            batch_size: Number of examples per batch.
            seed: Run seed. Combined with the epoch to derive every draw.
            mega_batch_factor: Number of batches per sorting window. A larger
                window groups lengths more tightly but shuffles less.
            shuffle: Whether to shuffle. Set to ``False`` for evaluation, where
                the corpus order must stay stable across runs.
            drop_last: Whether to discard a trailing incomplete batch.

        Raises:
            ValueError: If a size is not strictly positive.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be strictly positive, got {batch_size}.")
        if mega_batch_factor <= 0:
            raise ValueError(
                f"mega_batch_factor must be strictly positive, got {mega_batch_factor}."
            )

        self._lengths = tuple(lengths)
        self._batch_size = batch_size
        self._seed = seed
        self._mega_batch_factor = mega_batch_factor
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch whose shuffling is replayed.

        The trainer calls this before every epoch. Two runs of the same seed
        therefore see the same batches in the same order, which is what makes
        the ablation on corpus size comparable.

        Args:
            epoch: Zero based epoch index.
        """
        self._epoch = epoch

    def batches(self) -> list[list[int]]:
        """Build the batches of the current epoch.

        Returns:
            One list of dataset indices per batch, in the order they are fed.
        """
        count = len(self._lengths)
        order = list(range(count))
        if not order:
            return []

        # A private generator, never the global one: the batching must not
        # depend on how many random numbers the rest of the run consumed.
        # nosec B311 - the draw orders training batches, it protects nothing.
        # A cryptographic generator would also be unseedable, which is the one
        # property this code needs.
        generator = random.Random(self._seed + self._epoch)  # nosec B311
        if self._shuffle:
            generator.shuffle(order)

        window = self._batch_size * self._mega_batch_factor
        batches: list[list[int]] = []
        for start in range(0, count, window):
            chunk = order[start : start + window]
            chunk.sort(key=lambda index: self._lengths[index], reverse=True)
            batches.extend(
                chunk[offset : offset + self._batch_size]
                for offset in range(0, len(chunk), self._batch_size)
            )

        if self._drop_last:
            batches = [batch for batch in batches if len(batch) == self._batch_size]

        if self._shuffle and batches:
            generator.shuffle(batches)
            widest = max(
                range(len(batches)),
                key=lambda position: max(self._lengths[index] for index in batches[position]),
            )
            batches[0], batches[widest] = batches[widest], batches[0]

        return batches

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate over the batches of the current epoch.

        Yields:
            One list of dataset indices per batch.
        """
        yield from self.batches()

    def __len__(self) -> int:
        """Return the number of batches in one epoch.

        Returns:
            The batch count, which the DataLoader reports as its length.
        """
        count = len(self._lengths)
        if self._drop_last:
            return count // self._batch_size
        return (count + self._batch_size - 1) // self._batch_size


def padding_waste(lengths: Sequence[int], batches: Iterable[Sequence[int]]) -> float:
    """Return the share of source positions that carry padding.

    Args:
        lengths: Token length of every example, in dataset order.
        batches: The batches to measure, as lists of dataset indices.

    Returns:
        A ratio in ``[0, 1]``. Zero means every position holds a real token.
    """
    real = 0
    allocated = 0
    for batch in batches:
        if not batch:
            continue
        widths = [lengths[index] for index in batch]
        real += sum(widths)
        allocated += max(widths) * len(widths)

    if allocated == 0:
        return 0.0
    return 1.0 - real / allocated


def build_training_dataloader(
    examples: Sequence[Example],
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_config: TokenizerConfig,
    *,
    batch_size: int,
    seed: int,
    group_by_length: bool = True,
    mega_batch_factor: int = 50,
    num_workers: int = 0,
) -> tuple[DataLoader[Example], LengthGroupedSampler | None]:
    """Build the training DataLoader, with length grouping when asked.

    Measuring the lengths costs one tokeniser pass over the training split.
    Only the integer lengths are kept, so the memory footprint stays negligible
    and the tokenised tensors are still produced batch by batch in the collator.

    Args:
        examples: Training examples.
        tokenizer: The shared tokeniser.
        tokenizer_config: Prefix and truncation lengths.
        batch_size: Number of examples per batch.
        seed: Run seed, used by the sampler.
        group_by_length: Whether to group documents of similar length.
        mega_batch_factor: Number of batches per sorting window.
        num_workers: Number of DataLoader worker processes.

    Returns:
        A pair holding the DataLoader and the sampler, or ``None`` when length
        grouping is disabled. The trainer needs the sampler to call
        :meth:`LengthGroupedSampler.set_epoch`.
    """
    if not group_by_length:
        loader = build_dataloader(
            examples,
            tokenizer,
            tokenizer_config,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        return loader, None

    lengths = source_token_lengths(examples, tokenizer, tokenizer_config)
    sampler = LengthGroupedSampler(
        lengths,
        batch_size,
        seed=seed,
        mega_batch_factor=mega_batch_factor,
        shuffle=True,
    )
    loader = build_dataloader(
        examples,
        tokenizer,
        tokenizer_config,
        num_workers=num_workers,
        batch_sampler=sampler,
    )
    return loader, sampler
