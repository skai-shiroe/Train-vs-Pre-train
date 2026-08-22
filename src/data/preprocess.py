"""Text cleaning applied before validation and tokenisation.

The rules are deliberately conservative. Aggressive normalisation would change
what the models are asked to summarise, and would make the comparison with
published CNN/DailyMail numbers meaningless.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator

from src.data.config import PreprocessConfig
from src.data.example import Example

#: Control characters carry no linguistic content and break tokenisers.
#: The tab, line feed and carriage return are handled by the whitespace rule.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

#: Runs of whitespace, including the newlines CNN/DailyMail puts between the
#: paragraphs of an article and between the bullets of its highlights.
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str, config: PreprocessConfig) -> str:
    """Normalise a single text field.

    Args:
        text: Raw text coming from the upstream corpus.
        config: Cleaning rules to apply.

    Returns:
        The cleaned text, stripped of leading and trailing whitespace.
    """
    cleaned = _CONTROL_CHARS.sub(" ", text)

    if config.normalise_unicode:
        cleaned = unicodedata.normalize("NFKC", cleaned)

    if config.collapse_whitespace:
        cleaned = _WHITESPACE.sub(" ", cleaned)

    return cleaned.strip()


def is_within_bounds(example: Example, config: PreprocessConfig) -> bool:
    """Return whether an example satisfies the configured length bounds.

    Args:
        example: The cleaned example.
        config: Bounds to enforce.

    Returns:
        ``True`` when both fields fall inside their bounds.
    """
    source_length = len(example.source)
    target_length = len(example.target)

    return (
        config.min_source_chars <= source_length <= config.max_source_chars
        and config.min_target_chars <= target_length <= config.max_target_chars
    )


def preprocess_example(example: Example, config: PreprocessConfig) -> Example:
    """Clean both fields of an example.

    Args:
        example: Raw example.
        config: Cleaning rules to apply.

    Returns:
        A new example with cleaned fields. The identifier is untouched.
    """
    return Example(
        example_id=example.example_id,
        source=clean_text(example.source, config),
        target=clean_text(example.target, config),
    )


def preprocess(examples: Iterable[Example], config: PreprocessConfig) -> Iterator[Example]:
    """Clean a stream of examples and drop those outside the bounds.

    Dropping happens after cleaning, on purpose: an example padded with
    whitespace must be judged on its cleaned length.

    Args:
        examples: Raw examples.
        config: Cleaning rules and length bounds.

    Yields:
        Cleaned examples that satisfy the bounds.
    """
    for example in examples:
        cleaned = preprocess_example(example, config)
        if is_within_bounds(cleaned, config):
            yield cleaned
