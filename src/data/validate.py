"""Corpus validation.

Validation runs twice: once on the raw download, to catch a schema change
upstream, and once on the frozen working corpus, to guarantee the properties
the scientific comparison depends on.

The most important check is leakage: no test example may appear in a training
split. A leaked test set turns every reported score into a fabricated result.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from src.data.example import Example


class ValidationError(Exception):
    """Raised when a corpus violates a property the pipeline depends on."""


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Outcome of validating one split.

    Attributes:
        split: Name of the validated split.
        total: Number of examples inspected.
        empty_source: Count of examples whose document is empty.
        empty_target: Count of examples whose summary is empty.
        duplicate_ids: Identifiers appearing more than once.
        duplicate_sources: Number of documents appearing more than once.
        target_longer_than_source: Count of summaries longer than their document.
    """

    split: str
    total: int
    empty_source: int = 0
    empty_target: int = 0
    duplicate_ids: tuple[str, ...] = field(default_factory=tuple)
    duplicate_sources: int = 0
    target_longer_than_source: int = 0

    @property
    def is_clean(self) -> bool:
        """Return whether the split is free of blocking problems.

        A summary longer than its document is suspicious but not blocking: some
        upstream examples are legitimately terse.

        Returns:
            ``True`` when no blocking problem was found.
        """
        return self.empty_source == 0 and self.empty_target == 0 and len(self.duplicate_ids) == 0

    def describe(self) -> str:
        """Return a one line summary suitable for logs.

        Returns:
            A readable description of the report.
        """
        return (
            f"{self.split}: {self.total} examples, "
            f"{self.empty_source} empty documents, "
            f"{self.empty_target} empty summaries, "
            f"{len(self.duplicate_ids)} duplicate ids, "
            f"{self.duplicate_sources} duplicate documents, "
            f"{self.target_longer_than_source} summaries longer than their document"
        )


def validate_split(split: str, examples: Sequence[Example]) -> ValidationReport:
    """Inspect one split and report every anomaly found.

    Args:
        split: Name of the split, used in the report and in error messages.
        examples: Examples to inspect.

    Returns:
        The validation report. Inspecting never raises: the caller decides
        whether a report is acceptable.
    """
    id_counts = Counter(example.example_id for example in examples)
    source_counts = Counter(example.source for example in examples)

    return ValidationReport(
        split=split,
        total=len(examples),
        empty_source=sum(1 for example in examples if not example.source.strip()),
        empty_target=sum(1 for example in examples if not example.target.strip()),
        duplicate_ids=tuple(
            sorted(example_id for example_id, count in id_counts.items() if count > 1)
        ),
        duplicate_sources=sum(count - 1 for count in source_counts.values() if count > 1),
        target_longer_than_source=sum(
            1 for example in examples if len(example.target) > len(example.source)
        ),
    )


def require_clean(report: ValidationReport) -> None:
    """Raise when a report contains a blocking problem.

    Args:
        report: The report to check.

    Raises:
        ValidationError: If the split contains empty fields or duplicate ids.
    """
    if report.is_clean:
        return

    problems: list[str] = []
    if report.empty_source:
        problems.append(f"{report.empty_source} empty documents")
    if report.empty_target:
        problems.append(f"{report.empty_target} empty summaries")
    if report.duplicate_ids:
        shown = ", ".join(report.duplicate_ids[:5])
        problems.append(f"{len(report.duplicate_ids)} duplicate ids ({shown})")

    raise ValidationError(f"Split {report.split!r} is not usable: " + "; ".join(problems))


def require_no_leakage(
    train: Sequence[Example],
    validation: Sequence[Example],
    test: Sequence[Example],
) -> None:
    """Raise when the splits overlap.

    Overlap is checked on identifiers and on document text. Two examples can
    carry different identifiers and still hold the same document, which would
    leak the test set just as effectively.

    Args:
        train: Training examples.
        validation: Validation examples.
        test: Test examples.

    Raises:
        ValidationError: If any pair of splits shares an identifier or a document.
    """
    splits = {"train": train, "validation": validation, "test": test}
    pairs = (("train", "test"), ("train", "validation"), ("validation", "test"))

    for left, right in pairs:
        shared_ids = {example.example_id for example in splits[left]} & {
            example.example_id for example in splits[right]
        }
        if shared_ids:
            shown = ", ".join(sorted(shared_ids)[:5])
            raise ValidationError(
                f"{len(shared_ids)} identifier(s) shared between {left} and {right}: {shown}"
            )

        shared_sources = {example.source for example in splits[left]} & {
            example.source for example in splits[right]
        }
        if shared_sources:
            raise ValidationError(
                f"{len(shared_sources)} document(s) shared between {left} and {right}"
            )
