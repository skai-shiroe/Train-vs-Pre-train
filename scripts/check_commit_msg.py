"""Enforce the Conventional Commits format on commit messages.

Section 37.4 of the specification requires an exploitable commit message
format. The accepted types are listed in ``.pre-commit-config.yaml`` and mirrored
here so the hook stays the single mechanical source of truth.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TYPES = (
    "feat",
    "fix",
    "docs",
    "test",
    "refactor",
    "perf",
    "build",
    "ci",
    "chore",
    "revert",
)

# type(scope)!: subject
PATTERN = re.compile(
    rf"^(?P<type>{'|'.join(TYPES)})"
    r"(?P<scope>\([a-z0-9._/-]+\))?"
    r"(?P<breaking>!)?"
    r": (?P<subject>[^\s].{0,71})$"
)

MERGE_PREFIXES = ("Merge ", "Revert ", "fixup!", "squash!")

MAX_SUBJECT_LENGTH = 72


def first_meaningful_line(message: str) -> str:
    """Return the first line that is neither empty nor a comment.

    Args:
        message: Raw content of the commit message file.

    Returns:
        The subject line, stripped of trailing whitespace.
    """
    for line in message.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


def explain() -> str:
    """Return the help text shown when a commit message is rejected.

    Returns:
        A multi line explanation of the expected format.
    """
    return (
        "\nExpected format: type(scope): subject\n"
        f"Allowed types: {', '.join(TYPES)}\n"
        f"The subject must be at most {MAX_SUBJECT_LENGTH} characters and must not "
        "end with a period.\n"
        "Examples:\n"
        "  feat(api): add the compare endpoint\n"
        "  fix(scratch): correct the causal mask shape\n"
        "  docs(ml): document the ablation protocol\n"
        "See the Conventional Commits specification.\n"
    )


def main(argv: list[str]) -> int:
    """Validate the commit message file passed by the commit-msg hook.

    Args:
        argv: Command line arguments, the first one being the message file.

    Returns:
        Process exit code: 0 when the message is valid, 1 otherwise.
    """
    if not argv:
        print("check_commit_msg.py expects the commit message file path.")
        return 1

    message = Path(argv[0]).read_text(encoding="utf-8")
    subject = first_meaningful_line(message)

    if subject.startswith(MERGE_PREFIXES):
        return 0

    if not subject:
        print("Empty commit message.")
        print(explain())
        return 1

    if subject.endswith("."):
        print(f"Commit subject must not end with a period: {subject!r}")
        print(explain())
        return 1

    if not PATTERN.match(subject):
        print(f"Commit subject does not follow Conventional Commits: {subject!r}")
        print(explain())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
