"""Reject em dashes and en dashes across the repository.

Section 1.1 of the specification forbids long dashes in every artifact produced
by the project: documentation, docstrings, comments, commit messages, CI job
descriptions and error messages. The constraint is blocking, so it is checked
mechanically here rather than by proofreading.

Allowed replacements: colon, comma, parenthesis, or two separate sentences.
The ordinary hyphen stays legal in compound words and command line options.
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

# Declared by code point on purpose: writing the characters literally would make
# this file fail its own check.
FORBIDDEN: dict[str, str] = {
    chr(0x2012): "figure dash",
    chr(0x2013): "en dash",
    chr(0x2014): "em dash",
    chr(0x2015): "horizontal bar",
}

SUGGESTION = "replace with a colon, a comma, a parenthesis, or split into two sentences"


def scan_file(path: Path) -> list[tuple[int, int, str]]:
    """Return every forbidden dash found in ``path``.

    Args:
        path: File to inspect. Binary and unreadable files are skipped.

    Returns:
        A list of ``(line_number, column, character)`` tuples, 1-indexed.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    hits: list[tuple[int, int, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for column, char in enumerate(line, start=1):
            if char in FORBIDDEN:
                hits.append((line_number, column, char))
    return hits


def describe(char: str) -> str:
    """Return a readable label for a forbidden character.

    Args:
        char: The offending character.

    Returns:
        A label combining the Unicode code point and its common name.
    """
    name = FORBIDDEN.get(char, unicodedata.name(char, "unknown"))
    return f"U+{ord(char):04X} ({name})"


def main(argv: list[str]) -> int:
    """Check every path given on the command line.

    Args:
        argv: Candidate file paths, normally supplied by pre-commit.

    Returns:
        Process exit code: 0 when the repository is clean, 1 otherwise.
    """
    failures = 0
    for raw_path in argv:
        path = Path(raw_path)
        if not path.is_file():
            continue
        for line_number, column, char in scan_file(path):
            failures += 1
            print(f"{path}:{line_number}:{column}: forbidden {describe(char)}")

    if failures:
        print(f"\n{failures} forbidden dash(es) found, {SUGGESTION}.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
