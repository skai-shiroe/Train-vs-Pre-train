"""Which code produced a run.

Section 19 asks for ``git_commit`` beside every traced run, and section 14 makes
that commit part of what a rerun needs. Reading it from ``git`` rather than from
a hand maintained constant is what keeps the two from drifting.

**What git cannot answer is reported as unknown, never omitted.** A repository
whose first commit has not been made yet, a source tree exported without its
history, a machine without ``git`` on the path: all three produce a run that is
worth tracing and a commit that does not exist. Dropping the field would leave a
tracked run that silently looks like every other one; ``unknown`` is a value a
reader can act on.

**A modified working tree is recorded, not refused.** Refusing to track a run
started from an uncommitted change would push the measurement out of the
tracking store, which is the opposite of what section 19 is for. The state is a
tag instead, so a score produced from code that was never committed can be told
apart from one that was.
"""

from __future__ import annotations

import shutil
import subprocess  # nosec B404 - only invokes git, with a fixed argument list
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

#: Reported when ``git`` cannot answer. Never an empty string: an empty commit
#: renders as a blank cell, which reads as "not applicable" rather than "not
#: known".
UNKNOWN = "unknown"

#: Repository the commit is read from, resolved from this file rather than from
#: the working directory: a run started from another directory belongs to the
#: same checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Seconds a git invocation is allowed to take. A hung git must not hang a run
#: that is otherwise finished.
GIT_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True, slots=True)
class Provenance:
    """The code a run was started from.

    Attributes:
        commit: Full commit identifier, or :data:`UNKNOWN`.
        branch: Branch name, or :data:`UNKNOWN`.
        dirty: Whether the working tree carried uncommitted changes. ``None``
            when git could not be asked, which is not the same answer as a
            clean tree.
    """

    commit: str
    branch: str
    dirty: bool | None

    def to_dict(self) -> dict[str, str]:
        """Render the provenance as tags.

        Returns:
            A flat mapping of strings, with an unknown working tree state
            rendered as :data:`UNKNOWN` rather than as ``False``.
        """
        return {
            "git_commit": self.commit,
            "git_branch": self.branch,
            "git_dirty": UNKNOWN if self.dirty is None else str(self.dirty).lower(),
        }


def run_git(arguments: Sequence[str], repo: Path) -> str | None:
    """Run one read only git command and return its output.

    Args:
        arguments: Arguments passed to ``git``, without the executable.
        repo: Directory the command runs in.

    Returns:
        The trimmed standard output, or ``None`` when git is absent, fails or
        times out. The three are not distinguished on purpose: every one of
        them means the answer is unknown.
    """
    executable = shutil.which("git")
    if executable is None:
        return None

    try:
        completed = subprocess.run(  # nosec B603 - fixed argument list, no shell
            [executable, *arguments],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def describe_provenance(repo: Path | None = None) -> Provenance:
    """Describe the code the current process was started from.

    Args:
        repo: Repository to inspect. Defaults to :data:`REPO_ROOT`.

    Returns:
        The provenance, with :data:`UNKNOWN` in place of anything git could not
        answer.
    """
    root = Path(repo) if repo is not None else REPO_ROOT
    commit = run_git(("rev-parse", "HEAD"), root)
    branch = run_git(("rev-parse", "--abbrev-ref", "HEAD"), root)
    status = run_git(("status", "--porcelain"), root)

    return Provenance(
        commit=commit or UNKNOWN,
        branch=branch or UNKNOWN,
        dirty=None if status is None else bool(status),
    )
