"""Detect drift between the code and the artifacts derived from it.

Section 38.4 of the specification states that a wrong documentation is worse
than a missing one, and that the synchronisation must be checked mechanically.

Checks performed:

1. ``.env.example`` exposes exactly the fields declared by ``Settings``.
2. ``docs/api/openapi.json`` and ``docs/api/syntra.postman_environment.json``
   match what the FastAPI application and its settings regenerate.
3. ``docs/_generated`` and the marked region of ``README.md`` match what the
   run records produce, which is what keeps a page from stating the scores of
   the previous campaign.
4. The generated tables of the corpus page match what ``make data`` wrote
   beside the corpus, which is what keeps that page from describing the
   previous build.

A check whose target does not exist yet is reported as ``PENDING`` and does not
fail the run. It is never reported as passing: section 44 forbids presenting a
missing result as a successful one.
"""

from __future__ import annotations

import json
import re
import subprocess  # nosec B404 - only invokes the project's own export script
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = REPO_ROOT / ".env.example"
OPENAPI_REFERENCE = REPO_ROOT / "docs" / "api" / "openapi.json"
POSTMAN_ENVIRONMENT = REPO_ROOT / "docs" / "api" / "syntra.postman_environment.json"
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "export_openapi.py"
FRAGMENTS_MODULE = REPO_ROOT / "src" / "experiments" / "fragments.py"
CORPUS_FRAGMENTS_MODULE = REPO_ROOT / "src" / "data" / "fragments.py"
FRAGMENTS_DIR = REPO_ROOT / "docs" / "_generated"
EXPERIMENTS_DIR = REPO_ROOT / "configs" / "experiments"
RESULTS_DIR = REPO_ROOT / "reports" / "results"
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "xsum"
README = REPO_ROOT / "README.md"

ENV_PREFIX = "SYNTRA_"
ENV_LINE = re.compile(r"^(?P<key>[A-Z][A-Z0-9_]*)=")

OK = "OK"
FAILED = "FAILED"
PENDING = "PENDING"


def _env_example_keys() -> set[str]:
    """Return the ``SYNTRA_`` keys declared in ``.env.example``.

    Returns:
        The set of setting names, prefix stripped and lowercased.
    """
    keys: set[str] = set()
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        match = ENV_LINE.match(line.strip())
        if match and match.group("key").startswith(ENV_PREFIX):
            keys.add(match.group("key")[len(ENV_PREFIX) :].lower())
    return keys


def check_settings_parity() -> tuple[str, str]:
    """Compare ``.env.example`` with the ``Settings`` model.

    Returns:
        A ``(status, message)`` pair.
    """
    if not ENV_EXAMPLE.is_file():
        return FAILED, ".env.example is missing."

    sys.path.insert(0, str(REPO_ROOT))
    from backend.app.core.config import Settings

    declared = set(Settings.model_fields)
    documented = _env_example_keys()

    missing = sorted(declared - documented)
    extra = sorted(documented - declared)

    if not missing and not extra:
        return OK, f"{len(declared)} settings documented in .env.example."

    problems = []
    if missing:
        problems.append("missing from .env.example: " + ", ".join(missing))
    if extra:
        problems.append("unknown in Settings: " + ", ".join(extra))
    return FAILED, "; ".join(problems)


def check_openapi_freshness() -> tuple[str, str]:
    """Compare the versioned API artifacts with freshly generated ones.

    Both files the export writes are checked in one run of it. The export takes
    a few seconds because importing the application imports torch, and the two
    artifacts are derived from the same settings anyway: regenerating them
    separately would double the cost to compare the same thing twice.

    **Both destinations are redirected into the temporary directory.** The
    export writes the environment beside the specification by default, so a
    check that only redirected ``--output`` would rewrite a versioned file while
    claiming to verify it, and would report a match it had just created.

    Returns:
        A ``(status, message)`` pair.
    """
    if not EXPORT_SCRIPT.is_file():
        return PENDING, "scripts/export_openapi.py does not exist yet."

    missing = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in (OPENAPI_REFERENCE, POSTMAN_ENVIRONMENT)
        if not path.is_file()
    ]
    if missing:
        return FAILED, f"{', '.join(missing)} missing, run: make docs-sync"

    with tempfile.TemporaryDirectory() as tmp:
        generated_document = Path(tmp) / "openapi.json"
        generated_environment = Path(tmp) / "environment.json"
        result = subprocess.run(  # nosec B603 - fixed argument list, no shell
            [
                sys.executable,
                str(EXPORT_SCRIPT),
                "--output",
                str(generated_document),
                "--environment-output",
                str(generated_environment),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return FAILED, f"export_openapi.py failed: {result.stderr.strip()}"

        stale_files = [
            reference.relative_to(REPO_ROOT).as_posix()
            for reference, fresh in (
                (OPENAPI_REFERENCE, generated_document),
                (POSTMAN_ENVIRONMENT, generated_environment),
            )
            if json.loads(reference.read_text(encoding="utf-8"))
            != json.loads(fresh.read_text(encoding="utf-8"))
        ]

    if not stale_files:
        return OK, "openapi.json and the Postman environment match the application."
    return FAILED, f"{', '.join(stale_files)} stale, run: make docs-sync"


def check_report_freshness() -> tuple[str, str]:
    """Compare the fragments the report includes with the run records.

    The records are deliberately not versioned: section 17 keeps
    ``reports/results`` out of git and publishes it as a pipeline artefact. So
    this check only has something to compare on a machine holding a campaign,
    which is also the only machine where a stale fragment can be produced.
    Anywhere else it reports ``PENDING`` rather than passing over an empty
    directory: a comparison with nothing to compare must not read as a match,
    and regenerating from no records at all would rewrite every table of the
    report as an unrun experiment.

    Returns:
        A ``(status, message)`` pair.
    """
    if not FRAGMENTS_MODULE.is_file():
        return PENDING, "src/experiments/fragments.py does not exist yet."

    sys.path.insert(0, str(REPO_ROOT))
    from src.experiments.fragments import build, stale
    from src.experiments.registry import collect

    if not any(row.record is not None for row in collect(EXPERIMENTS_DIR, RESULTS_DIR)):
        return PENDING, f"{RESULTS_DIR.name}/ holds no run record on this machine."

    try:
        rendered = build(
            experiments_dir=EXPERIMENTS_DIR,
            results_dir=RESULTS_DIR,
            output_dir=FRAGMENTS_DIR,
            readme=README,
        )
    except ValueError as error:
        return FAILED, str(error)

    outdated = stale(rendered)
    if not outdated:
        return OK, f"{len(rendered)} generated tables match the run records."
    names = ", ".join(path.name for path in outdated)
    return FAILED, f"a generated table is stale ({names}), run: make report-sync"


def check_corpus_freshness() -> tuple[str, str]:
    """Compare the fragments the corpus page includes with the corpus record.

    The corpus is no more versioned than the run records are: ``.gitignore``
    keeps ``data/`` out of the repository and ``make data`` rebuilds it from the
    configuration. So this check, like the one above it, only has something to
    compare on a machine holding a built corpus, and reports ``PENDING``
    elsewhere rather than passing over an absent record.

    Returns:
        A ``(status, message)`` pair.
    """
    if not CORPUS_FRAGMENTS_MODULE.is_file():
        return PENDING, "src/data/fragments.py does not exist yet."

    sys.path.insert(0, str(REPO_ROOT))
    from src.data.fragments import MANIFEST_NAME, build, stale

    if not (PROCESSED_DIR / MANIFEST_NAME).is_file():
        return PENDING, f"{PROCESSED_DIR.name}/ holds no built corpus on this machine."

    try:
        rendered = build(processed_dir=PROCESSED_DIR, output_dir=FRAGMENTS_DIR)
    except ValueError as error:
        return FAILED, str(error)

    outdated = stale(rendered)
    if not outdated:
        return OK, f"{len(rendered)} generated tables match the corpus record."
    names = ", ".join(path.name for path in outdated)
    return FAILED, f"a generated table is stale ({names}), run: make corpus-sync"


def main() -> int:
    """Run every synchronisation check and report a summary.

    Returns:
        Process exit code: 0 when no check failed, 1 otherwise.
    """
    checks = {
        "settings vs .env.example": check_settings_parity,
        "OpenAPI and Postman vs application": check_openapi_freshness,
        "report vs run records": check_report_freshness,
        "corpus page vs corpus record": check_corpus_freshness,
    }

    failed = 0
    for name, check in checks.items():
        status, message = check()
        print(f"[{status:>7}] {name}: {message}")
        if status == FAILED:
            failed += 1

    if failed:
        print(f"\n{failed} synchronisation check(s) failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
