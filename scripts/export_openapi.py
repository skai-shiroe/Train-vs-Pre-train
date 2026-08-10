"""Write the OpenAPI document the frontend is built against, and its environment.

Section 22 asks for a versioned contract, and section 38.4 asks for every
artifact derived from the code to be regenerated rather than maintained by hand.
``scripts/check_sync.py`` runs this script into a temporary file and
compares the result with the committed one, so a route added without a
``make docs-sync`` fails the pipeline instead of reaching a client as a
surprise.

**The document is exported from the declared defaults, not from the
environment.** The prefix of the routes and the title of the specification are
settings. Exporting under whatever ``.env`` the developer happens to have would
produce a document that differs from one machine to the next, and the freshness
check would fail for a reason that has nothing to do with the code.

**The keys are sorted.** FastAPI builds the document from dictionaries whose
order follows the declaration order of the models. Sorting makes the file a
stable diff: a review sees the route that changed, not the fifty lines that
moved.

**The Postman environment comes out of the same settings.** The origin the
specification publishes in its ``servers`` block and the ``baseUrl`` a Postman
collection sends to are the same value, ``SYNTRA_PUBLIC_BASE_URL``. Writing the
two files from one run of one script is what keeps them from disagreeing, and
what makes the freshness check able to say so when they do.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.config import Settings  # noqa: E402
from backend.app.main import create_app  # noqa: E402

#: Where the committed document lives.
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "api" / "openapi.json"

#: Where the committed Postman environment lives. It is generated beside the
#: specification rather than written by hand for the reason of section 38.4: the
#: origin it carries is a setting, and a copy maintained by hand would keep
#: pointing at a port the application stopped listening on.
DEFAULT_ENVIRONMENT_OUTPUT = REPO_ROOT / "docs" / "api" / "syntra.postman_environment.json"

#: Identifier of the generated environment. Postman keys an import on it, so a
#: value drawn at random would make every regeneration a different environment
#: in the client and every export a diff. Derived from a fixed name instead, so
#: it is the same on every machine and across every run.
ENVIRONMENT_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "https://syntra/postman/environment"))


def default_settings() -> Settings:
    """Return the settings the document is exported under.

    Every field is passed explicitly. Values given to the constructor win over
    the environment in pydantic-settings, which is what makes the export
    independent of the machine running it.

    Returns:
        The settings, built from the declared defaults.
    """
    values: dict[str, Any] = {
        name: field.get_default(call_default_factory=True)
        for name, field in Settings.model_fields.items()
    }
    return Settings(**values, _env_file=None)


def build_document() -> dict[str, Any]:
    """Generate the specification of the application.

    Returns:
        The OpenAPI document, as a mapping.
    """
    application = create_app(default_settings())
    document: dict[str, Any] = application.openapi()
    return document


def build_environment() -> dict[str, Any]:
    """Generate the Postman environment that goes with the specification.

    Importing ``openapi.json`` into Postman builds a collection whose requests
    are addressed to a ``baseUrl`` variable, taken from the ``servers`` block.
    This file is what fills that variable, and what a second one would fill
    differently to aim the same collection at a deployment.

    ``apiPrefix`` is not used by the imported collection, whose paths already
    carry it. It is there for a request written by hand, so that it is written
    against the same setting as the routes rather than against a literal.

    Returns:
        The environment, in the export format Postman reads.
    """
    settings = default_settings()
    return {
        "id": ENVIRONMENT_ID,
        "name": f"{settings.app_name} {settings.environment.value}",
        "values": [
            {
                "key": "baseUrl",
                "value": settings.public_base_url,
                "type": "default",
                "enabled": True,
            },
            {
                "key": "apiPrefix",
                "value": settings.api_v1_prefix,
                "type": "default",
                "enabled": True,
            },
        ],
        "_postman_variable_scope": "environment",
    }


def write_document(document: dict[str, Any], output: Path) -> Path:
    """Write the specification to disk.

    Args:
        document: The generated specification.
        output: Destination file. Its parent is created when missing.

    Returns:
        The path written.
    """
    # newline="\n" : sans lui, l'ecriture en mode texte sous Windows produit des
    # fins de ligne CRLF que le hook mixed-line-ending reconvertit aussitot. Le
    # fichier versionne changerait a chaque aller-retour entre make docs-sync et
    # pre-commit, sans qu'aucune route ait bouge.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output


def build_argument_parser() -> argparse.ArgumentParser:
    """Describe the command line.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python scripts/export_openapi.py",
        description="Export the OpenAPI document of the Syntra API.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination file. Defaults to {DEFAULT_OUTPUT.relative_to(REPO_ROOT)}.",
    )
    parser.add_argument(
        "--environment-output",
        type=Path,
        default=DEFAULT_ENVIRONMENT_OUTPUT,
        help=(
            "Destination of the Postman environment. Defaults to "
            f"{DEFAULT_ENVIRONMENT_OUTPUT.relative_to(REPO_ROOT)}."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Export the document from the command line.

    Args:
        argv: Arguments to parse. Defaults to the process arguments.

    Returns:
        Process exit code, always zero: the export either writes the file or
        raises.
    """
    args = build_argument_parser().parse_args(argv)
    written = write_document(build_document(), args.output)
    print(f"OpenAPI written to {written}")
    environment = write_document(build_environment(), args.environment_output)
    print(f"Postman environment written to {environment}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
