"""The error payload of section 26, as an OpenAPI schema.

:mod:`backend.app.core.errors` owns the codes and the statuses. This module is
the other half of that contract: it describes the same payload to the generated
specification, so a client can be written against ``docs/api/openapi.json``
without reading the Python.

**The two halves are checked against each other.** :func:`error_responses`
derives the documented status of a code from ``ERROR_STATUS`` rather than
repeating it, which is what stops an endpoint from advertising a 404 for a code
the handlers answer 503 for.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backend.app.core.errors import ERROR_STATUS, ErrorCode


class ErrorBody(BaseModel):
    """What went wrong.

    Attributes:
        code: The stable code of section 26. Renaming one is a breaking change.
        message: Human readable message. Safe to display.
        details: Structured context. Never holds the submitted text, and never
            a stack trace.
    """

    model_config = ConfigDict(extra="forbid")

    code: ErrorCode = Field(description="Code stable de la section 26.")
    message: str = Field(description="Message lisible, affichable tel quel.")
    details: dict[str, Any] | None = Field(default=None, description="Contexte structure.")


class ErrorResponse(BaseModel):
    """The envelope every error is returned in.

    Attributes:
        error: The error itself.
    """

    model_config = ConfigDict(extra="forbid")

    error: ErrorBody = Field(description="L'erreur, sous l'enveloppe unique de la section 26.")


def error_responses(*codes: ErrorCode) -> dict[int | str, dict[str, Any]]:
    """Describe the errors one endpoint may answer with.

    Args:
        *codes: The codes the endpoint raises.

    Returns:
        The ``responses`` mapping FastAPI documents the operation with, one
        entry per HTTP status, listing the codes that share it.
    """
    grouped: dict[int, list[str]] = {}
    for code in codes:
        grouped.setdefault(ERROR_STATUS[code], []).append(code.value)

    return {
        status: {"model": ErrorResponse, "description": ", ".join(sorted(values))}
        for status, values in sorted(grouped.items())
    }
