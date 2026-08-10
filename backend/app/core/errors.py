"""Single source of truth for application error codes.

Section 26 of the specification standardises the error payload returned by the
API. This module owns the enumeration, the exception hierarchy and the mapping
to HTTP status codes. Everything else, including the generated documentation
page and the OpenAPI response models, derives from this file.

Services raise :class:`SyntraError` subclasses and stay unaware of HTTP. The
FastAPI exception handlers translate them into the normalised payload.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    """Stable error codes exposed to API clients.

    Codes are part of the public contract: renaming one is a breaking change.
    """

    INVALID_INPUT = "INVALID_INPUT"
    INVALID_TASK = "INVALID_TASK"
    INVALID_LANGUAGE = "INVALID_LANGUAGE"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_NOT_READY = "MODEL_NOT_READY"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    TIMEOUT = "TIMEOUT"


#: Mapping from error code to HTTP status, consumed by the exception handlers
#: and by the generated documentation. Every member of :class:`ErrorCode` must
#: appear here; the completeness is asserted by the unit tests.
ERROR_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INVALID_INPUT: 422,
    ErrorCode.INVALID_TASK: 422,
    ErrorCode.INVALID_LANGUAGE: 422,
    ErrorCode.MODEL_NOT_FOUND: 404,
    ErrorCode.MODEL_NOT_READY: 503,
    ErrorCode.INFERENCE_FAILED: 500,
    ErrorCode.TIMEOUT: 504,
}

#: Default human readable messages. They are overridable per raise site.
ERROR_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.INVALID_INPUT: "La requete est invalide.",
    ErrorCode.INVALID_TASK: "La tache demandee n'est pas supportee.",
    ErrorCode.INVALID_LANGUAGE: "La langue demandee n'est pas supportee.",
    ErrorCode.MODEL_NOT_FOUND: "Le modele demande n'existe pas.",
    ErrorCode.MODEL_NOT_READY: "Le modele demande n'est pas encore charge.",
    ErrorCode.INFERENCE_FAILED: "L'inference a echoue.",
    ErrorCode.TIMEOUT: "L'inference a depasse le delai autorise.",
}


class SyntraError(Exception):
    """Base application error, deliberately unaware of HTTP.

    Attributes:
        code: The stable error code returned to the client.
        message: Human readable message, safe to expose.
        details: Optional structured context, never containing user text.
    """

    code: ErrorCode = ErrorCode.INFERENCE_FAILED

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Build the error.

        Args:
            message: Overrides the default message attached to the code.
            details: Structured context added to the response payload.
        """
        self.message = message or ERROR_MESSAGES[self.code]
        self.details = details or {}
        super().__init__(self.message)

    @property
    def status_code(self) -> int:
        """Return the HTTP status associated with the error code.

        Returns:
            The HTTP status code defined in :data:`ERROR_STATUS`.
        """
        return ERROR_STATUS[self.code]

    def to_payload(self) -> dict[str, Any]:
        """Render the normalised error payload of section 26.

        Returns:
            A dictionary shaped as ``{"error": {"code": ..., "message": ...}}``,
            with an optional ``details`` key.
        """
        error: dict[str, Any] = {"code": self.code.value, "message": self.message}
        if self.details:
            error["details"] = self.details
        return {"error": error}


class InvalidInputError(SyntraError):
    """Raised when the payload fails business validation."""

    code = ErrorCode.INVALID_INPUT


class InvalidTaskError(SyntraError):
    """Raised when the requested task is outside the project scope."""

    code = ErrorCode.INVALID_TASK


class InvalidLanguageError(SyntraError):
    """Raised when the requested language is not supported by the model."""

    code = ErrorCode.INVALID_LANGUAGE


class ModelNotFoundError(SyntraError):
    """Raised when the registry holds no model under the requested name."""

    code = ErrorCode.MODEL_NOT_FOUND


class ModelNotReadyError(SyntraError):
    """Raised when a known model exists but is not loaded yet."""

    code = ErrorCode.MODEL_NOT_READY


class InferenceFailedError(SyntraError):
    """Raised when the model raises during generation."""

    code = ErrorCode.INFERENCE_FAILED


class InferenceTimeoutError(SyntraError):
    """Raised when generation exceeds the configured deadline."""

    code = ErrorCode.TIMEOUT
