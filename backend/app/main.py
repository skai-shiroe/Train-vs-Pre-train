"""The application, assembled.

Section 21. Everything the API is made of lives elsewhere; what is decided here
is how the pieces are put together and what happens when one of them fails.

**The models are loaded during the lifespan, not at import.** A module that
loaded weights when it was imported would make every tool that touches the
application, starting with the OpenAPI export, pull a few hundred megabytes into
memory. The lifespan is also what gives a failure somewhere to be reported:
``/ready`` answers for it.

**A registry that cannot be built stops the instance.** Section 20 already
refuses to fall back from an unwritten backend to the local one, and this is
where that refusal becomes visible: the exception escapes the lifespan and the
process exits with the name of the setting in the message. Serving requests with
weights the deployment did not ask for would be worse than not starting.

**A model that cannot be loaded does not stop it.** That failure is recorded and
reported. The difference between the two is who chose: the operator chose the
backend, nobody chose for a checkpoint to be corrupt.

**No exception reaches the client.** Every error leaves through the envelope of
section 26, whatever raised it. The generic handler answers ``INFERENCE_FAILED``
without a message of its own, because the message of an unexpected exception is
the one thing that could carry a path, a query or a fragment of the model.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import Any

import anyio.to_thread
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import Response

from backend.app.api.v1.health import API_VERSION
from backend.app.api.v1.router import router as v1_router
from backend.app.core.config import Settings, get_settings
from backend.app.core.errors import ERROR_MESSAGES, ErrorCode, SyntraError
from backend.app.monitoring.logging import configure_logging
from backend.app.monitoring.middleware import RequestLogMiddleware
from backend.app.registry.factory import build_registry
from backend.app.services.store import ModelStore
from src.utils.device import resolve_device

LOGGER = logging.getLogger(__name__)

#: What the specification says the API is. Kept here rather than in a settings
#: field: it describes the contract, and a contract is not configurable.
DESCRIPTION = (
    "Resume automatique de documents. Compare un Transformer encodeur-decodeur "
    "ecrit a la main a une baseline T5, servis depuis le registre de modeles.\n\n"
    "Les scores exposes par /models ont ete mesures sur le jeu de test fige du "
    "corpus de travail, jamais sur XSum complet."
)

#: What each group of operations answers. FastAPI orders the tag sections of the
#: document by this list, so a reader, and the folders a client generator builds
#: from it, meet the probes before the inference, as the router does.
TAGS_METADATA: list[dict[str, Any]] = [
    {
        "name": "monitoring",
        "description": "Sondes de vivacite et de disponibilite. Ne chargent aucun modele.",
    },
    {
        "name": "models",
        "description": (
            "Catalogue du registre : versions servies, alias de deploiement et "
            "scores publies. Lectures seules, sans chargement de poids."
        ),
    },
    {
        "name": "inference",
        "description": (
            "Execution des modeles sur un document. Les seules operations dont "
            "le cout depend de la taille de la charge utile."
        ),
    },
]


def operation_id(route: APIRoute) -> str:
    """Name an operation after the function that serves it.

    The default of FastAPI concatenates the name, the path and the method, which
    turns ``predict`` into ``predict_api_v1_predict_post``. That name reaches a
    generated client as a method name and a Postman request as a title, and it
    also embeds the prefix, so moving the API behind a gateway would rename
    every operation of the contract.

    Args:
        route: The route being documented.

    Returns:
        The name of the endpoint function, which is unique across the router.
    """
    return route.name


def code_for_status(status: int) -> ErrorCode:
    """Map a framework error onto the closed enumeration of section 26.

    An unknown route, a method that is not allowed and a body the parser could
    not read all reach the application as an HTTP status and nothing else.
    Reporting a 404 on a mistyped path as ``MODEL_NOT_FOUND`` would be worse
    than generic: a client switching on the code would tell its user that a
    model is missing when it is the URL that is.

    Args:
        status: The HTTP status the framework raised with.

    Returns:
        ``INVALID_INPUT`` for a client error, ``INFERENCE_FAILED`` otherwise.
    """
    return ErrorCode.INVALID_INPUT if status < 500 else ErrorCode.INFERENCE_FAILED


def error_payload(
    code: ErrorCode, message: str, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build the normalised payload of section 26.

    Args:
        code: The error code.
        message: The message shown to the client.
        details: Structured context, when there is any.

    Returns:
        The payload, shaped as ``{"error": {...}}``.
    """
    error: dict[str, Any] = {"code": code.value, "message": message}
    if details:
        error["details"] = details
    return {"error": error}


async def handle_syntra_error(_request: Request, exc: Exception) -> Response:
    """Render an application error.

    Args:
        _request: The request that raised. Unused: what the client is told
            never depends on the route.
        exc: The error.

    Returns:
        The normalised payload, under the status the code maps to.

    Raises:
        Exception: If the handler is reached with anything else, which would
            mean it was registered for a type it does not render.
    """
    if not isinstance(exc, SyntraError):
        raise exc
    return JSONResponse(status_code=exc.status_code, content=exc.to_payload())


def field_path(location: Iterable[Any]) -> str:
    """Name the part of the payload an error points at.

    Args:
        location: The ``loc`` tuple pydantic reported.

    Returns:
        The dotted path, with the body marker dropped so that a caller reads
        ``text`` rather than ``body.text``. A rule that validates the payload as
        a whole, such as the exclusion between sampling and beam search, points
        at no field of its own: it reports ``body``, because an empty string is
        something a client can neither display nor highlight.
    """
    return ".".join(str(part) for part in location if part != "body") or "body"


async def handle_validation_error(_request: Request, exc: Exception) -> Response:
    """Render a payload that failed validation.

    Only the field and the reason are returned. What pydantic also reports, the
    value it rejected, is the submitted document itself, and echoing it would
    put user content into a response and into every log that stores one.

    Args:
        _request: The request that raised.
        exc: The validation error.

    Returns:
        The normalised payload, with one entry per rejected field.

    Raises:
        Exception: If the handler is reached with anything else.
    """
    if not isinstance(exc, RequestValidationError):
        raise exc

    fields = [
        {"field": field_path(error.get("loc", ())), "message": str(error.get("msg", ""))}
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=error_payload(
            ErrorCode.INVALID_INPUT, ERROR_MESSAGES[ErrorCode.INVALID_INPUT], {"fields": fields}
        ),
    )


async def handle_http_exception(_request: Request, exc: Exception) -> Response:
    """Render a routing or method error in the application envelope.

    Args:
        _request: The request that raised.
        exc: The HTTP exception raised by the framework.

    Returns:
        The normalised payload, under the code :func:`code_for_status` gives
        the status, so the enumeration of section 26 stays closed.

    Raises:
        Exception: If the handler is reached with anything else.
    """
    if not isinstance(exc, StarletteHTTPException):
        raise exc

    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(code_for_status(exc.status_code), str(exc.detail)),
        headers=getattr(exc, "headers", None),
    )


async def handle_unexpected_error(_request: Request, exc: Exception) -> Response:
    """Render anything that was not foreseen.

    The exception is not inspected. Its message may hold a path, a fragment of
    a payload or the internals of a library, and none of that belongs in a
    response. The traceback is already in the log, written by the request
    middleware.

    Args:
        _request: The request that raised.
        exc: The exception.

    Returns:
        The generic failure payload, under status 500.
    """
    LOGGER.error("unhandled %s", type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content=error_payload(
            ErrorCode.INFERENCE_FAILED, ERROR_MESSAGES[ErrorCode.INFERENCE_FAILED]
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the registry and load the models, then release them.

    Args:
        app: The application being started.

    Yields:
        Nothing. The store is reachable through the application state.

    Raises:
        NotImplementedError: If the settings select a registry backend that is
            declared but not written.
        ValueError: If they name one that does not exist.
        RuntimeError: If they demand a device the machine does not have.
    """
    settings: Settings = app.state.settings
    device = resolve_device(settings.device)
    store = ModelStore(build_registry(settings), device=device)
    app.state.store = store

    if settings.eager_load_models:
        # Loading is torch work, and the event loop is already running: doing it
        # inline would block the probes of a container that has just started.
        loaded = await anyio.to_thread.run_sync(store.warm)
        LOGGER.info(
            "startup complete",
            extra={"loaded": loaded, "failed": sorted(store.failures), "device": device.type},
        )
    else:
        LOGGER.info("startup complete, models load on demand", extra={"device": device.type})

    yield

    app.state.store = None


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    Args:
        settings: Settings to run under. Defaults to the cached instance, which
            is what the server uses. A test, and the OpenAPI export, pass their
            own so the result does not depend on the machine it runs on.

    Returns:
        The configured application.
    """
    resolved = settings or get_settings()
    configure_logging(resolved)

    app = FastAPI(
        title=resolved.app_name,
        description=DESCRIPTION,
        version=API_VERSION,
        lifespan=lifespan,
        # Without it the document carries no origin, and every tool built from
        # it, Postman first, imports six paths it cannot send anything to. The
        # value is a setting because it is the one part of the contract that
        # genuinely differs between a laptop and a deployment.
        servers=[
            {
                "url": resolved.public_base_url,
                "description": (
                    f"Instance {resolved.environment.value}, " "reglee par SYNTRA_PUBLIC_BASE_URL."
                ),
            }
        ],
        openapi_tags=TAGS_METADATA,
        generate_unique_id_function=operation_id,
        docs_url="/docs" if resolved.docs_enabled else None,
        redoc_url="/redoc" if resolved.docs_enabled else None,
        openapi_url="/openapi.json" if resolved.docs_enabled else None,
    )
    app.state.settings = resolved
    app.state.store = None

    # Added first, so it sits inside CORS: a rejected origin is still timed and
    # logged, and an error response still carries the headers a browser needs
    # to read it.
    app.add_middleware(RequestLogMiddleware)
    if resolved.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved.cors_origins,
            allow_credentials=resolved.cors_allow_credentials,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    app.add_exception_handler(SyntraError, handle_syntra_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    app.add_exception_handler(Exception, handle_unexpected_error)

    app.include_router(v1_router, prefix=resolved.api_v1_prefix)
    return app


app = create_app()
