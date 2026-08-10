"""Liveness and readiness.

Both probes of section 25. ``/health`` never touches the registry, ``/ready``
answers for it. See :mod:`backend.app.schemas.health` for why the two are not
the same question.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.api.deps import SettingsDep, StoreDep
from backend.app.core.errors import ErrorCode, ModelNotReadyError
from backend.app.schemas.common import error_responses
from backend.app.schemas.health import HealthResponse, ReadyResponse

#: Version of the contract, not of the package. A breaking change to a schema
#: raises it; a release that only changes the implementation does not.
API_VERSION = "1.0.0"

router = APIRouter(tags=["monitoring"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Le processus repond",
    description=(
        "Sonde de vivacite. Ne lit ni le registre ni les modeles, "
        "pour qu'un checkpoint manquant ne provoque pas un redemarrage."
    ),
)
async def health(settings: SettingsDep) -> HealthResponse:
    """Report that the process is alive.

    Args:
        settings: The settings of the running instance.

    Returns:
        The liveness payload.
    """
    return HealthResponse(
        app=settings.app_name,
        environment=settings.environment.value,
        api_version=API_VERSION,
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Au moins un modele est charge",
    description=(
        "Sonde de disponibilite. Repond 503 tant qu'aucune version n'est en memoire, "
        "ce qui est l'etat d'une instance dont le registre est vide."
    ),
    responses=error_responses(ErrorCode.MODEL_NOT_READY),
)
async def ready(settings: SettingsDep, store: StoreDep) -> ReadyResponse:
    """Report whether an inference request would be answered.

    Args:
        settings: The settings of the running instance.
        store: The store of the running instance.

    Returns:
        The readiness payload.

    Raises:
        ModelNotReadyError: If no version is in memory. The failures recorded
            so far travel in the details, so an alert says which model is
            missing and why rather than only that the instance is not ready.
    """
    if not store.ready:
        raise ModelNotReadyError(
            "Aucun modele n'est charge sur cette instance.",
            details={"failures": store.failures, "registry_backend": settings.registry_backend},
        )

    return ReadyResponse(
        models_loaded=store.loaded,
        models_failed=store.failures,
        device=store.device.type,
        registry_backend=settings.registry_backend,
    )
