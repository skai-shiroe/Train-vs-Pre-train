"""The catalogue of what this instance can serve.

Section 22. Reading the registry is a file read, so both routes answer without
loading anything: asking what exists must not cost the memory of building it.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path

from backend.app.api.deps import StoreDep
from backend.app.core.errors import ErrorCode
from backend.app.registry.models import CHAMPION, NAME_PATTERN
from backend.app.schemas.common import error_responses
from backend.app.schemas.models import ModelInfo, ModelsResponse, ModelVersionsResponse

router = APIRouter(tags=["models"])


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="Modeles servis par cette instance",
    description=(
        "Version servie par defaut pour chaque nom enregistre, cibles des alias de "
        "deploiement, et ce qui est charge en memoire. Une liste vide est la reponse "
        "correcte d'un registre ou rien n'a encore ete publie."
    ),
)
async def list_models(store: StoreDep) -> ModelsResponse:
    """List the version each registered name currently serves.

    Args:
        store: The store of the running instance.

    Returns:
        The catalogue.
    """
    in_memory = set(store.loaded)
    return ModelsResponse(
        models=[
            ModelInfo.from_version(version, loaded=version.label in in_memory)
            for version in store.catalog()
        ],
        aliases=store.alias_targets(),
        default=CHAMPION,
    )


@router.get(
    "/models/{name}",
    response_model=ModelVersionsResponse,
    summary="Historique des versions d'un modele",
    description=(
        "Toutes les versions publiees sous un nom, de la plus ancienne a la plus "
        "recente. Un alias de deploiement n'est pas un nom et n'est pas accepte ici."
    ),
    # INVALID_INPUT is listed because ``name`` is validated against a pattern,
    # so a malformed one is refused before the handler runs. FastAPI documents
    # that refusal with its own ``HTTPValidationError`` unless the operation
    # declares a 422 of its own, and the application answers it through the
    # envelope of section 26 like every other error. Leaving the default in
    # place made this the one route whose documented error shape was not the
    # one a client receives.
    responses=error_responses(ErrorCode.INVALID_INPUT, ErrorCode.MODEL_NOT_FOUND),
)
async def list_versions(
    store: StoreDep,
    name: Annotated[str, Path(pattern=NAME_PATTERN, description="Nom enregistre du modele.")],
) -> ModelVersionsResponse:
    """List every published version of one model.

    Args:
        store: The store of the running instance.
        name: The registered name.

    Returns:
        The versions, oldest first.

    Raises:
        ModelNotFoundError: If nothing is registered under that name.
    """
    in_memory = set(store.loaded)
    return ModelVersionsResponse(
        name=name,
        versions=[
            ModelInfo.from_version(version, loaded=version.label in in_memory)
            for version in store.registry.versions(name)
        ],
    )
