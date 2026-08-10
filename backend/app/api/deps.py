"""What an endpoint asks the application for.

The store and the settings are built once, during the lifespan, and hung on the
application state. These dependencies are how a route reaches them, and the only
place that knows where they live.

**An instance without a store is not ready, it is not broken.** The state is
empty until the lifespan has run. Reaching it through a dependency that says so
turns a mistake in a test harness, and a request that arrives during startup,
into a 503 with a code rather than an ``AttributeError`` and a 500.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from backend.app.core.config import Settings, get_settings
from backend.app.core.errors import ModelNotReadyError
from backend.app.services.store import ModelStore


def app_settings(request: Request) -> Settings:
    """Return the settings the running application was built with.

    Args:
        request: The incoming request.

    Returns:
        The settings on the application state, falling back to the cached
        instance when the application was built without any.
    """
    settings = getattr(request.app.state, "settings", None)
    return settings if isinstance(settings, Settings) else get_settings()


def model_store(request: Request) -> ModelStore:
    """Return the store of the running instance.

    Args:
        request: The incoming request.

    Returns:
        The store built during the lifespan.

    Raises:
        ModelNotReadyError: If the lifespan has not run, which is the state of
            an application that is still starting.
    """
    store = getattr(request.app.state, "store", None)
    if not isinstance(store, ModelStore):
        raise ModelNotReadyError(
            "Le registre de modeles n'est pas encore initialise.",
            details={"reason": "startup incomplete"},
        )
    return store


#: The settings, injected into a route signature.
SettingsDep = Annotated[Settings, Depends(app_settings)]

#: The store, injected into a route signature.
StoreDep = Annotated[ModelStore, Depends(model_store)]
