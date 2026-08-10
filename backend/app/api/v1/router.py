"""The versioned router.

Section 22 puts every route behind ``/api/v1``. The prefix is not applied here:
it comes from ``SYNTRA_API_V1_PREFIX`` when the application mounts this router,
so a deployment behind a gateway can move it without touching any route.

The order of inclusion is the order the operations appear in the generated
specification, which is why the probes come first and inference last: a reader
of ``docs/api/openapi.json`` meets the cheap routes before the expensive ones.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.api.v1 import health, inference, models

router = APIRouter()
router.include_router(health.router)
router.include_router(models.router)
router.include_router(inference.router)
