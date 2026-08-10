"""What a probe reads.

Section 25 asks for two probes rather than one, and the difference between them
is the whole point.

**Liveness answers for the process, readiness answers for the models.**
``/health`` says the application is running and can be reached: it never touches
the registry, so a supervisor cannot restart a container because a checkpoint is
missing. ``/ready`` says an inference request would be answered, which is false
on an instance whose registry is empty, and false is what a load balancer needs
to hear before it sends traffic.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Liveness of the process.

    Attributes:
        status: Always ``ok``. A process that could not answer would not.
        app: Application name, as exposed in the specification.
        environment: Deployment environment of the instance.
        api_version: Version of the contract, not of the package.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = Field(default="ok", description="Le processus repond.")
    app: str = Field(description="Nom de l'application.")
    environment: str = Field(description="Environnement de deploiement.")
    api_version: str = Field(description="Version du contrat expose.")


class ReadyResponse(BaseModel):
    """Readiness of the models.

    Attributes:
        status: Always ``ready``. An instance holding no model answers 503 with
            the error payload instead, so this field never reports a negative.
        models_loaded: Labels currently in memory, as ``name:version``.
        models_failed: Labels that failed to load, with the reason. An instance
            serving one model out of two is degraded, and hiding the second
            failure behind a 200 would leave nothing to alert on.
        device: Where the weights live.
        registry_backend: The backend the names were resolved through.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready"] = Field(default="ready", description="Au moins un modele est charge.")
    models_loaded: list[str] = Field(description="Versions en memoire.")
    models_failed: dict[str, str] = Field(
        default_factory=dict, description="Versions en echec, avec leur raison."
    )
    device: str = Field(description="Peripherique portant les poids.")
    registry_backend: str = Field(description="Backend du registre de modeles.")
