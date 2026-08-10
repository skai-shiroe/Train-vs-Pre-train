"""What the catalogue endpoint returns.

Section 22 asks the API to expose the models it can serve. A name and a version
would be enough to route a request, and would be the wrong answer: the question
a reader of this endpoint actually has is which weights are behind a name and
what they scored.

**A model is reported with its measurement or not at all.** ``experiment``,
``git_commit`` and the ROUGE scores travel with a version since section 20. They
are copied through here so that a client comparing two models is comparing
numbers that were measured on the frozen test set, not numbers it invented from
two summaries.

**Being registered and being loaded are two different facts.** ``loaded`` says
what is in memory right now. A registered version that no instance has loaded
answers a request after a pause, and one that failed to load answers 503; both
are visible here rather than discovered on the first call.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from backend.app.registry.models import ModelVersion


class ModelInfo(BaseModel):
    """One registered version, as the API reports it.

    Attributes:
        name: Registered model name.
        version: Version identifier.
        source: Where the weights come from, ``checkpoint`` or ``huggingface``.
        description: One line describing the version.
        experiment: Run that produced the weights.
        dataset_version: Checksum of the corpus that run trained on.
        git_commit: Commit the run was started from.
        created_at: Publication timestamp, in UTC.
        metrics: Scores copied from the run record, empty for a version
            published without one.
        loaded: Whether this version is in memory on this instance.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Nom enregistre du modele.")
    version: str = Field(description="Identifiant de version.")
    source: str = Field(description="Origine des poids.")
    description: str = Field(default="", description="Une ligne decrivant la version.")
    experiment: str | None = Field(default=None, description="Run ayant produit les poids.")
    dataset_version: str | None = Field(default=None, description="Empreinte du corpus.")
    git_commit: str | None = Field(default=None, description="Commit du run.")
    created_at: str = Field(description="Horodatage de publication, en UTC.")
    metrics: dict[str, float] = Field(default_factory=dict, description="Scores du run.")
    loaded: bool = Field(description="Version presente en memoire sur cette instance.")

    @classmethod
    def from_version(cls, version: ModelVersion, *, loaded: bool) -> ModelInfo:
        """Render a registry document as the catalogue entry.

        Args:
            version: The published version.
            loaded: Whether it is in memory on this instance.

        Returns:
            The entry.
        """
        return cls(
            name=version.name,
            version=version.version,
            source=version.source.value,
            description=version.description,
            experiment=version.experiment,
            dataset_version=version.dataset_version,
            git_commit=version.git_commit,
            created_at=version.created_at,
            metrics=dict(version.metrics),
            loaded=loaded,
        )


class ModelsResponse(BaseModel):
    """The catalogue of one instance.

    Attributes:
        models: The version each registered name currently serves.
        aliases: Deployment aliases that point somewhere, as ``name:version``.
            An alias nobody promoted is absent rather than null.
        default: The name a request resolves to when it names none.
    """

    model_config = ConfigDict(extra="forbid")

    models: list[ModelInfo] = Field(description="Version servie par defaut pour chaque nom.")
    aliases: dict[str, str] = Field(default_factory=dict, description="Cibles des alias.")
    default: str = Field(description="Nom utilise quand la requete n'en donne aucun.")


class ModelVersionsResponse(BaseModel):
    """Every published version of one model.

    Attributes:
        name: The registered name asked for.
        versions: Its versions, oldest first. Nothing is ever removed from a
            registry, so this is the full history of what that name served.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Nom enregistre du modele.")
    versions: list[ModelInfo] = Field(description="Versions publiees, de la plus ancienne.")
