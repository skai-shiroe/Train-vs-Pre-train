"""Application settings.

Section 21.1 of the specification forbids reading ``os.environ`` outside this
module. Every runtime knob lives here so that ``.env.example`` and the generated
configuration reference can be checked against a single source of truth.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.app.registry.models import DEFAULT_REGISTRY_ROOT


class Environment(StrEnum):
    """Deployment environment of the running instance."""

    LOCAL = "local"
    CI = "ci"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    """Accepted logging verbosity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Settings(BaseSettings):
    """Runtime configuration, loaded from the environment.

    Attributes are grouped by concern. Every attribute must have a counterpart
    in ``.env.example``; ``scripts/check_docs_sync.py`` enforces the parity.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SYNTRA_",
        extra="forbid",
        frozen=True,
    )

    # -- Application ------------------------------------------------------
    app_name: str = Field(default="Syntra", description="Nom expose dans OpenAPI.")
    environment: Environment = Field(
        default=Environment.LOCAL, description="Environnement de deploiement."
    )
    debug: bool = Field(default=False, description="Active les traces detaillees.")
    api_v1_prefix: str = Field(default="/api/v1", description="Prefixe des routes versionnees.")
    docs_enabled: bool = Field(
        default=True, description="Expose /docs et /redoc. A desactiver en production."
    )
    public_base_url: str = Field(
        default="http://127.0.0.1:8000",
        description="Origine publique de l'instance, publiee dans le bloc servers d'OpenAPI.",
    )

    # -- Journalisation ---------------------------------------------------
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Verbosite des logs.")
    log_json: bool = Field(default=True, description="Journalisation structuree en JSON.")

    # -- CORS -------------------------------------------------------------
    cors_origins: list[str] = Field(
        default_factory=list,
        description="Origines autorisees pour le futur frontend. Jamais l'etoile avec credentials.",
    )
    cors_allow_credentials: bool = Field(
        default=False, description="Autorise l'envoi de cookies par le frontend."
    )

    # -- Inference --------------------------------------------------------
    device: str = Field(default="auto", description="Peripherique torch : auto, cpu ou cuda.")
    inference_timeout_s: float = Field(
        default=30.0, gt=0, description="Delai maximal d'une generation, en secondes."
    )
    max_input_chars: int = Field(
        default=5000, gt=0, description="Taille maximale du texte source accepte."
    )
    max_summary_tokens: int = Field(
        default=64, gt=0, description="Longueur maximale du resume genere, en tokens."
    )
    eager_load_models: bool = Field(
        default=True, description="Charge les modeles au demarrage plutot qu'a la demande."
    )

    # -- Model Registry ---------------------------------------------------
    registry_backend: str = Field(
        default="local", description="Backend du registre : local ou mlflow."
    )
    registry_root: Path = Field(
        default=DEFAULT_REGISTRY_ROOT,
        description="Racine du registre local de modeles.",
    )
    pretrained_model_id: str = Field(
        default="t5-small", description="Identifiant Hugging Face du modele pre-entraine."
    )

    # -- MLflow -----------------------------------------------------------
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", description="URI du serveur de tracking MLflow."
    )
    mlflow_experiment: str = Field(
        default="syntra-summarization", description="Nom de l'experience MLflow."
    )

    # -- Stockage objet ---------------------------------------------------
    s3_endpoint_url: str | None = Field(
        default=None, description="Endpoint MinIO, vide si le stockage objet est inutilise."
    )
    s3_access_key: SecretStr | None = Field(default=None, description="Cle d'acces MinIO.")
    s3_secret_key: SecretStr | None = Field(default=None, description="Cle secrete MinIO.")

    @field_validator("public_base_url")
    @classmethod
    def _normalise_base_url(cls, value: str) -> str:
        """Refuse an origin no client could resolve, and drop a trailing slash.

        The value is published in the ``servers`` block of the specification and
        prepended to a path that already starts with one. A trailing slash would
        therefore reach a generated client as ``//api/v1``, and an origin without
        a scheme is not a URL an importer can build a request from.

        Args:
            value: The configured origin.

        Returns:
            The origin without its trailing slash.

        Raises:
            ValueError: If the scheme is missing, or if a path is attached. The
                prefix of the routes is ``SYNTRA_API_V1_PREFIX`` and carrying a
                second one here would document it twice.
        """
        stripped = value.strip().rstrip("/")
        if not stripped.startswith(("http://", "https://")):
            raise ValueError("L'origine publique doit commencer par http:// ou https://.")

        scheme, _, remainder = stripped.partition("://")
        if not remainder:
            raise ValueError("L'origine publique doit porter un hote.")
        if "/" in remainder:
            raise ValueError(
                "L'origine publique ne porte que le schema, l'hote et le port. "
                "Le chemin des routes vient de SYNTRA_API_V1_PREFIX."
            )
        return f"{scheme}://{remainder}"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma separated string for ``SYNTRA_CORS_ORIGINS``.

        Args:
            value: Raw value coming from the environment or from a literal list.

        Returns:
            A list of origins when the input was a comma separated string,
            otherwise the untouched value.
        """
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("cors_origins")
    @classmethod
    def _reject_wildcard_with_credentials(cls, value: list[str]) -> list[str]:
        """Forbid the wildcard origin, which is unusable with credentials.

        Args:
            value: Parsed list of origins.

        Returns:
            The validated list.

        Raises:
            ValueError: If the wildcard origin is present.
        """
        if "*" in value:
            raise ValueError(
                "L'origine etoile est interdite. Lister explicitement les origines autorisees."
            )
        return value

    @property
    def is_production(self) -> bool:
        """Return whether the instance runs in production.

        Returns:
            ``True`` when the environment is production.
        """
        return self.environment is Environment.PRODUCTION


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings instance.

    The cache keeps a single instance per process, which is what the FastAPI
    dependency injection system expects. Tests clear it with
    ``get_settings.cache_clear()``.

    Returns:
        The application settings.
    """
    return Settings()
