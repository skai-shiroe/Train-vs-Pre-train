"""Unit tests for the application settings."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.core.config import Environment, LogLevel, Settings, get_settings


def build(**overrides: object) -> Settings:
    """Build settings isolated from any local .env file.

    Args:
        **overrides: Field values applied on top of the defaults.

    Returns:
        A settings instance built without reading the environment file.
    """
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


@pytest.mark.unit
def test_defaults_are_safe_for_local_development() -> None:
    settings = build()

    assert settings.environment is Environment.LOCAL
    assert settings.debug is False
    assert settings.api_v1_prefix == "/api/v1"
    assert settings.log_level is LogLevel.INFO
    assert settings.cors_origins == []
    assert settings.is_production is False


@pytest.mark.unit
def test_pretrained_model_matches_the_locked_decision() -> None:
    assert build().pretrained_model_id == "t5-small"


@pytest.mark.unit
def test_cors_origins_accept_a_comma_separated_string() -> None:
    settings = build(cors_origins="http://a.test, http://b.test ,")

    assert settings.cors_origins == ["http://a.test", "http://b.test"]


@pytest.mark.unit
def test_wildcard_origin_is_rejected() -> None:
    with pytest.raises(ValidationError, match="etoile est interdite"):
        build(cors_origins="*")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("configured", "published"),
    [
        ("http://127.0.0.1:8000", "http://127.0.0.1:8000"),
        # The routes of the document already start with a slash, so a trailing
        # one would reach a generated client as //api/v1.
        ("http://127.0.0.1:8000/", "http://127.0.0.1:8000"),
        ("  https://syntra.example  ", "https://syntra.example"),
    ],
)
def test_public_base_url_is_normalised(configured: str, published: str) -> None:
    assert build(public_base_url=configured).public_base_url == published


@pytest.mark.unit
@pytest.mark.parametrize(
    "configured",
    [
        "127.0.0.1:8000",  # no scheme: not a URL an importer can build on
        "ftp://127.0.0.1:8000",
        "http://",  # a scheme and no host
        "http://127.0.0.1:8000/api/v1",  # the prefix is SYNTRA_API_V1_PREFIX
    ],
)
def test_a_public_base_url_no_client_could_use_is_rejected(configured: str) -> None:
    # It is published in the servers block and in the Postman environment, so a
    # value that only looks like a URL fails once, silently, in whatever tool
    # imported the document rather than here.
    with pytest.raises(ValidationError):
        build(public_base_url=configured)


@pytest.mark.unit
def test_unknown_setting_is_rejected() -> None:
    with pytest.raises(ValidationError):
        build(unknown_knob="value")


@pytest.mark.unit
@pytest.mark.parametrize("field", ["inference_timeout_s", "max_input_chars", "max_summary_tokens"])
def test_positive_bounds_are_enforced(field: str) -> None:
    with pytest.raises(ValidationError):
        build(**{field: 0})


@pytest.mark.unit
def test_settings_are_immutable() -> None:
    settings = build()

    with pytest.raises(ValidationError):
        settings.debug = True  # type: ignore[misc]


@pytest.mark.unit
def test_registry_root_is_a_path() -> None:
    assert isinstance(build().registry_root, Path)


@pytest.mark.unit
def test_production_flag_follows_the_environment() -> None:
    assert build(environment="production").is_production is True


@pytest.mark.unit
def test_secrets_are_not_leaked_by_repr() -> None:
    settings = build(s3_secret_key="super-secret-value")

    assert "super-secret-value" not in repr(settings)
    assert settings.s3_secret_key is not None
    assert settings.s3_secret_key.get_secret_value() == "super-secret-value"


@pytest.mark.unit
def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
