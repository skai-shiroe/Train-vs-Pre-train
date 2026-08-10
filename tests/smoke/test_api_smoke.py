"""Smoke tests, run against a deployed instance.

Section 17 requires a deployment to be verified by requests against the real
environment rather than against a mock, and the four checks below are the ones
the test pyramid lists. They are driven by ``BASE_URL`` and skipped when it
is not set, so ``make test`` collects them without needing a server.

**Nothing here is mocked, and nothing here is loaded.** A smoke test that built
a fixture would be testing the fixture. What these assert is that the instance
answers, that it holds the models it was deployed with, and that a request goes
through end to end in a reasonable time.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import httpx
import pytest

pytestmark = pytest.mark.smoke

#: Prefix the routes are deployed behind. A deployment that moves it also sets
#: this, the same way it sets the base URL.
PREFIX = os.environ.get("API_PREFIX", "/api/v1")

#: Budget of one smoke run. Section 17 asks for under a minute in total, so a
#: single request that hangs must fail rather than hold the pipeline.
TIMEOUT_S = 20.0

DOCUMENT = (
    "The city council approved the renovation of the old bridge on Monday "
    "evening, after two years of debate about its cost and its safety."
)


@pytest.fixture(scope="session")
def client() -> Iterator[httpx.Client]:
    """Return a client pointed at the deployed instance."""
    base_url = os.environ.get("BASE_URL")
    if not base_url:
        pytest.skip("BASE_URL is not set, no deployed environment to smoke test.")

    with httpx.Client(base_url=base_url.rstrip("/"), timeout=TIMEOUT_S) as opened:
        yield opened


def test_the_instance_is_alive(client: httpx.Client) -> None:
    response = client.get(f"{PREFIX}/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_the_models_are_loaded(client: httpx.Client) -> None:
    response = client.get(f"{PREFIX}/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["models_loaded"], "Aucun modele n'est charge sur l'instance deployee."
    assert body["models_failed"] == {}


def test_the_catalogue_is_not_empty(client: httpx.Client) -> None:
    body = client.get(f"{PREFIX}/models").json()

    assert body["models"], "Le registre de l'instance deployee ne contient aucun modele."
    assert body["aliases"].get("champion"), "Aucune version n'est promue championne."


def test_a_prediction_comes_back_measured(client: httpx.Client) -> None:
    response = client.post(f"{PREFIX}/predict", json={"text": DOCUMENT})

    assert response.status_code == 200
    body = response.json()
    assert body["summary"].strip(), "Le modele deploye a renvoye un resume vide."
    assert body["latency_ms"] > 0.0
    assert body["version"].startswith("v")
