"""End to end tests of the API, from the HTTP request down to the weights."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.core.errors import ErrorCode

PREFIX = "/api/v1"
DOCUMENT = "Le conseil municipal a vote la renovation du pont ce lundi soir."

pytestmark = pytest.mark.e2e


# -- Probes -----------------------------------------------------------------


def test_the_instance_reports_itself_ready(api: TestClient) -> None:
    response = api.get(f"{PREFIX}/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["models_loaded"] == ["scratch:v2", "scratch_deep:v1"]
    assert body["models_failed"] == {}


def test_startup_loads_what_each_name_serves_and_nothing_else(api: TestClient) -> None:
    # v1 is published and promoted to champion, but v2 is what the name serves.
    # Warming loads the served version; the champion is loaded on first use.
    loaded = api.get(f"{PREFIX}/ready").json()["models_loaded"]

    assert "scratch:v1" not in loaded


# -- The catalogue ----------------------------------------------------------


def test_the_catalogue_reports_the_measurement_of_each_model(api: TestClient) -> None:
    body = api.get(f"{PREFIX}/models").json()

    assert [entry["name"] for entry in body["models"]] == ["scratch", "scratch_deep"]
    served = body["models"][0]
    assert served["version"] == "v2"
    assert served["experiment"] == "scratch_100"
    assert served["metrics"]["rouge1_f"] == pytest.approx(0.22)
    assert served["loaded"] is True
    assert body["aliases"] == {"champion": "scratch:v1"}
    assert body["default"] == "champion"


def test_the_history_of_a_name_holds_every_version(api: TestClient) -> None:
    body = api.get(f"{PREFIX}/models/scratch").json()

    assert [entry["version"] for entry in body["versions"]] == ["v1", "v2"]
    assert body["name"] == "scratch"


def test_an_unknown_name_is_a_missing_model(api: TestClient) -> None:
    response = api.get(f"{PREFIX}/models/absent")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == ErrorCode.MODEL_NOT_FOUND.value


def test_an_alias_is_not_a_name(api: TestClient) -> None:
    assert api.get(f"{PREFIX}/models/champion").status_code == 404


# -- Prediction -------------------------------------------------------------


def test_a_prediction_is_served_by_the_champion_by_default(api: TestClient) -> None:
    response = api.post(f"{PREFIX}/predict", json={"text": DOCUMENT})

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "scratch"
    assert body["version"] == "v1"
    assert isinstance(body["summary"], str)
    assert body["latency_ms"] >= 0.0
    # Asked for the budget of the instance, capped to what these weights were
    # trained to produce, and reported as what actually ran. The seed is drawn
    # by the instance, so it is checked for presence rather than for a value.
    decoding = body["decoding"]
    assert decoding["max_new_tokens"] == 8
    assert decoding["num_beams"] == 1
    assert decoding["no_repeat_ngram_size"] == 3
    assert decoding["do_sample"] is True
    assert isinstance(decoding["seed"], int)


def test_a_request_may_pin_an_exact_version(api: TestClient) -> None:
    body = api.post(
        f"{PREFIX}/predict", json={"text": DOCUMENT, "model": "scratch", "version": "v1"}
    ).json()

    assert body["version"] == "v1"


def test_a_name_without_a_version_is_served_by_its_default(api: TestClient) -> None:
    body = api.post(f"{PREFIX}/predict", json={"text": DOCUMENT, "model": "scratch"}).json()

    assert body["version"] == "v2"


def test_the_decoding_asked_for_is_the_decoding_reported(api: TestClient) -> None:
    body = api.post(
        f"{PREFIX}/predict",
        json={"text": DOCUMENT, "max_new_tokens": 8, "do_sample": False, "num_beams": 4},
    ).json()

    # A deterministic run reports no sampling parameters at all: there was no
    # temperature and no seed, and filling them would describe a run that did
    # not take place.
    assert body["decoding"] == {
        "max_new_tokens": 8,
        "num_beams": 4,
        "no_repeat_ngram_size": 3,
        "do_sample": False,
        "temperature": None,
        "top_k": None,
        "top_p": None,
        "seed": None,
    }


def test_two_identical_requests_give_two_different_summaries(api: TestClient) -> None:
    # The default draws, and draws a fresh seed each time. Ten attempts make an
    # accidental collision on this tiny vocabulary vanishingly unlikely.
    summaries = {
        api.post(f"{PREFIX}/predict", json={"text": DOCUMENT}).json()["summary"] for _ in range(10)
    }

    assert len(summaries) > 1


def test_a_deterministic_request_gives_the_same_summary_twice(api: TestClient) -> None:
    payload = {"text": DOCUMENT, "do_sample": False}
    first = api.post(f"{PREFIX}/predict", json=payload).json()
    second = api.post(f"{PREFIX}/predict", json=payload).json()

    assert first["summary"] == second["summary"]


def test_replaying_the_reported_seed_replays_the_summary(api: TestClient) -> None:
    # The promise the decoding block makes: a sampled answer is reproducible
    # from the answer alone.
    first = api.post(f"{PREFIX}/predict", json={"text": DOCUMENT}).json()
    replayed = api.post(
        f"{PREFIX}/predict", json={"text": DOCUMENT, "seed": first["decoding"]["seed"]}
    ).json()

    assert replayed["summary"] == first["summary"]
    assert replayed["decoding"]["seed"] == first["decoding"]["seed"]


def test_sampling_next_to_a_beam_search_is_refused(api: TestClient) -> None:
    response = api.post(f"{PREFIX}/predict", json={"text": DOCUMENT, "num_beams": 4})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == ErrorCode.INVALID_INPUT.value


def test_an_unknown_model_is_refused_before_any_generation(api: TestClient) -> None:
    response = api.post(f"{PREFIX}/predict", json={"text": DOCUMENT, "model": "absent"})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == ErrorCode.MODEL_NOT_FOUND.value


def test_an_unknown_version_is_refused(api: TestClient) -> None:
    response = api.post(
        f"{PREFIX}/predict", json={"text": DOCUMENT, "model": "scratch", "version": "v9"}
    )

    assert response.status_code == 404


def test_a_document_over_the_limit_is_refused(api: TestClient) -> None:
    response = api.post(f"{PREFIX}/predict", json={"text": "a" * 501})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == ErrorCode.INVALID_INPUT.value


# -- Comparison -------------------------------------------------------------


def test_a_comparison_runs_every_registered_model_by_default(api: TestClient) -> None:
    body = api.post(f"{PREFIX}/compare", json={"text": DOCUMENT}).json()

    assert [entry["model"] for entry in body["results"]] == ["scratch", "scratch_deep"]
    assert body["scored"] is False
    assert all(entry["rouge"] is None for entry in body["results"])


def test_a_comparison_may_name_its_models(api: TestClient) -> None:
    body = api.post(f"{PREFIX}/compare", json={"text": DOCUMENT, "models": ["scratch_deep"]}).json()

    assert [entry["model"] for entry in body["results"]] == ["scratch_deep"]


def test_a_reference_produces_the_three_variants_of_section_two(api: TestClient) -> None:
    body = api.post(
        f"{PREFIX}/compare",
        json={"text": DOCUMENT, "reference": "Le pont sera renove."},
    ).json()

    assert body["scored"] is True
    for entry in body["results"]:
        assert set(entry["rouge"]) == {"rouge1", "rouge2", "rougeL"}
        assert set(entry["rouge"]["rouge1"]) == {"precision", "recall", "fmeasure"}


def test_every_model_of_a_comparison_shares_one_decoding(api: TestClient) -> None:
    body = api.post(
        f"{PREFIX}/compare", json={"text": DOCUMENT, "do_sample": False, "num_beams": 2}
    ).json()

    assert body["decoding"]["num_beams"] == 2


def test_a_sampled_comparison_runs_every_model_under_one_seed(api: TestClient) -> None:
    # One seed for the whole request, not one per model: two draws under two
    # seeds would be two conditions, and the gap would say nothing.
    body = api.post(f"{PREFIX}/compare", json={"text": DOCUMENT}).json()

    assert body["decoding"]["do_sample"] is True
    assert isinstance(body["decoding"]["seed"], int)


def test_the_same_model_twice_is_refused(api: TestClient) -> None:
    response = api.post(
        f"{PREFIX}/compare", json={"text": DOCUMENT, "models": ["scratch", "scratch"]}
    )

    assert response.status_code == 422


# -- The envelope -----------------------------------------------------------


def test_an_error_is_returned_in_the_envelope_of_section_twenty_six(api: TestClient) -> None:
    body = api.post(f"{PREFIX}/predict", json={"text": DOCUMENT, "model": "absent"}).json()

    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "details"}


def test_the_specification_is_served_by_the_instance(api: TestClient) -> None:
    document = api.get("/openapi.json").json()

    assert f"{PREFIX}/predict" in document["paths"]
    assert document["info"]["version"] == "1.0.0"
