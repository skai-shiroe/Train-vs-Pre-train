"""Unit tests for the OpenAPI export of section 22."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.export_openapi import (
    build_document,
    build_environment,
    default_settings,
    main,
    write_document,
)

pytestmark = pytest.mark.unit

PREFIX = "/api/v1"


def test_the_document_describes_every_route() -> None:
    paths = build_document()["paths"]

    assert set(paths) == {
        f"{PREFIX}/health",
        f"{PREFIX}/ready",
        f"{PREFIX}/models",
        f"{PREFIX}/models/{{name}}",
        f"{PREFIX}/predict",
        f"{PREFIX}/compare",
    }


def test_the_errors_of_section_twenty_six_are_documented() -> None:
    responses = build_document()["paths"][f"{PREFIX}/predict"]["post"]["responses"]

    assert set(responses) == {"200", "404", "422", "500", "503", "504"}
    assert responses["504"]["description"] == "TIMEOUT"


def test_the_environment_does_not_change_the_document(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTRA_API_V1_PREFIX", "/somewhere-else")
    monkeypatch.setenv("SYNTRA_APP_NAME", "Autre")

    settings = default_settings()

    assert settings.api_v1_prefix == PREFIX
    assert settings.app_name == "Syntra"


def test_the_export_is_stable_between_two_runs(tmp_path: Path) -> None:
    first = write_document(build_document(), tmp_path / "first.json")
    second = write_document(build_document(), tmp_path / "second.json")

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_the_file_is_sorted_and_ends_with_a_newline(tmp_path: Path) -> None:
    written = write_document({"b": 1, "a": {"d": 2, "c": 3}}, tmp_path / "doc.json")
    content = written.read_text(encoding="utf-8")

    assert content == json.dumps({"a": {"c": 3, "d": 2}, "b": 1}, indent=2) + "\n"


def test_the_command_writes_where_it_was_told(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "openapi.json"
    environment = tmp_path / "nested" / "environment.json"

    assert main(["--output", str(output), "--environment-output", str(environment)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["info"]["title"] == "Syntra"
    assert json.loads(environment.read_text(encoding="utf-8"))["name"] == "Syntra local"


def test_the_document_carries_the_origin_of_the_instance() -> None:
    servers = build_document()["servers"]

    assert [server["url"] for server in servers] == [default_settings().public_base_url]


def test_every_error_leaves_through_the_envelope_of_section_twenty_six() -> None:
    document = build_document()

    documented = {
        response["content"]["application/json"]["schema"]["$ref"]
        for operation in document["paths"].values()
        for method in operation.values()
        for status, response in method["responses"].items()
        if status != "200"
    }

    # The default 422 of FastAPI is HTTPValidationError, which no handler of the
    # application ever returns. Its absence is what the check is about.
    assert documented == {"#/components/schemas/ErrorResponse"}
    assert "HTTPValidationError" not in document["components"]["schemas"]


def test_the_operations_are_named_after_their_endpoint() -> None:
    document = build_document()

    identifiers = {
        method["operationId"]
        for operation in document["paths"].values()
        for method in operation.values()
    }

    assert identifiers == {"health", "ready", "list_models", "list_versions", "predict", "compare"}


def test_the_bodies_carry_examples_that_can_be_sent_as_they_stand() -> None:
    document = build_document()

    for route in (f"{PREFIX}/predict", f"{PREFIX}/compare"):
        examples = document["paths"][route]["post"]["requestBody"]["content"]["application/json"][
            "examples"
        ]

        assert examples, f"{route} carries no example"
        for name, example in examples.items():
            assert example["value"]["text"].strip(), f"{route}/{name} would be refused as blank"

    # The tools that read the document offer the first example of the map, and
    # the export sorts the keys, so the minimal request has to sort first.
    predict = document["paths"][f"{PREFIX}/predict"]["post"]
    first = next(iter(predict["requestBody"]["content"]["application/json"]["examples"]))

    assert first == "defaut_echantillonne"


def test_the_postman_environment_is_built_from_the_same_settings() -> None:
    settings = default_settings()

    values = {entry["key"]: entry["value"] for entry in build_environment()["values"]}

    assert values == {
        "baseUrl": settings.public_base_url,
        "apiPrefix": settings.api_v1_prefix,
    }


def test_the_environment_identifier_does_not_move_between_two_runs() -> None:
    assert build_environment()["id"] == build_environment()["id"]
