"""Unit tests for the text cleaning stage."""

from __future__ import annotations

import pytest

from src.data.config import PreprocessConfig
from src.data.example import Example
from src.data.preprocess import (
    clean_text,
    is_within_bounds,
    preprocess,
    preprocess_example,
)


@pytest.fixture
def config() -> PreprocessConfig:
    return PreprocessConfig(
        min_source_chars=10,
        max_source_chars=100,
        min_target_chars=5,
        max_target_chars=50,
    )


@pytest.mark.unit
def test_runs_of_whitespace_collapse(config: PreprocessConfig) -> None:
    assert clean_text("a   b\n\n\tc", config) == "a b c"


@pytest.mark.unit
def test_leading_and_trailing_whitespace_is_stripped(config: PreprocessConfig) -> None:
    assert clean_text("   texte   ", config) == "texte"


@pytest.mark.unit
def test_control_characters_are_removed(config: PreprocessConfig) -> None:
    assert clean_text("a\x00b\x07c", config) == "a b c"


@pytest.mark.unit
def test_newlines_and_tabs_survive_as_spaces(config: PreprocessConfig) -> None:
    assert clean_text("ligne1\nligne2\tligne3", config) == "ligne1 ligne2 ligne3"


@pytest.mark.unit
def test_unicode_is_normalised(config: PreprocessConfig) -> None:
    # U+FB01 is the ligature fi, decomposed by NFKC into two ASCII letters.
    assert clean_text("ﬁn", config) == "fin"


@pytest.mark.unit
def test_normalisation_can_be_disabled(config: PreprocessConfig) -> None:
    without = config.model_copy(update={"normalise_unicode": False})

    assert clean_text("ﬁn", without) == "ﬁn"


@pytest.mark.unit
def test_whitespace_collapsing_can_be_disabled(config: PreprocessConfig) -> None:
    without = config.model_copy(update={"collapse_whitespace": False})

    assert clean_text("a   b", without) == "a   b"


@pytest.mark.unit
def test_identifier_survives_cleaning(config: PreprocessConfig) -> None:
    cleaned = preprocess_example(Example("id-7", "  a   b  ", " resume "), config)

    assert cleaned.example_id == "id-7"
    assert cleaned.source == "a b"
    assert cleaned.target == "resume"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        ("x" * 50, "y" * 20, True),
        ("x" * 9, "y" * 20, False),
        ("x" * 101, "y" * 20, False),
        ("x" * 50, "y" * 4, False),
        ("x" * 50, "y" * 51, False),
        ("x" * 10, "y" * 5, True),
        ("x" * 100, "y" * 50, True),
    ],
)
def test_bounds_are_inclusive(
    config: PreprocessConfig, source: str, target: str, expected: bool
) -> None:
    assert is_within_bounds(Example("id", source, target), config) is expected


@pytest.mark.unit
def test_examples_outside_the_bounds_are_dropped(config: PreprocessConfig) -> None:
    examples = [
        Example("keep", "x" * 50, "y" * 20),
        Example("too-short-source", "x" * 5, "y" * 20),
        Example("too-long-target", "x" * 50, "y" * 80),
    ]

    kept = list(preprocess(examples, config))

    assert [example.example_id for example in kept] == ["keep"]


@pytest.mark.unit
def test_length_is_judged_after_cleaning(config: PreprocessConfig) -> None:
    # 40 spaces plus 5 letters: rejected before cleaning, accepted after, but
    # 5 characters is below the source minimum, so it must still be dropped.
    padded = Example("padded", " " * 40 + "abcde", "y" * 20)

    assert list(preprocess([padded], config)) == []


@pytest.mark.unit
def test_cleaning_is_idempotent(config: PreprocessConfig) -> None:
    once = clean_text("  a \n b  ", config)

    assert clean_text(once, config) == once
