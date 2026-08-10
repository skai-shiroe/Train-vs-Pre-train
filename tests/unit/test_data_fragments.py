"""Unit tests of the generated tables of the corpus page.

The fragments exist so that the page cannot describe a corpus other than the one
on disk. Five properties carry that, and each is a way the mechanism would fail
while still producing a page that reads plausibly: an absent record must stop
the run rather than render as an empty corpus, a quantity the record does not
hold must not render as a zero, a rendering must not depend on how the command
was invoked, a hand edit must be detected, and the fragments must carry the
command that actually rewrites them rather than the one the report tables use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.data.fragments import (
    CHECKSUMS_FRAGMENT,
    COMMAND,
    MANIFEST_NAME,
    STATISTICS_FRAGMENT,
    STATISTICS_NAME,
    TRUNCATION_FRAGMENT,
    build,
    checksums_table,
    code,
    cut_cell,
    main,
    read_record,
    statistics_table,
    truncation_table,
)
from src.utils.markdown import MISSING, banner, stale, write

pytestmark = pytest.mark.unit


def manifest_payload(**overrides: Any) -> dict[str, Any]:
    """Return a corpus manifest shaped like the one ``make data`` writes."""
    payload: dict[str, Any] = {
        "name": "xsum",
        "seed": 42,
        "tokenizer": "t5-small",
        "source_dataset": "EdinburghNLP/xsum",
        "dataset_version": "aaaa1111",
        "splits": {"train": 200, "validation": 10, "test": 10},
        "split_checksums": {"train": "bbbb2222", "validation": "cccc3333", "test": "dddd4444"},
        "proportions": {"10": 20, "50": 100, "100": 200},
    }
    payload.update(overrides)
    return payload


def lengths(mean: float, median: float, p95: float) -> dict[str, Any]:
    """Return one length distribution of the record."""
    return {"mean": mean, "median": median, "p95": p95, "minimum": 1, "maximum": 999}


def split_payload(size: int, *, ratio: float = 0.0966) -> dict[str, Any]:
    """Return the character and word statistics of one split."""
    return {
        "split": "train",
        "size": size,
        "source_chars": lengths(2198.47, 1751.0, 5416.05),
        "target_chars": lengths(125.64, 126.0, 176.0),
        "source_words": lengths(373.25, 296.5, 924.0),
        "target_words": lengths(21.13, 21.0, 30.0),
        "compression_ratio": ratio,
    }


def tokens_payload(truncated_sources: int, truncated_targets: int) -> dict[str, Any]:
    """Return the token statistics of one split."""
    return {
        "source_tokens": lengths(525.25, 414.0, 1305.0),
        "target_tokens": lengths(30.34, 30.0, 43.0),
        "truncated_sources": truncated_sources,
        "truncated_sources_ratio": 0.3843,
        "truncated_targets": truncated_targets,
        "truncated_targets_ratio": 0.0042,
    }


def statistics_payload(**overrides: Any) -> dict[str, Any]:
    """Return corpus statistics shaped like the ones ``make data`` writes."""
    payload: dict[str, Any] = {
        "characters_and_words": {
            "train": split_payload(200),
            "validation": split_payload(10, ratio=0.0984),
            "test": split_payload(10, ratio=0.0932),
        },
        "tokens": {
            "train": tokens_payload(76, 8),
            "validation": tokens_payload(4, 0),
            "test": tokens_payload(3, 1),
        },
    }
    payload.update(overrides)
    return payload


def prepare(directory: Path, *, manifest: Any = None, statistics: Any = None) -> Path:
    """Write a corpus record into a directory and return it."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest_payload() if manifest is None else manifest), encoding="utf-8"
    )
    (directory / STATISTICS_NAME).write_text(
        json.dumps(statistics_payload() if statistics is None else statistics), encoding="utf-8"
    )
    return directory


def test_read_record_refuses_a_missing_file(tmp_path: Path) -> None:
    """An absent corpus stops the run and names the command that builds one."""
    with pytest.raises(ValueError, match="make data"):
        read_record(tmp_path / MANIFEST_NAME)


def test_read_record_refuses_unreadable_json(tmp_path: Path) -> None:
    """A truncated record is a failure, not an empty corpus."""
    path = tmp_path / MANIFEST_NAME
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="readable JSON"):
        read_record(path)


def test_read_record_refuses_a_record_that_is_not_an_object(tmp_path: Path) -> None:
    """A list where a mapping belongs would render every cell empty in silence."""
    path = tmp_path / MANIFEST_NAME
    path.write_text("[1, 2]", encoding="utf-8")

    with pytest.raises(ValueError, match="expected an object"):
        read_record(path)


def test_statistics_table_reads_every_quantity_from_the_record() -> None:
    """The eight rows the page argues from, one column per split."""
    rendered = statistics_table(statistics_payload())
    lines = rendered.splitlines()

    assert lines[0] == "| Grandeur | Entraînement | Validation | Test |"
    assert "| Tokens par document, moyenne | 525,3 | 525,3 | 525,3 |" in lines
    assert "| Mots par document, médiane | 296,5 | 296,5 | 296,5 |" in lines
    assert "| Taux de compression | 0,097 | 0,098 | 0,093 |" in lines


def test_statistics_table_leaves_an_unmeasured_quantity_empty() -> None:
    """A record missing a series renders empty cells, never zeros."""
    statistics = statistics_payload()
    del statistics["tokens"]["train"]["source_tokens"]

    rendered = statistics_table(statistics)

    assert "| Tokens par document, moyenne |  | 525,3 | 525,3 |" in rendered.splitlines()


def test_statistics_table_leaves_an_undescribed_split_empty() -> None:
    """A split the record never described is not a corpus of zero examples."""
    statistics = statistics_payload()
    del statistics["characters_and_words"]["test"]
    del statistics["tokens"]["test"]

    rendered = statistics_table(statistics)

    assert "| Taux de compression | 0,097 | 0,098 |  |" in rendered.splitlines()


def test_statistics_table_ignores_a_section_that_is_not_a_mapping() -> None:
    """A record whose shape changed renders empty rather than raising on a page build."""
    rendered = statistics_table({"characters_and_words": "unexpected", "tokens": {}})

    assert "| Taux de compression |  |  |  |" in rendered.splitlines()


def test_a_boolean_is_not_read_as_a_measurement() -> None:
    """``True`` is an ``int`` in Python, and would render as a length of one."""
    statistics = statistics_payload()
    statistics["characters_and_words"]["train"]["compression_ratio"] = True

    rendered = statistics_table(statistics)

    assert "| Taux de compression |  | 0,098 | 0,093 |" in rendered.splitlines()


def test_truncation_table_counts_both_ceilings() -> None:
    """The page argues that one ceiling is expensive and the other is nearly free."""
    lines = truncation_table(statistics_payload()).splitlines()

    assert lines[0] == "| Split | Documents tronqués | Résumés tronqués |"
    assert "| Entraînement | 76 sur 200 (38,0 %) | 8 sur 200 (4,0 %) |" in lines
    assert "| Test | 3 sur 10 (30,0 %) | 1 sur 10 (10,0 %) |" in lines


def test_truncation_table_names_no_ceiling() -> None:
    """The record holds the counts and not the setting they were measured under."""
    assert "512" not in truncation_table(statistics_payload())
    assert "64" not in truncation_table(statistics_payload())


def test_cut_cell_is_empty_when_the_count_is_absent() -> None:
    """An unmeasured truncation is empty, so it cannot be read as nothing cut."""
    statistics = statistics_payload()
    del statistics["tokens"]["train"]["truncated_sources"]

    assert cut_cell(statistics, "train", "truncated_sources") == MISSING


def test_cut_cell_is_empty_when_the_split_holds_no_example() -> None:
    """A share of an empty split is not zero percent, it is no share at all."""
    statistics = statistics_payload()
    statistics["characters_and_words"]["train"]["size"] = 0

    assert cut_cell(statistics, "train", "truncated_sources") == MISSING


def test_cut_cell_keeps_the_count_beside_the_share() -> None:
    """A percentage alone hides how many examples it stands for."""
    assert cut_cell(statistics_payload(), "test", "truncated_sources") == "3 sur 10 (30,0 %)"


def test_checksums_table_renders_the_version_first() -> None:
    """The value tracked with every experiment, then what localises a change."""
    lines = checksums_table(manifest_payload()).splitlines()

    assert lines[2] == "| `dataset_version` | `aaaa1111` |"
    assert lines[3] == "| Entraînement | `bbbb2222` |"
    assert lines[5] == "| Test | `dddd4444` |"


def test_checksums_table_leaves_an_absent_fingerprint_empty() -> None:
    """An empty cell, rather than the word None a reader could take for a hash."""
    rendered = checksums_table(manifest_payload(split_checksums={"train": "bbbb2222"}))

    assert "| Validation |  |" in rendered.splitlines()
    assert "None" not in rendered


def test_checksums_table_survives_a_manifest_without_checksums() -> None:
    """A manifest whose shape changed must not raise while a page is being built."""
    rendered = checksums_table(manifest_payload(split_checksums="unexpected"))

    assert "| Entraînement |  |" in rendered.splitlines()


def test_code_renders_nothing_for_an_absent_value() -> None:
    """Empty backticks would render as an inline code span holding nothing."""
    assert code(None) == MISSING
    assert code("") == MISSING


def test_build_keys_three_fragments_under_the_output_directory(tmp_path: Path) -> None:
    """The three tables of the page, and nothing else."""
    processed = prepare(tmp_path / "corpus")
    output = tmp_path / "out"

    rendered = build(processed_dir=processed, output_dir=output)

    assert set(rendered) == {
        output / CHECKSUMS_FRAGMENT,
        output / STATISTICS_FRAGMENT,
        output / TRUNCATION_FRAGMENT,
    }


def test_every_fragment_names_the_command_that_rewrites_it(tmp_path: Path) -> None:
    """Pointing a reader at the generator of the report tables would waste their time."""
    rendered = build(processed_dir=prepare(tmp_path / "corpus"), output_dir=tmp_path / "out")

    for text in rendered.values():
        assert text.startswith(banner(COMMAND))
        assert "src.experiments.fragments" not in text


def test_rendering_does_not_depend_on_how_the_command_was_invoked(tmp_path: Path) -> None:
    """A fragment that changed with a relative path would disagree with itself."""
    processed = prepare(tmp_path / "corpus")

    absolute = build(processed_dir=processed.resolve(), output_dir=tmp_path / "out")
    relative = build(processed_dir=processed, output_dir=tmp_path / "out")

    assert list(absolute.values()) == list(relative.values())


def test_build_refuses_a_directory_holding_no_corpus(tmp_path: Path) -> None:
    """Rendering from nothing would rewrite the page as a corpus of empty cells."""
    with pytest.raises(ValueError, match="make data"):
        build(processed_dir=tmp_path / "absent", output_dir=tmp_path / "out")


def test_main_writes_the_fragments(tmp_path: Path) -> None:
    """The command puts on disk exactly what build renders."""
    processed = prepare(tmp_path / "corpus")
    output = tmp_path / "out"

    exit_code = main(["--processed", str(processed), "--output", str(output)])

    assert exit_code == 0
    assert (output / STATISTICS_FRAGMENT).is_file()
    assert stale(build(processed_dir=processed, output_dir=output)) == []


def test_main_reports_up_to_date_fragments(tmp_path: Path) -> None:
    """The check passes on a page that matches the corpus on disk."""
    processed = prepare(tmp_path / "corpus")
    output = tmp_path / "out"
    write(build(processed_dir=processed, output_dir=output))

    assert main(["--processed", str(processed), "--output", str(output), "--check"]) == 0


def test_main_fails_on_a_table_retyped_by_hand(tmp_path: Path) -> None:
    """The property the mechanism rests on, checked through the command line."""
    processed = prepare(tmp_path / "corpus")
    output = tmp_path / "out"
    write(build(processed_dir=processed, output_dir=output))
    (output / STATISTICS_FRAGMENT).write_text("| Grandeur |\n", encoding="utf-8")

    assert main(["--processed", str(processed), "--output", str(output), "--check"]) == 1


def test_main_fails_on_a_missing_fragment(tmp_path: Path) -> None:
    """A page whose table was never generated is out of date, not up to date."""
    processed = prepare(tmp_path / "corpus")

    assert main(["--processed", str(processed), "--output", str(tmp_path / "out"), "--check"]) == 1


def test_main_reports_a_missing_record_on_the_error_stream(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A machine without a corpus gets a reason and a non zero exit, not a blank page."""
    exit_code = main(["--processed", str(tmp_path / "absent"), "--output", str(tmp_path / "out")])

    assert exit_code == 1
    assert "make data" in capsys.readouterr().err
