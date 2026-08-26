"""Unit tests of the evaluator.

The summariser is a stub here: what is checked is the path a score takes from a
list of documents to the files on disk, not what any model produces.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from transformers import T5ForConditionalGeneration

from src.data.config import TokenizerConfig
from src.data.dataset import write_jsonl
from src.data.example import Example
from src.evaluation.evaluator import (
    METRICS_FILE,
    PREDICTIONS_FILE,
    QUALITATIVE_FILE,
    EvaluationResult,
    build_argument_parser,
    evaluate_summarizer,
    format_summary,
    main,
    run_name,
    write_results,
)
from src.evaluation.protocol import Summarizer
from src.metrics.rouge import RougeConfig
from src.models.generation import GenerationConfig
from src.models.pretrained.base import BaselineConfig
from src.models.pretrained.t5 import T5Summarizer
from tests.conftest import FakeTokenizer

NO_BOOTSTRAP = RougeConfig(bootstrap_samples=0)

REFERENCES = [
    "the cat sat on the mat",
    "the dog slept on the rug",
    "the bird sang in the tree",
    "the fox crossed the empty road",
]


class StubSummarizer:
    """Return a canned summary per document and record how it was called."""

    def __init__(self, summaries: Sequence[str], *, mode: str = "zero_shot") -> None:
        self._summaries = list(summaries)
        self._mode = mode
        self.seen_config: GenerationConfig | None = None
        self.seen_batch_size: int | None = None
        self.seen_documents: list[str] = []

    def describe(self) -> dict[str, str]:
        return {"baseline": "stub", "mode": self._mode}

    def summarize(
        self,
        documents: Sequence[str],
        config: GenerationConfig | None = None,
        *,
        batch_size: int = 8,
    ) -> list[str]:
        self.seen_config = config
        self.seen_batch_size = batch_size
        self.seen_documents = list(documents)
        return list(self._summaries)


@pytest.fixture
def examples() -> list[Example]:
    return [
        Example(example_id=f"id-{index}", source=f"document {index}", target=reference)
        for index, reference in enumerate(REFERENCES)
    ]


@pytest.fixture
def result(examples: list[Example]) -> EvaluationResult:
    return evaluate_summarizer(
        StubSummarizer(REFERENCES), examples, split="test", rouge=NO_BOOTSTRAP
    )


# ---------------------------------------------------------------------------
# The shared interface
# ---------------------------------------------------------------------------


def test_both_sides_of_the_comparison_satisfy_the_shared_interface(
    fake_tokenizer: FakeTokenizer, tiny_t5: Callable[..., T5ForConditionalGeneration]
) -> None:
    """One evaluator path means neither model gets a branch of its own."""
    tokenizer_config = TokenizerConfig(
        hf_id="t5-small", source_prefix="summarize: ", max_source_tokens=32, max_target_tokens=8
    )
    random_init = T5Summarizer(
        BaselineConfig.from_tokenizer_config(tokenizer_config),
        tiny_t5(),
        fake_tokenizer,  # type: ignore[arg-type]
        pretrained=False,
    )
    pretrained = T5Summarizer(
        BaselineConfig(hf_id="t5-small", source_prefix="summarize: "),
        tiny_t5(),
        fake_tokenizer,  # type: ignore[arg-type]
    )

    assert isinstance(random_init, Summarizer)
    assert isinstance(pretrained, Summarizer)


# ---------------------------------------------------------------------------
# Running an evaluation
# ---------------------------------------------------------------------------


def test_a_perfect_summariser_scores_one(result: EvaluationResult) -> None:
    assert result.report.size == 4
    assert result.report.reported == pytest.approx(1.0)
    assert result.model == {"baseline": "stub", "mode": "zero_shot"}
    assert result.split == "test"


def test_the_decoding_is_passed_explicitly_rather_than_left_to_the_model(
    examples: list[Example],
) -> None:
    """What lands in the run record has to be what was actually used."""
    summarizer = StubSummarizer(REFERENCES)
    decoding = GenerationConfig(max_new_tokens=17, num_beams=3)

    outcome = evaluate_summarizer(
        summarizer, examples, generation=decoding, rouge=NO_BOOTSTRAP, batch_size=2
    )

    assert summarizer.seen_config == decoding
    assert summarizer.seen_batch_size == 2
    assert outcome.generation == decoding


def test_the_documents_reach_the_model_without_their_references(
    examples: list[Example],
) -> None:
    summarizer = StubSummarizer(REFERENCES)

    evaluate_summarizer(summarizer, examples, rouge=NO_BOOTSTRAP)

    assert summarizer.seen_documents == [example.source for example in examples]


def test_a_summariser_that_drops_a_summary_is_refused(examples: list[Example]) -> None:
    """One missing summary would shift every following prediction onto the wrong reference."""
    with pytest.raises(ValueError, match="wrong reference"):
        evaluate_summarizer(StubSummarizer(REFERENCES[:3]), examples, rouge=NO_BOOTSTRAP)


def test_an_empty_split_cannot_be_evaluated() -> None:
    with pytest.raises(ValueError, match="empty split"):
        evaluate_summarizer(StubSummarizer([]), [], rouge=NO_BOOTSTRAP)


def test_a_split_size_below_the_supplied_count_is_refused(examples: list[Example]) -> None:
    with pytest.raises(ValueError, match="holds 2 examples"):
        evaluate_summarizer(StubSummarizer(REFERENCES), examples, split_size=2, rouge=NO_BOOTSTRAP)


def test_a_complete_run_is_not_partial(result: EvaluationResult) -> None:
    assert result.partial is False
    assert result.split_size == 4


def test_a_subset_of_the_split_is_marked_partial(examples: list[Example]) -> None:
    """Section 44: a score over a subset presented as the score of the split is invented."""
    outcome = evaluate_summarizer(
        StubSummarizer(REFERENCES[:2]), examples[:2], split_size=1000, rouge=NO_BOOTSTRAP
    )

    assert outcome.partial is True
    assert outcome.to_dict()["partial"] is True
    assert "PARTIAL RUN" in format_summary(outcome)


def test_the_qualitative_selection_comes_out_of_the_same_run(result: EvaluationResult) -> None:
    assert result.qualitative.variant == "rougeL"
    assert len(result.predictions) == 4
    assert result.predictions[0].example_id == "id-0"


def test_the_run_record_renders_as_a_json_serialisable_mapping(
    result: EvaluationResult,
) -> None:
    payload = result.to_dict()

    assert payload["split_size"] == 4
    assert payload["generation"]["num_beams"] == 1
    assert payload["rouge_config"]["use_stemmer"] is True
    assert payload["duration_seconds"] >= 0.0
    assert json.dumps(payload)


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------


def test_the_three_artefacts_are_written(result: EvaluationResult, tmp_path: Path) -> None:
    directory = write_results(result, tmp_path / "run")

    assert (directory / METRICS_FILE).is_file()
    assert (directory / QUALITATIVE_FILE).is_file()
    assert (directory / PREDICTIONS_FILE).is_file()


def test_the_predictions_file_holds_one_record_per_example_without_the_document(
    result: EvaluationResult, tmp_path: Path
) -> None:
    directory = write_results(result, tmp_path / "run")

    lines = (directory / PREDICTIONS_FILE).read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]

    assert len(records) == 4
    assert records[0]["id"] == "id-0"
    assert "source" not in records[0]
    assert records[0]["scores"]["rougeL"] == pytest.approx(1.0)


def test_the_qualitative_file_keeps_the_documents(result: EvaluationResult, tmp_path: Path) -> None:
    directory = write_results(result, tmp_path / "run")
    payload = json.loads((directory / QUALITATIVE_FILE).read_text(encoding="utf-8"))

    assert payload["best"][0]["source"].startswith("document ")


def test_the_summary_lists_every_measured_variant(result: EvaluationResult) -> None:
    rendered = format_summary(result)

    assert "rouge1" in rendered
    assert "rouge2" in rendered
    assert "rougeL" in rendered
    assert "empty summaries  0" in rendered
    assert "PARTIAL RUN" not in rendered


def test_the_summary_shows_the_interval_when_it_was_measured(
    examples: list[Example],
) -> None:
    outcome = evaluate_summarizer(
        StubSummarizer(REFERENCES), examples, rouge=RougeConfig(bootstrap_samples=50)
    )

    assert "[" in format_summary(outcome)


# ---------------------------------------------------------------------------
# Run naming
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "partial", "expected"),
    [
        ({"baseline": "t5", "mode": "zero_shot"}, False, "t5_zero_shot"),
        ({"baseline": "t5", "mode": "fine_tuned"}, False, "t5_fine_tuned"),
        ({"model": "scratch", "mode": "trained"}, False, "scratch_trained"),
        ({"baseline": "t5", "mode": "zero_shot"}, True, "t5_zero_shot_partial"),
        ({}, False, "model_unknown"),
    ],
)
def test_the_run_directory_is_named_after_the_model_and_its_mode(
    model: dict[str, str], partial: bool, expected: str
) -> None:
    assert run_name(model, partial=partial) == expected


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def test_the_parser_defaults_to_a_greedy_run_on_the_test_split() -> None:
    args = build_argument_parser().parse_args([])

    assert args.split == "test"
    assert args.num_beams == 1
    assert args.baseline == "t5"
    assert args.limit is None


def write_corpus(root: Path) -> Path:
    """Write a tiny frozen corpus and the configuration that locates it."""
    processed = root / "processed"
    write_jsonl(
        processed / "test.jsonl",
        [
            Example(example_id=f"id-{index}", source=f"document {index}", target=reference)
            for index, reference in enumerate(REFERENCES)
        ],
    )

    payload: dict[str, Any] = {
        "name": "fixture",
        "dataset": {
            "hf_id": "fixture/corpus",
            "source_column": "document",
            "target_column": "summary",
        },
        "working_corpus": {
            "seed": 42,
            "train_size": 20,
            "validation_size": 4,
            "test_size": 4,
        },
        "proportions": [10, 100],
        "preprocess": {
            "min_source_chars": 1,
            "max_source_chars": 10000,
            "min_target_chars": 1,
            "max_target_chars": 500,
        },
        "tokenizer": {
            "hf_id": "t5-small",
            "source_prefix": "summarize: ",
            "max_source_tokens": 32,
            "max_target_tokens": 8,
        },
        "paths": {"raw": str(root / "raw"), "processed": str(processed)},
    }

    config_path = root / "corpus.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_the_command_evaluates_the_split_and_writes_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_corpus(tmp_path)
    summarizer = StubSummarizer(REFERENCES)
    monkeypatch.setattr(
        "src.evaluation.evaluator.build_summarizer",
        lambda *args, **kwargs: summarizer,
    )
    monkeypatch.setattr(StubSummarizer, "to", lambda self, device: self, raising=False)

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "results"),
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    run_directory = tmp_path / "results" / "stub_zero_shot"
    metrics = json.loads((run_directory / METRICS_FILE).read_text(encoding="utf-8"))
    assert metrics["split"] == "test"
    assert metrics["partial"] is False
    assert metrics["report"]["size"] == 4
    assert "rougeL" in capsys.readouterr().out


def test_a_limited_run_lands_in_its_own_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A partial run must not be able to overwrite a complete measurement."""
    config_path = write_corpus(tmp_path)
    summarizer = StubSummarizer(REFERENCES[:2])
    monkeypatch.setattr(
        "src.evaluation.evaluator.build_summarizer",
        lambda *args, **kwargs: summarizer,
    )
    monkeypatch.setattr(StubSummarizer, "to", lambda self, device: self, raising=False)

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "results"),
            "--device",
            "cpu",
            "--limit",
            "2",
        ]
    )

    assert exit_code == 0
    metrics_path = tmp_path / "results" / "stub_zero_shot_partial" / METRICS_FILE
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["partial"] is True
    assert metrics["split_size"] == 4
    assert metrics["report"]["size"] == 2
    assert "PARTIAL RUN" in capsys.readouterr().out
