"""Run one model over one split of the frozen corpus and score it.

This is the single path both sides of the comparison take. It receives a
:class:`src.evaluation.protocol.Summarizer`, never a model class, so the from
scratch Transformer and ``t5-small`` are measured by the same code, with the
same decoding budget, on the same examples.

Four things are enforced here rather than left to the caller.

**The decoding is explicit.** The evaluator always passes a
:class:`src.models.generation.GenerationConfig` instead of letting each model
fall back on its own default. What ends up in the run record is then what was
actually used, not what the record assumed.

**The prediction count is checked.** A summariser that returned one summary
short would shift every following prediction onto the wrong reference and still
produce a corpus score that looks ordinary. The mismatch is refused.

**A partial run says so.** ``--limit`` exists to exercise the chain without
waiting for a thousand documents. The result carries ``partial``, the run
directory is suffixed, and the artefacts cannot be mistaken for a full
measurement of the split. Section 44 treats a presented but unmeasured number
as a fabricated result, and a score over fifty documents presented as the score
of the test set is exactly that.

**Nothing is averaged across runs here.** One evaluation writes its own
directory. The aggregation into ``reports/results/experiments.csv`` and the
ablation tables belongs with the experiment runner, which knows what the runs
were.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from src.data.config import load_pipeline_config
from src.data.dataset import SPLIT_NAMES, load_split
from src.data.example import Example
from src.evaluation.protocol import Summarizer
from src.evaluation.qualitative import (
    QualitativeSelection,
    ScoredPrediction,
    score_predictions,
    select_qualitative,
)
from src.evaluation.summarization import SummarizationReport, evaluate_summaries
from src.metrics.rouge import RougeConfig, score_all
from src.models.generation import GenerationConfig
from src.models.pretrained.base import BaselineConfig
from src.models.pretrained.factory import available_baselines, build_summarizer
from src.utils.device import resolve_device
from src.utils.seed import DEFAULT_SEED, set_seed

#: File names written under the run directory.
METRICS_FILE = "metrics.json"
PREDICTIONS_FILE = "predictions.jsonl"
QUALITATIVE_FILE = "qualitative.json"

#: Where an evaluation writes, per section 17.
DEFAULT_RESULTS_DIR = Path("reports") / "results"


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Everything one evaluation produced.

    Attributes:
        model: Description returned by the summariser.
        split: Name of the split that was scored.
        split_size: Number of examples the split holds.
        generation: Decoding configuration actually used.
        rouge: Measurement settings actually used.
        report: Corpus level scores and length distributions.
        qualitative: Examples selected for the qualitative analysis.
        predictions: Every prediction with its per example scores.
        duration_seconds: Wall clock time spent generating and scoring.
    """

    model: dict[str, str]
    split: str
    split_size: int
    generation: GenerationConfig
    rouge: RougeConfig
    report: SummarizationReport
    qualitative: QualitativeSelection
    predictions: tuple[ScoredPrediction, ...]
    duration_seconds: float

    @property
    def partial(self) -> bool:
        """Return whether the run covered less than the whole split.

        Returns:
            ``True`` when fewer examples were scored than the split holds.
        """
        return self.report.size != self.split_size

    def to_dict(self) -> dict[str, Any]:
        """Render the run record, predictions excluded.

        The predictions go to their own file: a metrics document that a reader
        has to scroll past a thousand summaries to reach is not read.

        Returns:
            A JSON serialisable mapping.
        """
        return {
            "model": dict(self.model),
            "split": self.split,
            "split_size": self.split_size,
            "partial": self.partial,
            "generation": self.generation.to_dict(),
            "rouge_config": self.rouge.to_dict(),
            "report": self.report.to_dict(),
            "duration_seconds": round(self.duration_seconds, 3),
        }


def evaluate_summarizer(
    summarizer: Summarizer,
    examples: Sequence[Example],
    *,
    split: str = "test",
    split_size: int | None = None,
    generation: GenerationConfig | None = None,
    rouge: RougeConfig | None = None,
    batch_size: int = 8,
) -> EvaluationResult:
    """Summarise a split with one model and score the result.

    Args:
        summarizer: Any model satisfying the shared interface.
        examples: The examples to summarise, in corpus order.
        split: Name of the split, carried into the run record.
        split_size: Number of examples the full split holds. Defaults to the
            number supplied, which is the right answer for a complete run.
        generation: Decoding configuration. Defaults to the project defaults,
            and is recorded either way.
        rouge: Measurement settings.
        batch_size: Number of documents encoded at once.

    Returns:
        The evaluation result.

    Raises:
        ValueError: If the split is empty, if ``split_size`` is smaller than
            the number of examples supplied, or if the summariser returned a
            number of predictions that does not match the number of documents.
    """
    if not examples:
        raise ValueError("Cannot evaluate an empty split.")

    total = split_size if split_size is not None else len(examples)
    if total < len(examples):
        raise ValueError(f"The split holds {total} examples but {len(examples)} were supplied.")

    decoding = generation or GenerationConfig()
    measurement = rouge or RougeConfig()

    documents = [example.source for example in examples]
    references = [example.target for example in examples]

    started = perf_counter()
    predictions = summarizer.summarize(documents, decoding, batch_size=batch_size)
    if len(predictions) != len(documents):
        raise ValueError(
            f"The summariser returned {len(predictions)} summaries for {len(documents)} "
            "documents. Every following prediction would be scored against the wrong "
            "reference."
        )

    per_example = score_all(predictions, references, measurement)
    duration = perf_counter() - started

    scored = score_predictions(examples, predictions, per_example)
    return EvaluationResult(
        model=summarizer.describe(),
        split=split,
        split_size=total,
        generation=decoding,
        rouge=measurement,
        report=evaluate_summaries(
            predictions, references, per_example=per_example, rouge_config=measurement
        ),
        qualitative=select_qualitative(scored, seed=measurement.seed),
        predictions=tuple(scored),
        duration_seconds=duration,
    )


def write_results(result: EvaluationResult, directory: Path) -> Path:
    """Write the three artefacts of one evaluation.

    Args:
        result: The evaluation to persist.
        directory: Destination directory. Created when it does not exist.

    Returns:
        The directory written.
    """
    directory.mkdir(parents=True, exist_ok=True)

    (directory / METRICS_FILE).write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (directory / QUALITATIVE_FILE).write_text(
        json.dumps(result.qualitative.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with (directory / PREDICTIONS_FILE).open("w", encoding="utf-8", newline="\n") as handle:
        for prediction in result.predictions:
            payload = prediction.to_dict(include_source=False)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return directory


def format_summary(result: EvaluationResult) -> str:
    """Render the one screen summary printed at the end of a run.

    Args:
        result: The evaluation to describe.

    Returns:
        A short multi line report.
    """
    lines = [
        f"model            {result.model.get('model', result.model.get('baseline', 'unknown'))}"
        f" ({result.model.get('mode', 'unknown')})",
        f"split            {result.split}, {result.report.size} of {result.split_size} examples",
        f"duration         {result.duration_seconds:.1f} s",
    ]
    for variant, score in result.report.rouge.scores.items():
        interval = result.report.rouge.intervals.get(variant)
        bounds = f"  [{interval.low:.4f}, {interval.high:.4f}]" if interval else ""
        lines.append(f"{variant:<16} {score.fmeasure:.4f}{bounds}")

    lines.append(f"empty summaries  {result.report.empty_predictions}")
    if result.partial:
        lines.append("PARTIAL RUN: this score covers a subset of the split and is not a result.")
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    """Describe the command line of ``python -m src.evaluation.evaluator``.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.evaluator",
        description=(
            "Evaluate a pretrained baseline on a split of the frozen working corpus. "
            "Evaluating a from scratch run needs its checkpoint, which the experiment "
            "runner produces."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs") / "data" / "cnn_dailymail.yaml",
        help="Data pipeline configuration, which locates the corpus and the tokeniser.",
    )
    parser.add_argument(
        "--baseline",
        default="t5",
        choices=available_baselines(),
        help="Registered baseline to evaluate.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory holding fine tuned weights. Defaults to the hub checkpoint.",
    )
    parser.add_argument(
        "--split", default="test", choices=SPLIT_NAMES, help="Split to evaluate on."
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Documents encoded at once.")
    parser.add_argument("--num-beams", type=int, default=1, help="Beam width. One is greedy.")
    parser.add_argument(
        "--max-new-tokens", type=int, default=64, help="Generated token budget per summary."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate the first N examples only. Marks the run as partial.",
    )
    parser.add_argument("--device", default="auto", help="Device: auto, cpu or cuda.")
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Seed applied before the run."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory the run directory is created under.",
    )
    parser.add_argument("--name", default=None, help="Run directory name. Derived when omitted.")
    return parser


def run_name(model: dict[str, str], *, partial: bool) -> str:
    """Derive the run directory name from the model description.

    Args:
        model: Description returned by the summariser.
        partial: Whether the run covered a subset of the split.

    Returns:
        A name safe to use as a directory, suffixed when the run is partial so
        that it can never overwrite a complete measurement.
    """
    identity = model.get("baseline") or model.get("model") or "model"
    name = f"{identity}_{model.get('mode', 'unknown')}"
    return f"{name}_partial" if partial else name


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate a baseline from the command line.

    Args:
        argv: Arguments to parse. Defaults to the process arguments.

    Returns:
        Process exit code.
    """
    args = build_argument_parser().parse_args(argv)

    config = load_pipeline_config(args.config)
    examples = load_split(config.paths.processed, args.split)
    split_size = len(examples)
    if args.limit is not None:
        examples = examples[: args.limit]

    set_seed(args.seed)
    baseline_config = BaselineConfig.from_tokenizer_config(
        config.tokenizer, hf_id=str(args.model_dir) if args.model_dir else None
    )
    summarizer = build_summarizer(
        args.baseline, baseline_config, fine_tuned=args.model_dir is not None
    ).to(resolve_device(args.device))

    result = evaluate_summarizer(
        summarizer,
        examples,
        split=args.split,
        split_size=split_size,
        generation=GenerationConfig(max_new_tokens=args.max_new_tokens, num_beams=args.num_beams),
        rouge=RougeConfig(seed=args.seed),
        batch_size=args.batch_size,
    )

    directory = write_results(
        result, Path(args.output) / (args.name or run_name(result.model, partial=result.partial))
    )
    print(format_summary(result))
    print(f"written to       {directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
