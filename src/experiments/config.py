"""Typed configuration of one experiment.

Section 13 gives every experiment a file of its own, and section 14 requires
that file to be the only thing a rerun needs. Everything an experiment varies
lives here: the corpus proportion, the architecture, the optimisation, the
decoding. Nothing is passed on the command line except where the artefacts go.

Four rules are enforced at load time rather than left to a reader of the YAML.

**The seed belongs to the experiment, not to the training block.**
:class:`src.training.TrainingConfig` carries a ``seed`` field because the
trainer needs one. If an experiment file could also set it, a run could declare
``experiment.seed: 7`` and train under 42, and the run record would report the
wrong one. A ``seed`` inside the ``training`` block is refused, and the
experiment seed is injected into the training configuration instead.

**A zero shot experiment cannot carry a corpus proportion.** Section 11 states
that the zero shot baseline does not depend on the training corpus size and
warns against creating three identical zero shot experiments. A percentage on a
zero shot run would file a measurement under a corpus the weights never saw,
which is the same fabrication with one file instead of three.

**A trained experiment must carry both.** A ``training`` block without a
proportion has no corpus to train on, and a proportion without a training block
declares data that nothing reads. Either way the file says something the run
would not do.

**The name is a directory name.** It becomes the run directory under
``reports/results/`` and a row key in every table, so it is restricted to
lowercase, digits and underscores. A name with a slash or a space would either
escape the results directory or come back mangled from a CSV.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.metrics.rouge import ROUGE_VARIANTS, RougeConfig
from src.models.generation import GenerationConfig
from src.models.pretrained.factory import available_baselines
from src.models.scratch.config import ScratchTransformerConfig
from src.training.config import TrainingConfig
from src.utils.seed import DEFAULT_SEED

#: Ablation studies an experiment may belong to. Sections 11 and 12.
STUDIES: tuple[str, ...] = ("dataset_size", "architecture")

#: Variant reported in the ablation tables, derived from the model block.
SCRATCH = "scratch"
PRETRAINED_FINE_TUNED = "pretrained_ft"
PRETRAINED_ZERO_SHOT = "pretrained_zero_shot"

#: Default data pipeline an experiment reads its corpus from.
DEFAULT_DATA_CONFIG = Path("configs") / "data" / "cnn_dailymail.yaml"

#: Directory holding the experiment files, per section 13.
DEFAULT_EXPERIMENTS_DIR = Path("configs") / "experiments"


class ExperimentMeta(BaseModel):
    """Identity of an experiment.

    Attributes:
        name: Short name. Becomes the run directory and the row key of every
            table, hence the restricted character set.
        seed: The single seed of the run, applied by
            :func:`src.utils.seed.set_seed` before anything is built.
        studies: Ablation studies this experiment belongs to. An experiment may
            belong to two: ``scratch_100`` is both the largest point of the
            corpus size ablation and the reference depth of the architecture
            ablation, and running it once rather than twice is what keeps the
            second study to two extra runs.
        description: One line stating what the experiment is for. Carried into
            the run record so a table row can be read without the YAML.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$", description="Run directory and table key.")
    seed: int = Field(default=DEFAULT_SEED, ge=0, description="The single seed of the run.")
    studies: tuple[str, ...] = Field(default=(), description="Ablation studies this run feeds.")
    description: str = Field(default="", description="One line describing the experiment.")

    @model_validator(mode="after")
    def _check_studies(self) -> ExperimentMeta:
        """Reject an unknown or repeated study name.

        Returns:
            The validated metadata.

        Raises:
            ValueError: If a study is unknown or listed twice.
        """
        if len(set(self.studies)) != len(self.studies):
            raise ValueError(f"Duplicate studies: {list(self.studies)}.")

        unknown = [study for study in self.studies if study not in STUDIES]
        if unknown:
            raise ValueError(f"Unknown studies: {unknown}. Available: {', '.join(STUDIES)}.")
        return self


class DatasetSelection(BaseModel):
    """Which corpus the experiment trains on, and how much of it.

    Attributes:
        config: Data pipeline configuration locating the frozen corpus, the
            tokeniser, the prefix and the truncation lengths.
        percentage: Ablation proportion of the training split, in percent.
            ``None`` for an experiment that does not train. The value must be
            one of the proportions the manifest holds, which is checked when
            the corpus is loaded rather than here: only the manifest knows.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    config: Path = Field(default=DEFAULT_DATA_CONFIG, description="Data pipeline configuration.")
    percentage: int | None = Field(default=None, gt=0, le=100, description="Training proportion.")


class ScratchModelConfig(BaseModel):
    """Architecture of the from scratch Transformer.

    The vocabulary size is absent on purpose: it comes from the shared
    tokeniser, and an experiment file that could state a different one would
    let a run build a model whose embedding table does not cover the corpus.

    Attributes:
        type: Discriminator selecting this branch.
        d_model: Width of the residual stream.
        num_heads: Number of attention heads.
        encoder_layers: Depth of the encoder.
        decoder_layers: Depth of the decoder.
        d_ff: Inner width of the feed forward network.
        dropout: Dropout probability after every sublayer.
        max_position: Longest sequence the positional encoding supports. Must
            reach the source truncation length of the corpus, which
            :class:`src.models.scratch.summarizer.ScratchSummarizer` checks.
        tie_embeddings: Whether the output projection reuses the embedding.
        norm_first: Whether to normalise before each sublayer.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["scratch"]
    d_model: int = Field(default=256, gt=0, description="Width of the residual stream.")
    num_heads: int = Field(default=8, gt=0, description="Number of attention heads.")
    encoder_layers: int = Field(default=4, gt=0, description="Depth of the encoder.")
    decoder_layers: int = Field(default=4, gt=0, description="Depth of the decoder.")
    d_ff: int = Field(default=1024, gt=0, description="Inner width of the feed forward network.")
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0, description="Dropout probability.")
    max_position: int = Field(default=512, gt=0, description="Positional encoding budget.")
    tie_embeddings: bool = Field(default=True, description="Reuse the embedding as output head.")
    norm_first: bool = Field(default=True, description="Pre-norm rather than post-norm.")

    def architecture(
        self, *, vocab_size: int, pad_token_id: int, eos_token_id: int
    ) -> ScratchTransformerConfig:
        """Build the model configuration the Transformer is constructed from.

        Args:
            vocab_size: Size of the shared vocabulary, read from the tokeniser.
            pad_token_id: Padding identifier of the shared tokeniser.
            eos_token_id: End of sequence identifier of the shared tokeniser.

        Returns:
            The architecture configuration.

        Raises:
            ValueError: If the hyperparameters are inconsistent, for instance a
                head count that does not divide ``d_model``.
        """
        return ScratchTransformerConfig(
            vocab_size=vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_encoder_layers=self.encoder_layers,
            num_decoder_layers=self.decoder_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            max_position=self.max_position,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            # The T5 tokeniser has no beginning of sequence token, so padding
            # opens the decoder, exactly as T5 itself does.
            decoder_start_token_id=pad_token_id,
            tie_embeddings=self.tie_embeddings,
            norm_first=self.norm_first,
        )


class PretrainedModelConfig(BaseModel):
    """Identification of the pretrained baseline.

    Attributes:
        type: Discriminator selecting this branch.
        baseline: Registered baseline name, resolved by
            :mod:`src.models.pretrained.factory`.
        hf_id: Hugging Face identifier. Defaults to the tokeniser identifier of
            the corpus, which is the right answer when the baseline is the
            model the shared tokeniser came from.
        revision: Commit or tag pinned for the download. Section 14 tolerates
            an unpinned revision only outside a reported experiment.
        mode: ``zero_shot`` for weights that are never updated, ``fine_tuned``
            for a run that trains on the working corpus.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["pretrained"]
    baseline: str = Field(default="t5", description="Registered baseline name.")
    hf_id: str | None = Field(default=None, description="Identifier, defaults to the tokeniser.")
    revision: str | None = Field(default=None, description="Commit or tag pinned for the download.")
    mode: Literal["zero_shot", "fine_tuned"] = Field(
        default="fine_tuned", description="Whether the weights are updated."
    )

    @model_validator(mode="after")
    def _check_baseline(self) -> PretrainedModelConfig:
        """Reject a baseline the factory cannot build.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If the baseline is not registered.
        """
        if self.baseline not in available_baselines():
            raise ValueError(
                f"Unknown baseline {self.baseline!r}. "
                f"Available: {', '.join(available_baselines())}."
            )
        return self


#: The ``model`` block, discriminated on its ``type`` field.
ModelSelection = Annotated[ScratchModelConfig | PretrainedModelConfig, Field(discriminator="type")]


class EvaluationSettings(BaseModel):
    """How the run is measured once the weights are final.

    The decoding budget and the measurement settings sit in the experiment file
    so a rerun needs nothing else, and :mod:`src.experiments.ablation` refuses
    to build a comparison table out of runs whose settings differ. Section 2.1
    compares two models on one test set; a beam width that differed between them
    would move the reported gap for a reason unrelated to either model.

    Attributes:
        split: Split the run is scored on.
        batch_size: Documents encoded at once during generation.
        max_new_tokens: Generated token budget per summary.
        min_new_tokens: Tokens generated before the end of sequence token is
            allowed. Left at zero: forcing a model to keep emitting past the
            point where it wanted to stop raises the score of a model that
            learned nothing to say, and the empty prediction count already
            separates a low score from a broken one.
        num_beams: Beam width. One is greedy search.
        length_penalty: Exponent applied to the length of a finished beam.
        no_repeat_ngram_size: Forbid repeating an n-gram of this size. Zero
            disables the constraint.
        bootstrap_samples: Resamples behind the confidence interval of the mean.
        confidence: Nominal coverage of that interval.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    split: Literal["train", "validation", "test"] = Field(
        default="test", description="Split the run is scored on."
    )
    batch_size: int = Field(default=8, gt=0, description="Documents encoded at once.")
    max_new_tokens: int = Field(default=64, gt=0, description="Generated token budget.")
    min_new_tokens: int = Field(default=0, ge=0, description="Tokens before an end is allowed.")
    num_beams: int = Field(default=4, ge=1, description="Beam width, one is greedy.")
    length_penalty: float = Field(default=1.0, description="Length exponent of a finished beam.")
    no_repeat_ngram_size: int = Field(default=3, ge=0, description="Forbidden repeated n-gram.")
    bootstrap_samples: int = Field(default=1000, ge=0, description="Bootstrap resamples.")
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0, description="Interval coverage.")

    def generation(self) -> GenerationConfig:
        """Build the decoding configuration handed to the summariser.

        Returns:
            The decoding configuration.

        Raises:
            ValueError: If the budgets are inconsistent.
        """
        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
            num_beams=self.num_beams,
            length_penalty=self.length_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
        )

    def rouge(self, seed: int) -> RougeConfig:
        """Build the measurement settings.

        The variants are not configurable: section 2.1 locks them to ROUGE-1,
        ROUGE-2 and ROUGE-L, and a run that measured a subset could not be put
        in the same table as the others.

        Args:
            seed: The experiment seed, which also seeds the resampling so that
                the interval is reproducible.

        Returns:
            The measurement settings.
        """
        return RougeConfig(
            variants=ROUGE_VARIANTS,
            bootstrap_samples=self.bootstrap_samples,
            confidence=self.confidence,
            seed=seed,
        )


class ExperimentConfig(BaseModel):
    """One experiment, fully described.

    Attributes:
        experiment: Identity, seed and studies.
        dataset: Corpus and proportion.
        model: Architecture or baseline.
        training: Optimisation hyperparameters. ``None`` for a zero shot run.
        evaluation: Decoding and measurement settings.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment: ExperimentMeta
    dataset: DatasetSelection = DatasetSelection()
    model: ModelSelection
    training: TrainingConfig | None = None
    evaluation: EvaluationSettings = EvaluationSettings()

    @model_validator(mode="before")
    @classmethod
    def _reject_a_second_seed(cls, data: Any) -> Any:
        """Refuse a seed declared inside the training block.

        Args:
            data: The raw mapping loaded from the YAML file.

        Returns:
            The mapping, unchanged.

        Raises:
            ValueError: If the training block declares a seed of its own.
        """
        if isinstance(data, dict):
            training = data.get("training")
            if isinstance(training, dict) and "seed" in training:
                raise ValueError(
                    "The seed belongs to the experiment block. A seed under training "
                    "would be a second source of truth, and the run record could report "
                    "a value the run did not use."
                )
        return data

    @model_validator(mode="after")
    def _check_coherence(self) -> ExperimentConfig:
        """Reject a file that declares something the run would not do.

        Returns:
            The validated configuration, with the experiment seed pushed into
            the training block.

        Raises:
            ValueError: If a zero shot run carries a proportion or a training
                block, or if a trained run is missing either of them.
        """
        zero_shot = isinstance(self.model, PretrainedModelConfig) and self.model.mode == "zero_shot"

        if zero_shot:
            if self.dataset.percentage is not None:
                raise ValueError(
                    "A zero shot run does not depend on the training corpus size, so it "
                    "cannot declare a proportion. Section 11 asks for one zero shot "
                    "measurement, not one per proportion."
                )
            if self.training is not None:
                raise ValueError("A zero shot run has no training block: its weights never move.")
            return self

        if self.dataset.percentage is None:
            raise ValueError(
                f"Experiment {self.experiment.name!r} trains but declares no corpus "
                "proportion. Set dataset.percentage to one of the proportions of the "
                "data pipeline configuration."
            )
        if self.training is None:
            raise ValueError(
                f"Experiment {self.experiment.name!r} declares a corpus proportion but no "
                "training block, so nothing would read the data it selects."
            )

        if self.training.seed != self.experiment.seed:
            return self.model_copy(
                update={"training": self.training.model_copy(update={"seed": self.experiment.seed})}
            )
        return self

    @property
    def name(self) -> str:
        """Return the experiment name.

        Returns:
            The name, which is also the run directory.
        """
        return self.experiment.name

    @property
    def trains(self) -> bool:
        """Return whether the run updates any weight.

        Returns:
            ``True`` for a from scratch run or a fine tuning, ``False`` for the
            zero shot baseline.
        """
        return self.training is not None

    @property
    def variant(self) -> str:
        """Return the row key used by the ablation tables.

        Returns:
            One of :data:`SCRATCH`, :data:`PRETRAINED_FINE_TUNED` or
            :data:`PRETRAINED_ZERO_SHOT`.
        """
        if isinstance(self.model, ScratchModelConfig):
            return SCRATCH
        return PRETRAINED_FINE_TUNED if self.model.mode == "fine_tuned" else PRETRAINED_ZERO_SHOT

    @property
    def capped(self) -> bool:
        """Return whether the training budget was deliberately cut short.

        A capped run exercises the pipeline; it does not produce a reportable
        score, so it is written as a partial run.

        Returns:
            ``True`` when the training block sets ``max_steps``.
        """
        return self.training is not None and self.training.max_steps is not None

    def to_dict(self) -> dict[str, Any]:
        """Render the configuration as a JSON serialisable mapping.

        Returns:
            The mapping stored in the run record, so that a table row can be
            traced back to what produced it without reopening the YAML file.
        """
        return {
            "experiment": {
                "name": self.experiment.name,
                "seed": self.experiment.seed,
                "studies": list(self.experiment.studies),
                "description": self.experiment.description,
            },
            "dataset": {
                "config": str(self.dataset.config),
                "percentage": self.dataset.percentage,
            },
            "model": self.model.model_dump(mode="json"),
            "training": self.training.model_dump(mode="json") if self.training else None,
            "evaluation": self.evaluation.model_dump(mode="json"),
        }


def load_experiment_config(path: Path | str) -> ExperimentConfig:
    """Load and validate one experiment file.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not hold a YAML mapping, or if the
            configuration is inconsistent.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment configuration not found: {config_path}")

    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping.")

    return ExperimentConfig.model_validate(raw)


def iter_experiment_files(directory: Path | str = DEFAULT_EXPERIMENTS_DIR) -> Iterator[Path]:
    """Walk the experiment files of a directory, in name order.

    Args:
        directory: Directory holding the experiment files.

    Yields:
        One path per ``.yaml`` file, sorted so that a listing is stable across
        machines and a table keeps the same row order.
    """
    yield from sorted(Path(directory).glob("*.yaml"))


def discover_experiments(
    directory: Path | str = DEFAULT_EXPERIMENTS_DIR,
) -> list[ExperimentConfig]:
    """Load every experiment declared under a directory.

    The declared set is what the tables are built from: an experiment that has
    a file but no result is reported as ``NOT_RUN`` rather than omitted, so a
    missing measurement stays visible instead of leaving a shorter table that
    looks complete.

    Args:
        directory: Directory holding the experiment files.

    Returns:
        The configurations, sorted by name.

    Raises:
        ValueError: If two files declare the same experiment name, or if a file
            is invalid. Both are refused here rather than at run time: a
            duplicate name means two runs writing to one directory, and the
            second would silently overwrite the first.
    """
    configs: dict[str, ExperimentConfig] = {}
    for path in iter_experiment_files(directory):
        config = load_experiment_config(path)
        if config.name in configs:
            raise ValueError(f"Two experiment files declare the name {config.name!r}.")
        configs[config.name] = config

    return [configs[name] for name in sorted(configs)]
