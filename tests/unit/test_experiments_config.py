"""Unit tests of the experiment configuration.

Most of these check a refusal. An experiment file is the only thing a rerun
reads, so a file that says something the run would not do is the defect this
layer exists to catch: the run would still succeed, and the record would report
the file rather than the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.experiments.config import (
    DEFAULT_EXPERIMENTS_DIR,
    PRETRAINED_FINE_TUNED,
    PRETRAINED_ZERO_SHOT,
    SCRATCH,
    STUDIES,
    EvaluationSettings,
    ExperimentConfig,
    PretrainedModelConfig,
    ScratchModelConfig,
    campaign_order,
    discover_experiments,
    iter_experiment_files,
    load_experiment_config,
)
from src.metrics.rouge import ROUGE_VARIANTS

pytestmark = pytest.mark.unit


def scratch_payload(**overrides: Any) -> dict[str, Any]:
    """Return a valid from scratch experiment, with optional overrides."""
    payload: dict[str, Any] = {
        "experiment": {"name": "scratch_test", "seed": 7, "studies": ["dataset_size"]},
        "dataset": {"config": "configs/data/cnn_dailymail.yaml", "percentage": 10},
        "model": {"type": "scratch", "d_model": 32, "num_heads": 2},
        "training": {"epochs": 1, "batch_size": 2},
    }
    payload.update(overrides)
    return payload


def zero_shot_payload(**overrides: Any) -> dict[str, Any]:
    """Return a valid zero shot experiment, with optional overrides."""
    payload: dict[str, Any] = {
        "experiment": {"name": "zero_shot_test"},
        "dataset": {"config": "configs/data/cnn_dailymail.yaml"},
        "model": {"type": "pretrained", "baseline": "t5", "mode": "zero_shot"},
    }
    payload.update(overrides)
    return payload


def write_config(path: Path, payload: dict[str, Any]) -> Path:
    """Write a payload as YAML and return the path."""
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The single seed
# ---------------------------------------------------------------------------


def test_the_experiment_seed_reaches_the_training_block() -> None:
    config = ExperimentConfig.model_validate(scratch_payload())

    assert config.experiment.seed == 7
    assert config.training is not None
    assert config.training.seed == 7


def test_a_seed_declared_under_training_is_refused() -> None:
    payload = scratch_payload(training={"epochs": 1, "batch_size": 2, "seed": 7})

    with pytest.raises(ValueError, match="belongs to the experiment block"):
        ExperimentConfig.model_validate(payload)


def test_the_seed_is_refused_even_when_it_agrees() -> None:
    # Agreeing today is not the point: the second declaration is what lets the
    # two drift apart tomorrow without anything failing.
    payload = scratch_payload(
        experiment={"name": "scratch_test", "seed": 42},
        training={"epochs": 1, "batch_size": 2, "seed": 42},
    )

    with pytest.raises(ValueError, match="second source of truth"):
        ExperimentConfig.model_validate(payload)


# ---------------------------------------------------------------------------
# Zero shot has no corpus size
# ---------------------------------------------------------------------------


def test_a_zero_shot_run_cannot_declare_a_proportion() -> None:
    payload = zero_shot_payload(
        dataset={"config": "configs/data/cnn_dailymail.yaml", "percentage": 10}
    )

    with pytest.raises(ValueError, match="does not depend on the training corpus size"):
        ExperimentConfig.model_validate(payload)


def test_a_zero_shot_run_cannot_declare_a_training_block() -> None:
    payload = zero_shot_payload(training={"epochs": 1})

    with pytest.raises(ValueError, match="weights never move"):
        ExperimentConfig.model_validate(payload)


def test_a_zero_shot_run_is_valid_without_either() -> None:
    config = ExperimentConfig.model_validate(zero_shot_payload())

    assert config.trains is False
    assert config.variant == PRETRAINED_ZERO_SHOT
    assert config.dataset.percentage is None


# ---------------------------------------------------------------------------
# A trained run needs both
# ---------------------------------------------------------------------------


def test_a_trained_run_without_a_proportion_is_refused() -> None:
    payload = scratch_payload(dataset={"config": "configs/data/cnn_dailymail.yaml"})

    with pytest.raises(ValueError, match="declares no corpus"):
        ExperimentConfig.model_validate(payload)


def test_a_trained_run_without_a_training_block_is_refused() -> None:
    payload = scratch_payload()
    del payload["training"]

    with pytest.raises(ValueError, match="no\ntraining block|no training block"):
        ExperimentConfig.model_validate(payload)


def test_a_fine_tuned_baseline_needs_a_proportion() -> None:
    payload = {
        "experiment": {"name": "ft_test"},
        "model": {"type": "pretrained", "mode": "fine_tuned"},
        "training": {"epochs": 1},
    }

    with pytest.raises(ValueError, match="declares no corpus"):
        ExperimentConfig.model_validate(payload)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Scratch10", "scratch-10", "scratch 10", "10_scratch", ""])
def test_a_name_that_is_not_a_directory_name_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="experiment"):
        ExperimentConfig.model_validate(scratch_payload(experiment={"name": name, "seed": 7}))


def test_an_unknown_study_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown studies"):
        ExperimentConfig.model_validate(
            scratch_payload(experiment={"name": "scratch_test", "studies": ["speed"]})
        )


def test_a_repeated_study_is_refused() -> None:
    with pytest.raises(ValueError, match="Duplicate studies"):
        ExperimentConfig.model_validate(
            scratch_payload(
                experiment={"name": "scratch_test", "studies": ["dataset_size", "dataset_size"]}
            )
        )


def test_an_unknown_baseline_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown baseline"):
        ExperimentConfig.model_validate(
            zero_shot_payload(
                model={"type": "pretrained", "baseline": "mbart", "mode": "zero_shot"}
            )
        )


def test_an_unknown_field_is_refused() -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        ExperimentConfig.model_validate(scratch_payload(mlflow={"tracking_uri": "http://x"}))


def test_something_that_is_not_a_mapping_is_refused() -> None:
    with pytest.raises(ValueError, match="valid dictionary"):
        ExperimentConfig.model_validate(["experiment", "dataset"])


# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------


def test_the_variant_follows_the_model_block() -> None:
    scratch = ExperimentConfig.model_validate(scratch_payload())
    fine_tuned = ExperimentConfig.model_validate(
        scratch_payload(model={"type": "pretrained", "mode": "fine_tuned"})
    )

    assert scratch.variant == SCRATCH
    assert fine_tuned.variant == PRETRAINED_FINE_TUNED


def test_a_capped_training_budget_marks_the_run() -> None:
    uncapped = ExperimentConfig.model_validate(scratch_payload())
    capped = ExperimentConfig.model_validate(
        scratch_payload(training={"epochs": 1, "batch_size": 2, "max_steps": 3})
    )

    assert uncapped.capped is False
    assert capped.capped is True


def test_the_decoder_starts_on_the_padding_token() -> None:
    # The T5 tokeniser has no beginning of sequence token. A decoder started on
    # anything else would be fed a token the corpus never contains.
    model = ScratchModelConfig(type="scratch", d_model=32, num_heads=2)

    architecture = model.architecture(vocab_size=100, pad_token_id=0, eos_token_id=1)

    assert architecture.decoder_start_token_id == 0
    assert architecture.pad_token_id == 0
    assert architecture.eos_token_id == 1
    assert architecture.vocab_size == 100


def test_the_measured_variants_are_not_configurable() -> None:
    # Section 2.1 locks them. A run that measured a subset could not be put in
    # the same table as the others.
    settings = EvaluationSettings()

    assert settings.rouge(seed=7).variants == ROUGE_VARIANTS
    assert settings.rouge(seed=7).seed == 7


def test_the_decoding_block_becomes_a_generation_config() -> None:
    settings = EvaluationSettings(max_new_tokens=32, num_beams=2, no_repeat_ngram_size=3)

    generation = settings.generation()

    assert generation.max_new_tokens == 32
    assert generation.num_beams == 2
    assert generation.no_repeat_ngram_size == 3


def test_the_record_view_carries_the_whole_file() -> None:
    config = ExperimentConfig.model_validate(scratch_payload())

    payload = config.to_dict()

    assert payload["experiment"]["seed"] == 7
    assert payload["dataset"]["percentage"] == 10
    assert payload["model"]["type"] == "scratch"
    assert payload["training"]["seed"] == 7
    assert payload["evaluation"]["split"] == "test"


def test_a_zero_shot_record_view_has_no_training() -> None:
    config = ExperimentConfig.model_validate(zero_shot_payload())

    assert config.to_dict()["training"] is None


# ---------------------------------------------------------------------------
# Loading and discovery
# ---------------------------------------------------------------------------


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Experiment configuration not found"):
        load_experiment_config(tmp_path / "absent.yaml")


def test_a_file_that_is_not_a_mapping_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        load_experiment_config(path)


def test_discovery_is_sorted_and_complete(tmp_path: Path) -> None:
    write_config(tmp_path / "b.yaml", scratch_payload(experiment={"name": "b_run", "seed": 1}))
    write_config(tmp_path / "a.yaml", zero_shot_payload(experiment={"name": "a_run"}))

    names = [config.name for config in discover_experiments(tmp_path)]

    assert names == ["a_run", "b_run"]


def test_two_files_declaring_one_name_are_refused(tmp_path: Path) -> None:
    # Two files with one name means two runs writing to one directory, and the
    # second silently overwriting the first.
    write_config(tmp_path / "one.yaml", scratch_payload())
    write_config(tmp_path / "two.yaml", scratch_payload(dataset={"percentage": 50}))

    with pytest.raises(ValueError, match="declare the name"):
        discover_experiments(tmp_path)


def test_an_empty_directory_declares_nothing(tmp_path: Path) -> None:
    assert discover_experiments(tmp_path) == []
    assert list(iter_experiment_files(tmp_path)) == []


# ---------------------------------------------------------------------------
# The order a campaign runs in
# ---------------------------------------------------------------------------


def test_a_campaign_trains_from_scratch_before_touching_the_baseline() -> None:
    scratch = ExperimentConfig.model_validate(scratch_payload())
    zero_shot = ExperimentConfig.model_validate(zero_shot_payload())
    fine_tuned = ExperimentConfig.model_validate(
        {
            "experiment": {"name": "pretrained_ft_10"},
            "dataset": {"percentage": 10},
            "model": {"type": "pretrained", "mode": "fine_tuned"},
            "training": {"epochs": 1, "batch_size": 2},
        }
    )

    ordered = campaign_order([fine_tuned, zero_shot, scratch])

    assert [config.variant for config in ordered] == [
        SCRATCH,
        PRETRAINED_ZERO_SHOT,
        PRETRAINED_FINE_TUNED,
    ]


def test_a_family_is_ordered_by_growing_corpus_proportion() -> None:
    configs = [
        ExperimentConfig.model_validate(
            scratch_payload(
                experiment={"name": f"scratch_{percentage}", "studies": ["dataset_size"]},
                dataset={"percentage": percentage},
            )
        )
        for percentage in (100, 10, 50)
    ]

    assert [config.dataset.percentage for config in campaign_order(configs)] == [10, 50, 100]


def test_the_campaign_order_does_not_depend_on_the_order_the_files_were_read() -> None:
    # The order has to be a property of the experiments, not of the directory
    # listing: a campaign resumed on another machine runs the same sweep.
    declared = discover_experiments(DEFAULT_EXPERIMENTS_DIR)

    assert campaign_order(reversed(declared)) == campaign_order(declared)


# ---------------------------------------------------------------------------
# The experiments the repository actually declares
# ---------------------------------------------------------------------------


def test_every_declared_experiment_validates() -> None:
    configs = discover_experiments(DEFAULT_EXPERIMENTS_DIR)

    assert {config.name for config in configs} == {
        "scratch_10",
        "scratch_50",
        "scratch_100",
        "scratch_100_layers2",
        "scratch_100_layers6",
        "pretrained_zero_shot",
        "pretrained_ft_10",
        "pretrained_ft_50",
        "pretrained_ft_100",
    }


def test_the_declared_campaign_opens_on_the_from_scratch_family() -> None:
    ordered = campaign_order(discover_experiments(DEFAULT_EXPERIMENTS_DIR))

    assert [config.name for config in ordered] == [
        "scratch_10",
        "scratch_50",
        "scratch_100",
        "scratch_100_layers2",
        "scratch_100_layers6",
        "pretrained_zero_shot",
        "pretrained_ft_10",
        "pretrained_ft_50",
        "pretrained_ft_100",
    ]


def test_the_corpus_size_ablation_has_one_zero_shot_point() -> None:
    # Section 11 forbids three identical zero shot experiments.
    configs = discover_experiments(DEFAULT_EXPERIMENTS_DIR)
    zero_shot = [config for config in configs if config.variant == PRETRAINED_ZERO_SHOT]

    assert len(zero_shot) == 1
    assert zero_shot[0].dataset.percentage is None


def test_the_corpus_size_ablation_covers_the_three_proportions() -> None:
    configs = discover_experiments(DEFAULT_EXPERIMENTS_DIR)

    for variant in (SCRATCH, PRETRAINED_FINE_TUNED):
        proportions = sorted(
            config.dataset.percentage
            for config in configs
            if config.variant == variant and "dataset_size" in config.experiment.studies
        )
        assert proportions == [10, 50, 100]


def test_the_architecture_ablation_varies_one_quantity() -> None:
    # Section 12 compares one dimension. Anything else moving with it would
    # make the gap unattributable.
    configs = [
        config
        for config in discover_experiments(DEFAULT_EXPERIMENTS_DIR)
        if "architecture" in config.experiment.studies
    ]
    models = [config.model for config in configs]

    assert len(models) == 3
    assert all(isinstance(model, ScratchModelConfig) for model in models)
    scratch_models = [model for model in models if isinstance(model, ScratchModelConfig)]
    assert {model.encoder_layers for model in scratch_models} == {2, 4, 6}
    assert {model.d_model for model in scratch_models} == {256}
    assert {model.num_heads for model in scratch_models} == {8}
    assert {model.d_ff for model in scratch_models} == {1024}


def test_every_declared_experiment_is_measured_the_same_way() -> None:
    # The comparability check of the aggregation can only pass if the files
    # agree in the first place.
    configs = discover_experiments(DEFAULT_EXPERIMENTS_DIR)

    assert {config.evaluation for config in configs} == {configs[0].evaluation}


def test_every_declared_experiment_belongs_to_a_known_study() -> None:
    for config in discover_experiments(DEFAULT_EXPERIMENTS_DIR):
        assert config.experiment.studies, f"{config.name} feeds no study"
        assert set(config.experiment.studies) <= set(STUDIES)


def test_the_positional_encoding_covers_the_corpus_truncation() -> None:
    # A model whose positional encoding stops before the source truncation
    # would refuse to build at run time, three hours into the sweep.
    for config in discover_experiments(DEFAULT_EXPERIMENTS_DIR):
        if isinstance(config.model, ScratchModelConfig):
            assert config.model.max_position >= 512


def test_the_pretrained_experiments_name_a_registered_baseline() -> None:
    for config in discover_experiments(DEFAULT_EXPERIMENTS_DIR):
        if isinstance(config.model, PretrainedModelConfig):
            assert config.model.baseline == "t5"
