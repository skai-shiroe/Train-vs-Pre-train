"""Putting the measured weights into the tracking store.

Until the corpus moved to CNN/DailyMail this project deliberately kept the
weights out of MLflow: the checkpoints lived under ``runs/`` and the store held
four small JSON files. That decision is reversed here, on request, and the
reversal is worth stating rather than hiding, because the old rationale was
sound and what changed is the requirement, not the argument.

**What the store now holds.** One logged model per measured run, in the flavour
that matches it: ``mlflow.transformers`` for ``t5-small``, which round trips the
tokeniser and the generation configuration with the weights, and
``mlflow.pytorch`` for the hand written Transformer, which has no flavour of its
own. Each is registered under ``syntra-<experiment>`` so it can be fetched by
name instead of by run identifier.

**PostgreSQL never holds a weight.** The database stores the metadata of a run
and a pointer; the artefacts, model directories included, are written under
:data:`src.tracking.store.ARTIFACT_ROOT_VARIABLE`. A 240 MB tensor in a
relational column would be a bad database and a bad model store at once.

**Only a complete run is registered.** A ``PARTIAL`` run trained for two steps
and was scored on eight documents. Its weights exist and are meaningless, and a
registry entry is exactly the kind of artefact somebody later loads without
reading the status beside it. :func:`should_log` is the single place that says
so.

**The from scratch model travels with the code that defines it.**
``mlflow.pytorch`` serialises the object, so loading it needs
``src.models.scratch`` importable. ``code_paths`` copies ``src`` next to the
weights, which makes the logged model loadable from a checkout that does not
have this repository installed.

**Pickled, not traced.** MLflow 3 defaults the PyTorch flavour to ``pt2``, which
records a traced graph and therefore demands an example input to virtually
execute ``forward`` with. The forward of this model takes a source, a padding
mask and a shifted target, and its generation loop is Python rather than graph:
a trace would either refuse the call or freeze one shape of it. The pickle
format stores the object as it is, and ``code_paths`` is what makes it loadable
again.

**The dependencies of a logged model are declared, not inferred.** Left to
itself, ``mlflow.transformers`` pins every package its default list names, and
that list carries ``torchvision`` whatever the task: it imports it to read its
version, and a text only project that never installed it gets a model logging
failure instead of a model. Declaring the list here keeps a computer vision
dependency out of the project and fixes a second defect at the same time, since
the inferred list also drops the local version label of Torch and pins
``torch==2.13.0`` where the environment runs ``2.13.0+cu130``.

The pin published is the public version, without the local label, because that
is the only form a plain ``pip install`` resolves. Reproducing the GPU
environment additionally needs the CUDA index the ``Makefile`` names; a wheel
tag in a requirements file would not have said that either.

**A failure here never fails a run.** Same rule as the rest of
:mod:`src.tracking`: the result of an experiment is the record on disk. A store
that refuses a 240 MB upload must not destroy the measurement that preceded it,
so the caller wraps this in :func:`log_model_safely`.
"""

from __future__ import annotations

import sys
from typing import Any, Protocol

#: Prefix of every name in the model registry. Without it the registry of a
#: shared database mixes this project's models with everybody else's.
REGISTRY_PREFIX = "syntra"

#: Artefact sub directory the model is written under, inside the run.
ARTIFACT_NAME = "model"

#: Sentence handed to MLflow as the input example of the pretrained flavour. It
#: carries the task prefix because the model was measured with it, and a
#: signature inferred without it would describe an input the run never used.
INPUT_EXAMPLE = "summarize: The council approved the plan on Tuesday evening."

#: Packages a logged model needs to be loaded again. Declared rather than left
#: to MLflow's inference, for the reason the module docstring gives.
REQUIRED_PACKAGES: tuple[str, ...] = ("torch", "transformers", "sentencepiece")


class Summarizer(Protocol):
    """The part of a summariser this module needs.

    Declared structurally rather than imported: :mod:`src.tracking` must not
    depend on :mod:`src.models`, and the two attributes below are all that
    logging a model requires.
    """

    @property
    def model(self) -> Any:
        """Return the underlying network."""

    @property
    def tokenizer(self) -> Any:
        """Return the tokeniser the network was measured with."""


def registered_name(experiment: str) -> str:
    """Return the registry name of one experiment's model.

    Args:
        experiment: Name of the experiment, as it appears in the run record.

    Returns:
        The name the model is registered under.
    """
    return f"{REGISTRY_PREFIX}-{experiment}"


def should_log(status: str, *, measured_statuses: tuple[str, ...] = ("OK",)) -> bool:
    """Return whether a run's weights belong in the store.

    Args:
        status: Status the record carries.
        measured_statuses: Statuses that count as a complete measurement.

    Returns:
        ``True`` only for a complete run. A partial or failed run keeps its
        record and its checkpoint on disk and stays out of the registry.
    """
    return status in measured_statuses


def pip_requirements() -> list[str]:
    """Return the pins written beside a logged model.

    The version comes from the installed distribution rather than from a
    constant, so the file describes the environment that produced the model
    instead of the one somebody expected. The local version label is dropped:
    ``torch==2.13.0+cu130`` resolves from the CUDA index and nowhere else, and a
    requirements file that no ``pip install`` can satisfy helps no one.

    Returns:
        One ``package==version`` string per entry of :data:`REQUIRED_PACKAGES`,
        skipping any that is not installed.
    """
    from importlib.metadata import PackageNotFoundError, version

    pins: list[str] = []
    for package in REQUIRED_PACKAGES:
        try:
            pins.append(f"{package}=={version(package).split('+')[0]}")
        except PackageNotFoundError:
            continue
    return pins


def _is_transformers_model(model: Any) -> bool:
    """Return whether a model belongs to the Hugging Face flavour.

    Args:
        model: The network to classify.

    Returns:
        ``True`` when ``transformers`` is importable and the model is one of
        its pretrained models. A missing ``transformers`` is not an error here:
        it only means the from scratch flavour is the right one.
    """
    try:
        from transformers import PreTrainedModel
    except ImportError:  # pragma: no cover - transformers is a hard dependency
        return False
    return isinstance(model, PreTrainedModel)


def log_model(summarizer: Summarizer, *, experiment: str, register: bool = True) -> str | None:
    """Log the weights of the open run and register them.

    The run must already be open: this is called from the runner between the
    evaluation and the closing of the record, so the model lands in the same
    MLflow run as the score it produced.

    Args:
        summarizer: The evaluated summariser, holding the weights the score was
            measured on. Not the training object: when early stopping selected
            an earlier epoch, this one was rebuilt from that checkpoint, and
            logging anything else would store weights nobody measured.
        experiment: Name of the experiment, used for the registry name.
        register: Whether to create a registry entry. ``False`` logs the model
            into the run without registering it, which is what a store without
            a registry backend needs.

    Returns:
        The name the model was registered under, or ``None`` when it was logged
        without registration.
    """
    import mlflow

    name = registered_name(experiment) if register else None

    if _is_transformers_model(summarizer.model):
        import mlflow.transformers

        mlflow.transformers.log_model(
            transformers_model={"model": summarizer.model, "tokenizer": summarizer.tokenizer},
            name=ARTIFACT_NAME,
            task="summarization",
            registered_model_name=name,
            input_example=INPUT_EXAMPLE,
            pip_requirements=pip_requirements(),
        )
        return name

    import mlflow.pytorch

    mlflow.pytorch.log_model(
        pytorch_model=summarizer.model,
        name=ARTIFACT_NAME,
        registered_model_name=name,
        code_paths=["src"],
        serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_PICKLE,
        pip_requirements=pip_requirements(),
    )
    return name


def log_model_safely(
    summarizer: Summarizer, *, experiment: str, register: bool = True
) -> str | None:
    """Log a model, reporting a failure instead of raising it.

    Args:
        summarizer: The evaluated summariser.
        experiment: Name of the experiment.
        register: Whether to create a registry entry.

    Returns:
        What :func:`log_model` returned, or ``None`` when it failed.
    """
    try:
        return log_model(summarizer, experiment=experiment, register=register)
    except Exception as error:  # noqa: BLE001 - a store failure must not fail a run
        print(f"Model tracking failed for {experiment!r}: {error}", file=sys.stderr)
        return None
