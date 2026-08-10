"""Running a model on a request, under a deadline.

Sections 23 and 24 ask for two endpoints. Both do the same three things: resolve
a model, generate, measure. The difference is how many models answer and whether
a reference is scored, so both live here and the routers stay thin.

**Generation runs in a worker thread.** A torch forward pass releases nothing to
the event loop, so a single request would otherwise block every other one,
including ``/health``. The thread is abandoned when the deadline fires rather
than joined: a beam search that overran its budget keeps running to its end on
its own thread, and the client stops waiting for it. Joining it would make the
timeout a report rather than a limit.

**The deadline covers generation, not loading.** ``SYNTRA_INFERENCE_TIMEOUT_S``
is the budget of one generation. Pulling several hundred megabytes into memory
the first time a version is asked for is a different event, and cutting it at
thirty seconds would turn a cold start into a permanent 504.

**The token budget is capped by the weights, and a comparison by the smallest
of them.** A model was trained to produce summaries of a given length, and asked
for more it generates outside the distribution it saw; a from scratch one runs
past its positional encoding and raises. The request is therefore capped rather
than refused, and the effective value travels back with the prediction so the
response reports the decoding that ran instead of the one that was asked for.
For a comparison the cap is the smallest budget of the models involved: section
2.1 compares under identical conditions, and two different budgets would be two
different conditions.

**A model failure is a 500, an overrun is a 504, and neither carries a trace.**
What the client receives names the model and the deadline; what the exception
said goes to the log.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

import anyio
import anyio.to_thread

from backend.app.core.errors import InferenceFailedError, InferenceTimeoutError, SyntraError
from backend.app.inference.base import LoadedModel
from backend.app.registry.models import ModelVersion
from backend.app.services.store import ModelStore
from src.metrics.rouge import RougeScore, score_example
from src.models.generation import GenerationConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Prediction:
    """One summary, and what produced it.

    Attributes:
        version: The version that answered, so a summary can be traced back to
            the weights and to the run that measured them.
        summary: The generated text. May be empty: a model that produces
            nothing is a measurement, not an error.
        latency_ms: Wall clock time of the generation alone, in milliseconds.
        decoding: The configuration the generation actually ran under, once
            capped to what the weights can produce.
        rouge: Scores against the reference the caller supplied, when it
            supplied one.
    """

    version: ModelVersion
    summary: str
    latency_ms: float
    decoding: GenerationConfig
    rouge: dict[str, RougeScore] | None = None


@dataclass(frozen=True, slots=True)
class Comparison:
    """Several models on one document.

    Attributes:
        predictions: One prediction per model, in the order asked for.
        decoding: The configuration every model ran under. One block, not one
            per model: a comparison under two configurations is not one.
    """

    predictions: list[Prediction]
    decoding: GenerationConfig


def cap_budget(config: GenerationConfig, budget: int) -> GenerationConfig:
    """Bring a decoding configuration within what a model can produce.

    Args:
        config: What the request asked for.
        budget: The longest summary the weights were trained to produce.

    Returns:
        The configuration itself when it already fits, a copy carrying the
        smaller budget otherwise.
    """
    if config.max_new_tokens <= budget:
        return config
    return replace(config, max_new_tokens=budget)


async def run_blocking[T](func: Callable[[], T]) -> T:
    """Run a blocking call on a worker thread.

    Args:
        func: The call, already bound to its arguments.

    Returns:
        Whatever the call returned.
    """
    return await anyio.to_thread.run_sync(func, abandon_on_cancel=True)


async def resolve_model(
    store: ModelStore, name: str | None = None, version: str | None = None
) -> LoadedModel:
    """Return the model a request named, loading it if it is not in memory.

    Args:
        store: The store of the running instance.
        name: A registered name or a deployment alias.
        version: An exact version.

    Returns:
        The loaded model.

    Raises:
        ModelNotFoundError: If nothing is registered under that name.
        ModelNotReadyError: If the version cannot be loaded.
    """
    return await run_blocking(partial(store.get, name, version))


async def generate(
    loaded: LoadedModel, text: str, config: GenerationConfig, timeout_s: float
) -> tuple[str, float]:
    """Generate one summary under a deadline, and time it.

    Args:
        loaded: The model that answers.
        text: The document to summarise.
        config: Decoding configuration.
        timeout_s: Maximum duration of the generation, in seconds.

    Returns:
        A pair ``(summary, latency_ms)``.

    Raises:
        InferenceTimeoutError: If the generation exceeded the deadline.
        InferenceFailedError: If the model raised.
    """
    started = time.perf_counter()
    try:
        with anyio.fail_after(timeout_s):
            summary = await run_blocking(partial(loaded.summarize, text, config))
    except TimeoutError as error:
        LOGGER.warning("generation on %s exceeded %.1fs", loaded.label, timeout_s)
        raise InferenceTimeoutError(
            details={"model": loaded.version.name, "timeout_s": timeout_s}
        ) from error
    except SyntraError:
        raise
    except Exception as error:
        # Anything torch or transformers raises stops here. The client is told
        # which model failed; the exception itself is logged and goes no
        # further, per the exposure rules of section 26.
        LOGGER.exception("generation on %s raised", loaded.label)
        raise InferenceFailedError(details={"model": loaded.version.name}) from error

    return summary, (time.perf_counter() - started) * 1000.0


async def summarize(
    store: ModelStore,
    text: str,
    *,
    name: str | None = None,
    version: str | None = None,
    config: GenerationConfig,
    timeout_s: float,
) -> Prediction:
    """Summarise one document with one model.

    Args:
        store: The store of the running instance.
        text: The document to summarise.
        name: A registered name or a deployment alias.
        version: An exact version.
        config: Decoding configuration.
        timeout_s: Maximum duration of the generation, in seconds.

    Returns:
        The prediction, without any score: a single request carries no
        reference to measure against.

    Raises:
        ModelNotFoundError: If nothing is registered under that name.
        ModelNotReadyError: If the version cannot be loaded.
        InferenceTimeoutError: If the generation exceeded the deadline.
        InferenceFailedError: If the model raised.
    """
    loaded = await resolve_model(store, name, version)
    decoding = cap_budget(config, loaded.token_budget)
    summary, latency_ms = await generate(loaded, text, decoding, timeout_s)

    LOGGER.info(
        "prediction served",
        extra={
            "model": loaded.version.name,
            "version": loaded.version.version,
            # The document itself is never logged. Its length is what a
            # latency is read against, and it is not user content.
            "input_chars": len(text),
            "summary_chars": len(summary),
            "latency_ms": round(latency_ms, 1),
        },
    )
    return Prediction(
        version=loaded.version, summary=summary, latency_ms=latency_ms, decoding=decoding
    )


async def compare(
    store: ModelStore,
    text: str,
    *,
    names: list[str],
    reference: str | None = None,
    config: GenerationConfig,
    timeout_s: float,
) -> Comparison:
    """Summarise one document with several models, and score them if asked.

    The models answer one after the other rather than concurrently. They share
    one device, so overlapping two generations would lengthen both and make the
    reported latencies meaningless as a comparison, which is the whole point of
    this endpoint.

    Args:
        store: The store of the running instance.
        text: The document to summarise.
        names: The names or aliases to compare, in the order they are reported.
        reference: A reference summary. When present, every prediction is
            scored against it with the ROUGE variants of section 2.1.
        config: Decoding configuration, shared by every model. A beam width
            that differed between two models would move the gap for a reason
            unrelated to either of them.
        timeout_s: Maximum duration of each generation, in seconds.

    Returns:
        The comparison: one prediction per name, in the order asked for, and
        the single decoding they all ran under.

    Raises:
        ModelNotFoundError: If one of the names is not registered.
        ModelNotReadyError: If one of the versions cannot be loaded.
        InferenceTimeoutError: If one generation exceeded the deadline.
        InferenceFailedError: If one model raised.
    """
    # Every model is resolved before the first one generates: the shared budget
    # cannot be known until they are all known, and running the first under a
    # budget the second cannot honour would compare two configurations.
    models = [await resolve_model(store, name) for name in names]
    decoding = cap_budget(config, min(loaded.token_budget for loaded in models))

    predictions: list[Prediction] = []
    for loaded in models:
        summary, latency_ms = await generate(loaded, text, decoding, timeout_s)
        predictions.append(
            Prediction(
                version=loaded.version,
                summary=summary,
                latency_ms=latency_ms,
                decoding=decoding,
                rouge=score_example(summary, reference) if reference else None,
            )
        )

    LOGGER.info(
        "comparison served",
        extra={
            "models": [prediction.version.label for prediction in predictions],
            "input_chars": len(text),
            "scored": reference is not None,
        },
    )
    return Comparison(predictions=predictions, decoding=decoding)
