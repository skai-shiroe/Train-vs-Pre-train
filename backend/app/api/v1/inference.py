"""The two endpoints that run a model.

Sections 23 and 24. Both translate a request into a decoding configuration, hand
it to :mod:`backend.app.services.summarization` and render what comes back. The
work is in the service; what lives here is the contract.

**One decoding configuration, built in one place.** Everything the request does
not carry is fixed to the settings the evaluation of section 10 used, so a
request with ``num_beams`` set to four and ``do_sample`` to false reproduces the
configuration the published ROUGE was measured under. Had the API decoded with a
different n-gram constraint, that claim would be false and nothing would have
said so.

**A sampled request without a seed is given one.** Drawing it here rather than
letting torch run on its ambient state is what keeps the promise of the
``decoding`` block: every response describes a run that can be replayed from the
response alone. The seed is drawn once per request, so the models of a
comparison share it, as they share every other decoding parameter.
"""

from __future__ import annotations

import secrets
from typing import Annotated, Any

from fastapi import APIRouter, Body

from backend.app.api.deps import SettingsDep, StoreDep
from backend.app.core.config import Settings
from backend.app.core.errors import ErrorCode, ModelNotFoundError
from backend.app.schemas.common import error_responses
from backend.app.schemas.inference import (
    MAX_SEED,
    CompareRequest,
    CompareResponse,
    CompareResult,
    Decoding,
    GenerationOptions,
    PredictRequest,
    PredictResponse,
    RougeVariantScore,
)
from backend.app.services import summarization
from src.models.generation import GenerationConfig

#: Repeated n-gram the decoding forbids. Three is what the evaluation settings
#: of section 13 default to, and matching it is what makes a request with four
#: beams the very configuration the published score was measured under.
NO_REPEAT_NGRAM_SIZE = 3

router = APIRouter(tags=["inference"])

#: Errors both endpoints may answer with. Listed once so the two operations
#: cannot document different halves of the same behaviour.
INFERENCE_ERRORS = error_responses(
    ErrorCode.INVALID_INPUT,
    ErrorCode.MODEL_NOT_FOUND,
    ErrorCode.MODEL_NOT_READY,
    ErrorCode.INFERENCE_FAILED,
    ErrorCode.TIMEOUT,
)

#: The document every example carries. English, because the corpus of section
#: 2.1 is XSum: a French document would sit outside the distribution every
#: published score was measured on, and the summary it produced would mislead
#: whoever pressed Send first. Short enough to stay readable in a request body,
#: long enough for a summary to be a reduction rather than a copy.
EXAMPLE_DOCUMENT = (
    "The city council approved a new plan to expand the tram network across the "
    "northern districts. Construction will begin next spring and is expected to "
    "take three years. Officials said the extension will serve about forty "
    "thousand additional residents and reduce car traffic on the main avenue."
)

#: The reference the scored example of ``/compare`` is measured against.
EXAMPLE_REFERENCE = "The council approved a tram extension serving forty thousand more residents."

#: Bodies offered for ``/predict``. A tool that builds a request from the schema
#: alone fills ``text`` with the word ``string``, which is a valid payload and a
#: meaningless one: it returns a summary of nothing and teaches nothing about
#: the API. Every example here is a request that can be sent as it stands
#: against the registry this project publishes.
#:
#: **The keys are named so that their alphabetical order is the reading order.**
#: The export sorts every key of the document, and Swagger UI and Postman both
#: offer the first example of the map as the default. Naming the minimal request
#: ``champion`` would have filed it behind ``challenger`` and handed a first time
#: caller the from scratch model, which is not what the API serves by default.
PREDICT_EXAMPLES: dict[str, dict[str, Any]] = {
    "defaut_echantillonne": {
        "summary": "Defaut : alias champion, tirage aleatoire",
        "description": (
            "Le corps minimal. Sans nom de modele la version promue repond, et "
            "la reponse dit laquelle. Deux envois de ce corps rendent deux "
            "resumes differents ; la graine tiree revient dans decoding."
        ),
        "value": {"text": EXAMPLE_DOCUMENT},
    },
    "faisceaux_evaluation": {
        "summary": "La configuration sous laquelle le ROUGE publie a ete mesure",
        "description": (
            "Quatre faisceaux, sans tirage. Le reste du decodage est deja fixe "
            "aux valeurs de l'evaluation, donc ces deux champs suffisent a "
            "reproduire la mesure."
        ),
        "value": {"text": EXAMPLE_DOCUMENT, "do_sample": False, "num_beams": 4},
    },
    "graine_rejouee": {
        "summary": "Rejouer un resume echantillonne",
        "description": (
            "La graine vient du bloc decoding d'une reponse precedente. Le meme "
            "document, la meme version et la meme graine rendent le meme resume."
        ),
        "value": {"text": EXAMPLE_DOCUMENT, "seed": 1234567},
    },
    "modele_challenger": {
        "summary": "Le Transformer from scratch, via son alias",
        "description": "Voir l'ecart avec la baseline sans passer par /compare.",
        "value": {
            "text": EXAMPLE_DOCUMENT,
            "model": "challenger",
            "do_sample": False,
            "num_beams": 4,
        },
    },
}

#: Bodies offered for ``/compare``, named under the same rule. The unscored one
#: comes first because it is the request that needs nothing but a document, and
#: the difference between it and the next is exactly what the ``scored`` flag of
#: the response reports.
COMPARE_EXAMPLES: dict[str, dict[str, Any]] = {
    "defaut_tous_les_modeles": {
        "summary": "Tous les modeles enregistres, sans score",
        "description": (
            "Sans reference il n'y a rien a mesurer : la reponse porte des "
            "resumes et des latences, et 'scored' vaut faux. Le tirage est "
            "desactive : une comparaison echantillonnee compare deux tirages."
        ),
        "value": {"text": EXAMPLE_DOCUMENT, "do_sample": False, "num_beams": 4},
    },
    "modeles_choisis": {
        "summary": "Deux modeles nommes explicitement",
        "description": "L'ordre demande est l'ordre des resultats.",
        "value": {
            "text": EXAMPLE_DOCUMENT,
            "models": ["pretrained", "scratch"],
            "do_sample": False,
            "num_beams": 4,
        },
    },
    "score_avec_reference": {
        "summary": "Comparaison scoree par le ROUGE de la section 2.1",
        "description": (
            "La reference declenche le calcul. Les scores mesurent ce document, "
            "pas le corpus : ceux de /models restent la reference publiee."
        ),
        "value": {
            "text": EXAMPLE_DOCUMENT,
            "reference": EXAMPLE_REFERENCE,
            "do_sample": False,
            "num_beams": 4,
        },
    },
}


def resolve_seed(options: GenerationOptions) -> int | None:
    """Return the seed a request will run under.

    A deterministic decoding gets none: seeding it would report a number that
    changed nothing. A sampled one always gets a seed, drawn here when the
    caller supplied none, because an answer nobody can reproduce is not an
    answer this API is willing to give.

    Args:
        options: What the request asked for.

    Returns:
        The seed, or ``None`` when the decoding is deterministic.
    """
    if not options.do_sample:
        return None
    if options.seed is not None:
        return options.seed
    # secrets rather than random: bandit flags the latter, and the cost of a
    # cryptographic draw once per request is invisible next to a forward pass.
    return secrets.randbelow(MAX_SEED + 1)


def build_decoding(options: GenerationOptions, settings: Settings) -> GenerationConfig:
    """Turn the knobs of a request into a decoding configuration.

    Args:
        options: What the request asked for.
        settings: The settings of the running instance.

    Returns:
        The configuration handed to the service. It is what the request asked
        for, not necessarily what runs: the service caps the budget to what the
        resolved weights can produce, and reports the result.
    """
    return GenerationConfig(
        max_new_tokens=options.max_new_tokens or settings.max_summary_tokens,
        num_beams=options.num_beams,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        do_sample=options.do_sample,
        temperature=options.temperature,
        top_k=options.top_k,
        top_p=options.top_p,
        seed=resolve_seed(options),
    )


def echo_decoding(config: GenerationConfig) -> Decoding:
    """Render the decoding a generation ran under.

    The sampling fields are reported only when there was a draw. On a
    deterministic run the temperature of the request did nothing, and echoing it
    would invite a caller to believe it mattered.

    Args:
        config: The configuration the service used, already capped.

    Returns:
        The block returned to the client. Built from the configuration itself
        rather than from the request, so the response cannot report a decoding
        that did not run.
    """
    return Decoding(
        max_new_tokens=config.max_new_tokens,
        num_beams=config.num_beams,
        no_repeat_ngram_size=config.no_repeat_ngram_size,
        do_sample=config.do_sample,
        temperature=config.temperature if config.do_sample else None,
        top_k=config.top_k if config.do_sample else None,
        top_p=config.top_p if config.do_sample else None,
        seed=config.seed if config.do_sample else None,
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Resumer un document",
    description=(
        "Resume un document avec un modele du registre. Sans nom de modele, "
        "la requete est servie par l'alias champion. La reponse nomme la version "
        "exacte qui a repondu, pas seulement le nom demande."
    ),
    responses=INFERENCE_ERRORS,
)
async def predict(
    payload: Annotated[PredictRequest, Body(openapi_examples=PREDICT_EXAMPLES)],
    settings: SettingsDep,
    store: StoreDep,
) -> PredictResponse:
    """Summarise one document.

    Args:
        payload: The request.
        settings: The settings of the running instance.
        store: The store of the running instance.

    Returns:
        The summary, the version that produced it and the decoding used.

    Raises:
        ModelNotFoundError: If the name, the alias or the version is unknown.
        ModelNotReadyError: If the version cannot be loaded.
        InferenceTimeoutError: If the generation exceeded the deadline.
        InferenceFailedError: If the model raised.
    """
    prediction = await summarization.summarize(
        store,
        payload.text,
        name=payload.model,
        version=payload.version,
        config=build_decoding(payload, settings),
        timeout_s=settings.inference_timeout_s,
    )
    return PredictResponse(
        summary=prediction.summary,
        model=prediction.version.name,
        version=prediction.version.version,
        latency_ms=prediction.latency_ms,
        decoding=echo_decoding(prediction.decoding),
    )


@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Comparer plusieurs modeles sur un document",
    description=(
        "Execute plusieurs modeles sur le meme document, avec le meme decodage. "
        "Une reference declenche le calcul du ROUGE de la section 2.1 ; sans elle, "
        "la reponse porte des resumes et des latences, et aucun score."
    ),
    responses=INFERENCE_ERRORS,
)
async def compare(
    payload: Annotated[CompareRequest, Body(openapi_examples=COMPARE_EXAMPLES)],
    settings: SettingsDep,
    store: StoreDep,
) -> CompareResponse:
    """Summarise one document with several models.

    Args:
        payload: The request.
        settings: The settings of the running instance.
        store: The store of the running instance.

    Returns:
        One result per model, in the order asked for.

    Raises:
        ModelNotFoundError: If a name is unknown, or if the request named none
            and the registry holds nothing to compare.
        ModelNotReadyError: If one of the versions cannot be loaded.
        InferenceTimeoutError: If one generation exceeded the deadline.
        InferenceFailedError: If one model raised.
    """
    names = payload.models or [version.name for version in store.catalog()]
    if not names:
        raise ModelNotFoundError(
            "Le registre ne contient aucun modele a comparer.",
            details={"registry_backend": settings.registry_backend},
        )

    comparison = await summarization.compare(
        store,
        payload.text,
        names=names,
        reference=payload.reference,
        config=build_decoding(payload, settings),
        timeout_s=settings.inference_timeout_s,
    )

    return CompareResponse(
        results=[
            CompareResult(
                model=prediction.version.name,
                version=prediction.version.version,
                summary=prediction.summary,
                latency_ms=prediction.latency_ms,
                rouge=(
                    None
                    if prediction.rouge is None
                    else {
                        variant: RougeVariantScore(**score.to_dict())
                        for variant, score in prediction.rouge.items()
                    }
                ),
            )
            for prediction in comparison.predictions
        ],
        scored=payload.reference is not None,
        decoding=echo_decoding(comparison.decoding),
    )
