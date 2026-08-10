"""What a client sends to ``/predict`` and ``/compare``, and what comes back.

**The bounds live in the settings, and the schema reads them.** Section 21.1
keeps every knob in :class:`backend.app.core.config.Settings`, and the exposure
rules of section 39 require the source text to be bounded by validation rather
than by whatever the model would truncate anyway. Repeating the limit as a
literal here would create a second source of truth, so the validators read the
settings instead. A payload over the limit is refused with ``INVALID_INPUT``
before a single token is encoded.

**Decoding samples by default, and the response says under which seed.** The
API is a demonstrator: a caller who sends the same document twice expects to see
that a language model is not a lookup table, and a decoding that returned the
same sentence forever hid that. So ``do_sample`` defaults to true. What that
normally costs is reproducibility, and it is bought straight back: when a
request carries no ``seed`` the endpoint draws one, generates under it, and
reports it in the ``decoding`` block. Re-sending the same body with that seed
returns the same summary, so no answer this API gives is unreproducible.

**Sampling is off wherever a score is measured.** ``do_sample: false`` with
``num_beams: 4`` is the configuration the published ROUGE was measured under,
and it stays one request away. The two cannot be combined: a sampled beam search
is a third strategy, which the from scratch decoder does not implement.

**A comparison shares one decoding configuration.** Section 2.1 compares two
models under identical conditions. The request carries one budget, one beam
width and one seed for every model it names, because a per model override would
produce a gap that says nothing about the models. Note that a sampled comparison
compares two draws, not two models: to read a gap, send ``do_sample: false``.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.core.config import get_settings

#: Widest beam a client may ask for. Four is what the evaluation uses; past it
#: the latency grows without the score following, and a public endpoint is not
#: the place to explore that.
MAX_BEAMS = 4

#: Most models one comparison may name. The models answer one after the other on
#: a single device, so the ceiling bounds the duration of one request.
MAX_COMPARED_MODELS = 4

#: Hottest draw a client may ask for. Past two the distribution of a summariser
#: is flat enough that the output stops being a summary of anything, and serving
#: that would be offering a knob whose far end is broken.
MAX_TEMPERATURE = 2.0

#: What the endpoint samples under when the request says nothing. Slightly below
#: one: a summary is a faithfulness task, so the default leans towards the
#: likely tokens and the nucleus does the rest of the work.
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.95

#: Bound of the drawn seed. Any 64 bit value would do for torch; this one keeps
#: the number readable in a response and re-typable by hand into a replay.
MAX_SEED = 2**31 - 1


class Decoding(BaseModel):
    """The decoding a response was produced under.

    Echoing it back is what makes a summary reproducible: re-sending the same
    document to the same version with this block copied into the request returns
    the same summary, whether the generation was sampled or not.

    **The sampling fields are null when nothing was sampled.** A deterministic
    run has no temperature and no seed to report, and filling them with the
    values that happened to sit in the request would describe a decoding that
    did not take place.

    Attributes:
        max_new_tokens: Generated token budget.
        num_beams: Beam width. One is greedy search.
        no_repeat_ngram_size: Repeated n-gram the decoding forbade. Fixed by the
            instance to the value the evaluation of section 10 uses, and
            reported because a summary produced under a different constraint is
            not the same summary.
        do_sample: Whether the tokens were drawn rather than maximised.
        temperature: Temperature of the draw, null on a deterministic run.
        top_k: Truncation applied before the draw, null on a deterministic run.
            Zero means no truncation.
        top_p: Nucleus kept before the draw, null on a deterministic run.
        seed: Seed the draw ran under, null on a deterministic run. Never null
            on a sampled one, even when the request carried none: the endpoint
            draws a seed rather than leaving the run unreplayable.
    """

    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int = Field(description="Budget de tokens generes.")
    num_beams: int = Field(description="Largeur de faisceau, un vaut recherche gloutonne.")
    no_repeat_ngram_size: int = Field(description="Taille de n-gramme interdite en repetition.")
    do_sample: bool = Field(description="Les tokens ont ete tires, et non maximises.")
    temperature: float | None = Field(
        default=None, description="Temperature du tirage. Nulle si le decodage etait deterministe."
    )
    top_k: int | None = Field(
        default=None, description="Troncature top_k appliquee. Nulle si aucun tirage."
    )
    top_p: float | None = Field(
        default=None, description="Noyau top_p conserve. Nul si aucun tirage."
    )
    seed: int | None = Field(
        default=None,
        description=(
            "Graine du tirage. Nulle si aucun tirage, sinon toujours presente : "
            "la renvoyer dans une requete identique rejoue exactement ce resume."
        ),
    )


class RougeVariantScore(BaseModel):
    """One ROUGE variant, in its three forms.

    Attributes:
        precision: Share of the generated units that appear in the reference.
        recall: Share of the reference units that appear in the generation.
        fmeasure: Harmonic mean of the two, the reported form.
    """

    model_config = ConfigDict(extra="forbid")

    precision: float = Field(description="Precision de la variante.")
    recall: float = Field(description="Rappel de la variante.")
    fmeasure: float = Field(description="F-mesure, la forme reportee.")


class GenerationOptions(BaseModel):
    """The decoding knobs a request may carry, shared by both endpoints.

    Attributes:
        max_new_tokens: Generated token budget. ``None`` leaves the budget of
            the instance, ``SYNTRA_MAX_SUMMARY_TOKENS``.
        num_beams: Beam width, one to :data:`MAX_BEAMS`.
        do_sample: Draw the tokens instead of maximising them. True by default,
            so two identical requests return two different summaries.
        temperature: Temperature of the draw, up to :data:`MAX_TEMPERATURE`.
        top_k: Keep only the ``k`` most likely tokens. Zero disables the cut.
        top_p: Keep the nucleus carrying this much probability mass.
        seed: Seed to draw under. ``None`` lets the endpoint draw one and report
            it, which is what keeps every sampled answer replayable.
    """

    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Budget de tokens generes. Borne par SYNTRA_MAX_SUMMARY_TOKENS.",
    )
    num_beams: int = Field(
        default=1,
        ge=1,
        le=MAX_BEAMS,
        description=(
            "Largeur de faisceau. Un vaut recherche gloutonne. "
            "Le ROUGE publie a ete mesure avec quatre, et do_sample a faux."
        ),
    )
    do_sample: bool = Field(
        default=True,
        description=(
            "Tire chaque token au lieu de prendre le plus probable. Deux appels "
            "identiques rendent alors deux resumes differents. Mettre a faux "
            "pour un decodage reproductible sans passer de graine."
        ),
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        gt=0,
        le=MAX_TEMPERATURE,
        description="Temperature du tirage. Sous un, le tirage se rapproche du glouton.",
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=0,
        description="Ne tirer que parmi les k tokens les plus probables. Zero desactive.",
    )
    top_p: float = Field(
        default=DEFAULT_TOP_P,
        gt=0,
        le=1,
        description="Ne tirer que dans le noyau portant cette masse de probabilite.",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        le=MAX_SEED,
        description=(
            "Graine du tirage. Absente, l'instance en tire une et la renvoie "
            "dans le bloc decoding : la rejouer redonne exactement le meme resume."
        ),
    )

    @model_validator(mode="after")
    def _sampling_and_beams_are_exclusive(self) -> Self:
        """Refuse a sampled beam search rather than silently dropping one half.

        Returns:
            The validated options.

        Raises:
            ValueError: If both are asked for. Scoring beams on draws is a third
                strategy, and the from scratch decoder implements two.
        """
        if self.do_sample and self.num_beams > 1:
            raise ValueError(
                "do_sample et num_beams superieur a un sont exclusifs. "
                "Passer do_sample a faux pour une recherche par faisceaux."
            )
        return self

    @field_validator("max_new_tokens")
    @classmethod
    def _within_the_instance_budget(cls, value: int | None) -> int | None:
        """Refuse a budget above what the instance is configured to generate.

        Args:
            value: The requested budget.

        Returns:
            The validated budget.

        Raises:
            ValueError: If it exceeds ``SYNTRA_MAX_SUMMARY_TOKENS``.
        """
        limit = get_settings().max_summary_tokens
        if value is not None and value > limit:
            raise ValueError(f"max_new_tokens ne peut pas depasser {limit} tokens.")
        return value


def _validate_text(value: str, field: str) -> str:
    """Refuse a blank document and one over the configured limit.

    Args:
        value: The submitted text.
        field: Name of the field, used in the message.

    Returns:
        The text, unchanged. It is not stripped: what the model reads is what
        the client sent, and a silent rewrite would make a reported length
        disagree with the payload.

    Raises:
        ValueError: If the text is blank or longer than
            ``SYNTRA_MAX_INPUT_CHARS``.
    """
    if not value.strip():
        raise ValueError(f"{field} ne peut pas etre vide.")

    limit = get_settings().max_input_chars
    if len(value) > limit:
        raise ValueError(f"{field} depasse la limite de {limit} caracteres.")
    return value


class PredictRequest(GenerationOptions):
    """One document to summarise, with one model.

    Attributes:
        text: The document. Bounded by ``SYNTRA_MAX_INPUT_CHARS``.
        model: A registered name or a deployment alias. ``None`` resolves to
            the champion, which is the version the deployment promoted.
        version: An exact version. Refused next to an alias, which already
            names one.
    """

    text: str = Field(description="Document a resumer.")
    model: str | None = Field(default=None, description="Nom enregistre ou alias de deploiement.")
    version: str | None = Field(default=None, description="Version exacte, si elle est epinglee.")

    @field_validator("text")
    @classmethod
    def _bounded_text(cls, value: str) -> str:
        """Validate the document against the limits of the instance.

        Args:
            value: The submitted document.

        Returns:
            The validated document.

        Raises:
            ValueError: If it is blank or over the configured limit.
        """
        return _validate_text(value, "text")


class PredictResponse(BaseModel):
    """The summary, and what produced it.

    Attributes:
        summary: The generated text. May be empty: a model that produces
            nothing is a measurement, not a failure.
        model: The registered name that answered.
        version: The exact version that answered, whatever the request named.
        latency_ms: Wall clock duration of the generation alone.
        decoding: The decoding the summary was produced under.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="Resume genere.")
    model: str = Field(description="Nom du modele ayant repondu.")
    version: str = Field(description="Version exacte ayant repondu.")
    latency_ms: float = Field(description="Duree de la generation seule, en millisecondes.")
    decoding: Decoding = Field(description="Decodage applique.")


class CompareRequest(GenerationOptions):
    """One document, several models, one optional reference.

    Attributes:
        text: The document. Bounded by ``SYNTRA_MAX_INPUT_CHARS``.
        models: Names or aliases to compare, in the order they are reported.
            ``None`` compares every registered name, which is the comparison of
            section 2.1.
        reference: A reference summary. When present, every prediction is
            scored against it. Its absence is why the metrics block is
            optional: without a reference there is nothing to score, and
            returning zeros would be a fabricated measurement.
    """

    text: str = Field(description="Document a resumer.")
    models: list[str] | None = Field(
        default=None,
        max_length=MAX_COMPARED_MODELS,
        description="Modeles compares. Absent, tous les noms enregistres.",
    )
    reference: str | None = Field(
        default=None, description="Resume de reference. Declenche le calcul du ROUGE."
    )

    @field_validator("text")
    @classmethod
    def _bounded_text(cls, value: str) -> str:
        """Validate the document against the limits of the instance.

        Args:
            value: The submitted document.

        Returns:
            The validated document.

        Raises:
            ValueError: If it is blank or over the configured limit.
        """
        return _validate_text(value, "text")

    @field_validator("reference")
    @classmethod
    def _bounded_reference(cls, value: str | None) -> str | None:
        """Validate the reference summary, when one is supplied.

        A blank reference is refused rather than ignored: scoring against it
        would report a zero that measures the corpus, not the model.

        Args:
            value: The submitted reference.

        Returns:
            The validated reference.

        Raises:
            ValueError: If it is blank or over the configured limit.
        """
        return None if value is None else _validate_text(value, "reference")

    @field_validator("models")
    @classmethod
    def _distinct_models(cls, value: list[str] | None) -> list[str] | None:
        """Refuse the same model twice in one comparison.

        Args:
            value: The requested names.

        Returns:
            The validated list.

        Raises:
            ValueError: If the list is empty or holds a duplicate. Running one
                model twice would produce two identical rows and double the
                duration of the request.
        """
        if value is None:
            return None
        if not value:
            raise ValueError("models ne peut pas etre une liste vide.")
        if len(set(value)) != len(value):
            raise ValueError("models ne peut pas nommer deux fois le meme modele.")
        return value


class CompareResult(BaseModel):
    """One model, in a comparison.

    Attributes:
        model: The registered name that answered.
        version: The exact version that answered.
        summary: The generated text.
        latency_ms: Wall clock duration of that generation.
        rouge: Scores against the reference, one entry per variant of section
            2.1. Absent when the request carried no reference.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(description="Nom du modele.")
    version: str = Field(description="Version exacte ayant repondu.")
    summary: str = Field(description="Resume genere.")
    latency_ms: float = Field(description="Duree de la generation, en millisecondes.")
    rouge: dict[str, RougeVariantScore] | None = Field(
        default=None, description="Scores ROUGE, si une reference a ete fournie."
    )


class CompareResponse(BaseModel):
    """The comparison.

    Attributes:
        results: One entry per model, in the order asked for.
        scored: Whether a reference was supplied. Stated rather than inferred
            from a null, so a client never reads an absent metrics block as a
            score of zero.
        decoding: The decoding every model was run under.
    """

    model_config = ConfigDict(extra="forbid")

    results: list[CompareResult] = Field(description="Un resultat par modele.")
    scored: bool = Field(description="Une reference a ete fournie et le ROUGE a ete calcule.")
    decoding: Decoding = Field(description="Decodage partage par tous les modeles.")
