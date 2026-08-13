"""Selection of a pretrained baseline by name.

An experiment file names its baseline as a string. Resolving that string here
means the experiment layer never imports a model class directly, and adding a
baseline is one class plus one entry in :data:`BASELINES`.

**Only T5 is registered.** Section 9 lists mT5, mBART and MarianMT as
alternatives, and its file listing includes ``mbart.py``. Section 2.1 locks the
task to English summarisation and the pretrained model to ``t5-small``, which
leaves a multilingual translation model nothing to do here. A wrapper no
experiment builds and no test covers would be dead weight that still has to be
maintained, so it is not written. The registry below is the extension point if
that decision is ever reopened.
"""

from __future__ import annotations

from src.models.pretrained.base import BaselineConfig, PretrainedSummarizer
from src.models.pretrained.t5 import T5Summarizer

#: Baselines an experiment file may name.
BASELINES: dict[str, type[PretrainedSummarizer]] = {
    T5Summarizer.name: T5Summarizer,
}


def available_baselines() -> tuple[str, ...]:
    """List the baselines an experiment may ask for.

    Returns:
        The registered names, sorted.
    """
    return tuple(sorted(BASELINES))


def get_baseline_class(name: str) -> type[PretrainedSummarizer]:
    """Resolve a baseline name into its class.

    Args:
        name: Name used in the experiment file, for example ``t5``.

    Returns:
        The class implementing that baseline.

    Raises:
        KeyError: If the name is not registered. The message lists what is.
    """
    try:
        return BASELINES[name]
    except KeyError:
        available = ", ".join(available_baselines())
        raise KeyError(f"Unknown baseline {name!r}. Available: {available}.") from None


def build_summarizer(
    name: str,
    config: BaselineConfig | None = None,
    *,
    fine_tuned: bool = False,
) -> PretrainedSummarizer:
    """Build a pretrained baseline from its name.

    Args:
        name: Name used in the experiment file, for example ``t5``.
        config: Identification and input budget. Defaults to the class defaults
            of the selected baseline.
        fine_tuned: Whether the weights being loaded were trained on the working
            corpus.

    Returns:
        The baseline, ready for inference or for fine tuning.

    Raises:
        KeyError: If the name is not registered.
    """
    return get_baseline_class(name).from_pretrained(config, fine_tuned=fine_tuned)
