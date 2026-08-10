"""The single record type flowing through the data pipeline.

Every stage consumes and produces :class:`Example` instances. Using one frozen
record instead of raw dictionaries removes a whole class of silent column
renaming bugs between the download and the tokenisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Example:
    """A document paired with its reference summary.

    Attributes:
        example_id: Stable identifier coming from the upstream corpus.
        source: The document to summarise, already cleaned.
        target: The reference summary, already cleaned.
    """

    example_id: str
    source: str
    target: str

    def to_dict(self) -> dict[str, str]:
        """Render the example as a JSON serialisable mapping.

        Returns:
            A mapping with the ``id``, ``source`` and ``target`` keys.
        """
        return {"id": self.example_id, "source": self.source, "target": self.target}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Example:
        """Rebuild an example from its serialised form.

        Args:
            payload: Mapping produced by :meth:`to_dict`.

        Returns:
            The reconstructed example.
        """
        return cls(
            example_id=str(payload["id"]),
            source=str(payload["source"]),
            target=str(payload["target"]),
        )
