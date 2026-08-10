"""Fixtures shared by the backend unit tests.

The backend is tested without a network and without the corpus: what these
tests check is the wiring around a model, never what a model produces. The
tokeniser is therefore faked here rather than downloaded, and the vocabulary is
small enough for a Transformer to be built in a millisecond.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from backend.app.registry.local import WEIGHTS_PAYLOAD_KEY, WEIGHTS_PAYLOAD_VERSION
from backend.app.registry.models import ModelSource, ModelVersion

#: Small enough for a unit test, large enough that two words rarely collide.
FAKE_VOCAB_SIZE = 32


class FakeTokenizer:
    """Enough of the tokeniser surface for a summariser to run offline."""

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, vocab_size: int = FAKE_VOCAB_SIZE) -> None:
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.vocab_size

    def _ids(self, text: str, max_length: int) -> list[int]:
        words = text.split()
        ids = [2 + sum(ord(c) for c in word) % (self.vocab_size - 2) for word in words]
        return [*ids[: max_length - 1], self.eos_token_id]

    def __call__(
        self,
        texts: list[str],
        *,
        max_length: int = 512,
        truncation: bool = True,
        padding: str = "longest",
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        rows = [self._ids(text, max_length) for text in texts]
        width = max(len(row) for row in rows)

        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        attention_mask = torch.zeros(len(rows), width, dtype=torch.long)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention_mask[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch_decode(self, sequences: torch.Tensor, skip_special_tokens: bool = True) -> list[str]:
        special = {self.pad_token_id, self.eos_token_id} if skip_special_tokens else set()
        return [
            " ".join(f"t{int(token)}" for token in row if int(token) not in special)
            for row in sequences
        ]


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    """Return a tokeniser that needs no network."""
    return FakeTokenizer()


#: Architecture block of a Transformer small enough to build in a test. The
#: source truncation of the tokeniser block must stay under ``max_position``,
#: which the summariser checks when it is built.
TINY_SCRATCH: dict[str, Any] = {
    "type": "scratch",
    "d_model": 16,
    "num_heads": 2,
    "encoder_layers": 1,
    "decoder_layers": 1,
    "d_ff": 32,
    "dropout": 0.0,
    "max_position": 32,
    "tie_embeddings": True,
    "norm_first": True,
}

#: Tokeniser block published beside it.
TINY_TOKENIZER: dict[str, Any] = {
    "hf_id": "t5-small",
    "source_prefix": "summarize: ",
    "max_source_tokens": 16,
    "max_target_tokens": 8,
}


def build_version(**overrides: Any) -> ModelVersion:
    """Build a published version, defaulting to a tiny from scratch one.

    Args:
        **overrides: Fields replacing the defaults.

    Returns:
        The version document.
    """
    fields: dict[str, Any] = {
        "name": "scratch",
        "version": "v1",
        "source": ModelSource.CHECKPOINT,
        "artifact": "scratch/v1/weights.pt",
        "checksum": "0" * 64,
        "architecture": dict(TINY_SCRATCH),
        "tokenizer": dict(TINY_TOKENIZER),
        "experiment": "scratch_100",
        "dataset_version": "0" * 16,
        "git_commit": "unknown",
        "metrics": {"rougeL_f": 0.21},
        "created_at": "2026-01-01T00:00:00+00:00",
        "description": "Transformer from scratch, corpus complet.",
    }
    fields.update(overrides)
    return ModelVersion(**fields)


@pytest.fixture
def version_factory() -> Callable[..., ModelVersion]:
    """Return the version builder as a fixture."""
    return build_version


def write_weights(
    path: Path, state: dict[str, Any], version: int = WEIGHTS_PAYLOAD_VERSION
) -> Path:
    """Write a registry weights payload.

    Args:
        path: Destination file.
        state: The state dictionary to store.
        version: Payload version marker.

    Returns:
        The path written.
    """
    torch.save({"version": version, WEIGHTS_PAYLOAD_KEY: state}, path)
    return path
