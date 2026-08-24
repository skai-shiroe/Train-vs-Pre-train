"""Unit tests for the notebooks shipped with the repository."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))

# Ce que `make kernel` enregistre. Les deux doivent bouger ensemble.
KERNEL_NAME = "train-vs-pre-train"
KERNEL_DISPLAY_NAME = "Train-vs-Pre-train (.venv)"


@pytest.mark.unit
def test_the_repository_ships_the_expected_notebooks() -> None:
    assert [path.name for path in NOTEBOOKS] == [
        "00_environment_check.ipynb",
        "01_eda_cnn_dailymail.ipynb",
        "02_training.ipynb",
        "03_transformer_walkthrough.ipynb",
    ]


@pytest.mark.unit
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_every_notebook_is_pinned_to_the_repository_kernel(notebook: Path) -> None:
    """Ouvrir un carnet dans VS Code suffit a reecrire ce bloc.

    L'editeur y remet le noyau qu'il vient de choisir, et `python3` designe
    l'interpreteur ambiant, pas le venv du depot. Le carnet tourne alors avec
    d'autres versions que celles que `make install` a posees, et rien ne le
    signale a l'execution. Ce test est le seul endroit qui le voit.
    """
    kernelspec = json.loads(notebook.read_text(encoding="utf-8"))["metadata"]["kernelspec"]

    assert kernelspec["name"] == KERNEL_NAME
    assert kernelspec["display_name"] == KERNEL_DISPLAY_NAME
