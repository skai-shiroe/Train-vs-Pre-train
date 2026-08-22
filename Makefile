# Syntra. Les cibles de la chaine ML.

ifeq ($(OS),Windows_NT)
VENV_BIN := .venv/Scripts
else
VENV_BIN := .venv/bin
endif

# Repli sur l'interpreteur du PATH quand le venv du depot est absent ou
# incomplet. Un clone frais n'en a pas, et make echouait alors sur un fichier
# introuvable : l'erreur ne disait rien du vrai probleme, et aucune cible ne
# tournait tant que le venv n'etait pas recree. PY=... a la ligne de commande
# reste prioritaire.
PY     ?= $(if $(wildcard $(VENV_BIN)/python*),$(VENV_BIN)/python,python)
PIP    := $(PY) -m pip
PYTEST := $(PY) -m pytest

COV_ARGS := --cov=src \
            --cov-report=xml:reports/coverage.xml \
            --cov-report=term-missing \
            --cov-fail-under=80

# cu130 et non cu128 : le plancher torch est a 2.13 pour GHSA-rrmf-rvhw-rf47, et
# l'index cu128 s'arrete a 2.11. Demande un pilote NVIDIA compatible CUDA 13.
TORCH_INDEX ?= https://download.pytorch.org/whl/cu130

.DEFAULT_GOAL := help
.PHONY: help install test test-unit test-integration coverage \
        data train-scratch train-pretrained evaluate ablation figures mlflow-ui \
        report-sync corpus-sync reproduce clean

help: ## Affiche les cibles disponibles
	@echo "Cibles disponibles :"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

install: ## Installe les dependances
	$(PIP) install --upgrade pip
	$(PIP) install "torch>=2.13,<3.0" --index-url $(TORCH_INDEX)
	$(PIP) install -e ".[dev]"

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test: ## Lance toute la suite
	$(PYTEST)

test-unit: ## Lance les tests unitaires
	$(PYTEST) tests/unit

test-integration: ## Lance les tests d'integration
	$(PYTEST) tests/integration

coverage: ## Lance les tests avec le seuil de couverture de 80 pour cent
	@mkdir -p reports
	$(PYTEST) $(COV_ARGS) --junitxml=reports/junit.xml

# ---------------------------------------------------------------------------
# Chaine ML
# ---------------------------------------------------------------------------

data: ## Telecharge, valide et prepare le corpus de travail
	$(PY) -m src.data.build --config configs/data/cnn_dailymail.yaml

train-scratch: ## Entraine le Transformer from scratch sur 100 pour cent du corpus
	$(PY) -m src.experiments.run --config configs/experiments/scratch_100.yaml

train-pretrained: ## Fine-tune T5 sur 100 pour cent du corpus
	$(PY) -m src.experiments.run --config configs/experiments/pretrained_ft_100.yaml

evaluate: ## Evalue la baseline zero-shot sur le jeu de test commun
	$(PY) -m src.experiments.run --config configs/experiments/pretrained_zero_shot.yaml

ablation: ## Rejoue les ablations taille de corpus et architecture
	$(PY) -m src.experiments.ablation --study dataset_size
	$(PY) -m src.experiments.ablation --study architecture

# Les figures lisent les memes enregistrements que les tableaux, pas les CSV
# qu'ils produisent : une figure tracee depuis une valeur arrondie ne dit plus
# la meme chose que le tableau a cote.
figures: ## Trace les quatre figures dans reports/figures
	$(PY) -m src.experiments.figures

# Le magasin est passe explicitement : sans argument, la commande bascule vers
# ./mlruns des que ce repertoire existe, alors que le client de tracking ecrit
# dans la base SQLite dans tous les cas.
mlflow-ui: ## Sert l'interface MLflow sur le magasin local, port 5000
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db

# ---------------------------------------------------------------------------
# Artefacts derives
# ---------------------------------------------------------------------------

# Cette cible lit reports/results, que .gitignore garde hors du depot. Sur une
# machine sans campagne elle reecrirait tous les tableaux en experiences
# NOT_RUN, ce qui est la raison pour laquelle elle reste manuelle.
report-sync: ## Regenere les tableaux de resultats a partir des enregistrements de runs
	$(PY) -m src.experiments.fragments

# Separee de report-sync, avec une autre source : celle-ci lit data/processed,
# que .gitignore garde hors du depot et que make data reconstruit. Une machine
# peut porter un corpus sans campagne, et l'inverse.
corpus-sync: ## Regenere les tableaux du corpus a partir du manifeste et des statistiques
	$(PY) -m src.data.fragments

# ---------------------------------------------------------------------------
# Agregats
# ---------------------------------------------------------------------------

reproduce: ## Reproduit la chaine scientifique complete. MODE=full ou quick
	$(PY) -m src.experiments.reproduce --mode $(or $(MODE),quick)

clean: ## Supprime les caches et les rapports generes
	rm -rf .pytest_cache htmlcov
	rm -f .coverage reports/junit.xml reports/coverage.xml
