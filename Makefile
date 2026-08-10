# Syntra. Toutes les cibles de la section 40 du cahier des charges.
#
# make ci reproduit localement le verdict du pipeline GitHub Actions. Un developpeur
# doit pouvoir le lancer avant de pousser.

ifeq ($(OS),Windows_NT)
VENV_BIN := .venv/Scripts
else
VENV_BIN := .venv/bin
endif

PY       := $(VENV_BIN)/python
PIP      := $(PY) -m pip
PYTEST   := $(PY) -m pytest
SRC_DIRS := src backend/app
# backend/tests ne fait pas partie de testpaths par hasard : c'est un repertoire
# de tests a part entiere, que les hooks pre-commit formatent deja parce qu'ils
# travaillent par fichier. L'oublier ici laissait make lint valider un arbre que
# le hook reformatait ensuite.
FMT_DIRS := $(SRC_DIRS) tests backend/tests scripts
COV_ARGS := --cov=src --cov=backend/app \
            --cov-report=xml:reports/coverage.xml \
            --cov-report=term-missing \
            --cov-fail-under=80

IMAGE        ?= syntra-backend
IMAGE_TAG    ?= $(IMAGE):dev
TRAINING_TAG ?= syntra-training:dev
# cu130 et non cu128 : le plancher torch est a 2.13 pour GHSA-rrmf-rvhw-rf47, et
# l'index cu128 s'arrete a 2.11. Demande un pilote NVIDIA compatible CUDA 13.
TORCH_INDEX  ?= https://download.pytorch.org/whl/cu130

# Roues CPU pour les images : servir sur GPU demande une image de base CUDA, pas
# seulement une autre roue. Le poste de developpement garde l'index CUDA.
TORCH_CPU_INDEX ?= https://download.pytorch.org/whl/cpu

# Revision inscrite dans les labels OCI de l'image. Le depot peut ne porter
# aucun commit : la construction ne doit pas echouer pour autant.
GIT_REF := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)

# Vulnerabilites connues, tracees et acceptees. Chaque identifiant doit etre
# justifie dans le commit qui l'ajoute : une entree sans justification est une
# vulnerabilite masquee, pas une vulnerabilite traitee. La liste se relit a
# chaque montee de transformers.
AUDIT_IGNORES := --ignore-vuln PYSEC-2025-217 \
                 --ignore-vuln PYSEC-2026-2288 \
                 --ignore-vuln PYSEC-2026-2289 \
                 --ignore-vuln PYSEC-2026-2290 \
                 --ignore-vuln PYSEC-2026-3552

.DEFAULT_GOAL := help
.PHONY: help install hooks format lint spell prose typecheck security \
        test test-unit test-integration test-e2e test-smoke coverage \
        data train-scratch train-pretrained evaluate ablation figures mlflow-ui \
        api docker-build docker-build-training docker-size docker-scan \
        docker-up docker-down docker-train \
        api-sync report-sync report-lint corpus-sync deploy ci reproduce clean

help: ## Affiche les cibles disponibles
	@echo "Cibles disponibles :"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

install: ## Installe les dependances et pose les hooks
	$(PIP) install --upgrade pip
	$(PIP) install "torch>=2.13,<3.0" --index-url $(TORCH_INDEX)
	$(PIP) install -e ".[api,train,dev]"
	$(MAKE) hooks

hooks: ## Installe les hooks pre-commit, commit-msg et pre-push
	$(PY) -m pre_commit install
	$(PY) -m pre_commit install --hook-type commit-msg
	$(PY) -m pre_commit install --hook-type pre-push

# ---------------------------------------------------------------------------
# Qualite
# ---------------------------------------------------------------------------

format: ## Formate le code avec black et corrige ce que ruff sait corriger
	$(PY) -m black $(FMT_DIRS)
	$(PY) -m ruff check --fix $(FMT_DIRS)

lint: ## Verifie le formatage et le style, sans rien modifier
	$(PY) -m black --check --diff $(FMT_DIRS)
	$(PY) -m flake8 $(FMT_DIRS)
	$(PY) -m ruff check $(FMT_DIRS)
	$(PY) -m yamllint -c .yamllint.yaml --strict .
	$(PY) scripts/check_dashes.py $$(git ls-files '*.py' '*.md' '*.yml' '*.yaml' '*.toml')

spell: ## Verifie l'orthographe avec codespell
	$(PY) -m codespell_lib src backend scripts RAPPORT.md README.md \
	  --skip="*.lock,*.svg,*.png,data/*,.git" \
	  --ignore-words=.codespell-ignore

prose: ## Verifie la grammaire, le style et la terminologie du rapport
	@command -v vale >/dev/null 2>&1 || { \
	  echo "vale n'est pas installe : https://vale.sh/docs/install"; exit 1; }
	vale RAPPORT.md README.md

typecheck: ## Verifie le typage statique avec mypy
	$(PY) -m mypy

security: ## Analyse la securite du code et des dependances
	@mkdir -p reports
	$(PY) -m bandit -c pyproject.toml -r $(SRC_DIRS) -f json -o reports/bandit.json
	$(PY) -m bandit -c pyproject.toml -r $(SRC_DIRS) -ll
	$(PY) -m pip_audit --strict -s osv $(AUDIT_IGNORES)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test: ## Lance les tests hors smoke
	$(PYTEST) -m "not smoke"

test-unit: ## Lance les tests unitaires
	$(PYTEST) tests/unit backend/tests -m "not smoke"

test-integration: ## Lance les tests d'integration
	$(PYTEST) tests/integration

test-e2e: ## Lance les tests de bout en bout sur l'API
	$(PYTEST) tests/e2e

test-smoke: ## Lance les smoke tests contre BASE_URL
	@test -n "$(BASE_URL)" || { echo "BASE_URL doit etre defini."; exit 1; }
	BASE_URL=$(BASE_URL) $(PYTEST) tests/smoke -m smoke

coverage: ## Lance les tests avec le seuil de couverture de 80 pour cent
	@mkdir -p reports
	$(PYTEST) -m "not smoke" $(COV_ARGS) --junitxml=reports/junit.xml

# ---------------------------------------------------------------------------
# Chaine ML
# ---------------------------------------------------------------------------

data: ## Telecharge, valide et prepare le corpus de travail
	$(PY) -m src.data.build --config configs/data/xsum.yaml

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
# API et conteneurs
# ---------------------------------------------------------------------------

# --no-access-log : l'application journalise elle-meme chaque requete, sans la
# query string. Le log d'acces d'uvicorn ecrit l'URL complete, donc un document
# passe en parametre finirait sur disque.
api: ## Lance l'API en local avec rechargement automatique
	$(PY) -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000 --no-access-log

docker-build: ## Construit l'image backend
	docker build -f backend/Dockerfile \
	  --build-arg VCS_REF=$(GIT_REF) \
	  -t $(IMAGE_TAG) .

docker-build-training: ## Construit l'image d'entrainement. TORCH_INDEX pour le GPU
	docker build -f docker/training.Dockerfile \
	  --build-arg VCS_REF=$(GIT_REF) \
	  --build-arg TORCH_INDEX=$(TORCH_CPU_INDEX) \
	  -t $(TRAINING_TAG) .

docker-size: ## Mesure la taille de l'image backend construite
	@docker image inspect $(IMAGE_TAG) --format '{{.Size}}' \
	  | awk '{printf "%s : %.0f Mio\n", "$(IMAGE_TAG)", $$1 / 1048576}'

docker-scan: ## Scanne l'image, echoue sur une vulnerabilite CRITICAL
	@mkdir -p reports
	trivy image --exit-code 1 --severity CRITICAL --ignore-unfixed \
	  --format table $(IMAGE_TAG)
	trivy image --format json --output reports/trivy.json $(IMAGE_TAG)

docker-up: ## Demarre la stack locale sans lancer d'entrainement
	docker compose up -d backend mlflow postgres minio

docker-down: ## Arrete la stack locale
	docker compose down

# Le profil training garde l'entrainement hors de docker compose up. La
# configuration est obligatoire : il n'y a pas d'entrainement par defaut.
docker-train: ## Lance un entrainement dans la stack. CONFIG=configs/experiments/....yaml
	@test -n "$(CONFIG)" || { echo "CONFIG doit etre defini."; exit 1; }
	docker compose --profile training run --rm training \
	  -m src.experiments.run --config $(CONFIG)

# ---------------------------------------------------------------------------
# Artefacts derives
# ---------------------------------------------------------------------------

api-sync: ## Regenere le contrat d'API et l'environnement Postman
	$(PY) scripts/export_openapi.py \
	  --output backend/openapi.json \
	  --environment-output backend/syntra.postman_environment.json

# Separee de api-sync : celle-ci lit reports/results, que .gitignore garde hors
# du depot. Sur un runner sans campagne elle reecrirait tous les tableaux en
# experiences NOT_RUN. C'est pour la meme raison que la verification
# correspondante rend PENDING plutot que OK quand il n'y a aucun run.
report-sync: ## Regenere les tableaux de resultats a partir des enregistrements de runs
	$(PY) -m src.experiments.fragments

# Separee de report-sync pour la meme raison, avec une autre source : celle-ci
# lit data/processed, que .gitignore garde hors du depot et que make data
# reconstruit. Une machine peut porter un corpus sans campagne, et l'inverse.
corpus-sync: ## Regenere les tableaux du corpus a partir du manifeste et des statistiques
	$(PY) -m src.data.fragments

# markdownlint n'existe qu'en paquet npm : la cible reste separee de make ci
# pour que le verdict local ne depende pas d'une chaine node.
report-lint: ## Valide le rapport et le README
	$(PY) -m codespell_lib RAPPORT.md README.md --ignore-words=.codespell-ignore
	npx --yes markdownlint-cli --config .markdownlint.yaml RAPPORT.md README.md
	$(PY) scripts/check_sync.py

# ---------------------------------------------------------------------------
# Deploiement
# ---------------------------------------------------------------------------

deploy: ## Deploie l'image taggee sur l'hote Docker cible
	deploy/scripts/deploy.sh

# ---------------------------------------------------------------------------
# Agregats
# ---------------------------------------------------------------------------

# check_sync et non report-lint : la cible complete appelle markdownlint, qui
# n'existe qu'en paquet npm. Faire dependre le verdict local d'une chaine node
# le rendrait injouable sur un poste qui n'en a pas, alors que la verification
# de derive est du Python pur.
ci: ## Reproduit localement la sequence complete du pipeline
	$(MAKE) lint
	$(MAKE) spell
	$(MAKE) typecheck
	$(MAKE) security
	$(PY) scripts/check_sync.py
	$(MAKE) coverage

reproduce: ## Reproduit la chaine scientifique complete. MODE=full ou quick
	$(PY) -m src.experiments.reproduce --mode $(or $(MODE),quick)

clean: ## Supprime les caches et les rapports generes
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov
	rm -f .coverage reports/junit.xml reports/coverage.xml reports/bandit.json
