# Image de service de l'API Syntra. Section 28 du cahier des charges.
#
# Deux etapes. La premiere installe les dependances dans un environnement
# virtuel isole, la seconde ne recoit que cet environnement : aucune toolchain,
# aucun cache pip, aucun fichier de developpement n'atteint l'image finale.
#
# Ce qui n'entre jamais dans l'image :
#   les poids, qui viennent du Model Registry monte au demarrage,
#   le corpus, qui n'a rien a faire dans une image de service,
#   les tests et la documentation, ecartes par .dockerignore,
#   tout secret : la configuration entre par l'environnement, jamais par un layer.
#
# Construction :
#   docker build -f backend/Dockerfile -t syntra-backend:dev .

# Tag de base partage par les deux etapes. Les deux doivent porter la meme
# version mineure de Python : l'environnement virtuel copie d'une etape a
# l'autre pointe vers /usr/local/lib/python3.12.
ARG PYTHON_TAG=3.12-slim-bookworm

# ---------------------------------------------------------------------------
# Etape 1 : construction de l'environnement
# ---------------------------------------------------------------------------
FROM python:${PYTHON_TAG} AS builder

# Aucun compilateur n'est installe. Toutes les dependances de l'extra api
# publient une roue manylinux pour CPython 3.12, et rouge-score est une source
# pure Python. Une dependance qui exigerait un compilateur s'ajouterait ici, et
# nulle part ailleurs : c'est le sens de l'etape separee.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PATH="/opt/venv/bin:${PATH}"

# Index des roues torch. Le defaut est CPU : servir sur GPU demande une image de
# base CUDA, pas seulement une autre roue. Voir docs/deployment.md.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

# Outillage de build epingle a l'exact : c'est lui qui resout les dependances,
# donc une derive ici est une derive de tout le reste.
ARG PIP_VERSION=24.3.1
# 84.0.0 et non la serie 75 : PYSEC-2025-49 se corrige en 78.1.1 et
# PYSEC-2026-3447 en 83.0.0. Un outil de build epingle sous son correctif reste
# une dependance vulnerable embarquee dans l'image.
ARG SETUPTOOLS_VERSION=84.0.0
ARG WHEEL_VERSION=0.45.1

WORKDIR /build

RUN python -m venv /opt/venv \
 && pip install --upgrade \
      "pip==${PIP_VERSION}" \
      "setuptools==${SETUPTOOLS_VERSION}" \
      "wheel==${WHEEL_VERSION}"

# Les dependances avant le code source. Les deux paquets crees ici sont des
# coquilles vides : ce que cette couche installe, ce sont les dependances
# declarees par l'extra api, et elle reste valide en cache tant que
# pyproject.toml ne bouge pas. Le code source arrive a la couche suivante.
COPY pyproject.toml README.md ./
RUN mkdir -p src backend \
 && touch src/__init__.py backend/__init__.py \
 && pip install "torch>=2.13,<3.0" --index-url "${TORCH_INDEX}" \
 && pip install ".[api]"

# Le code source, puis le projet lui-meme sans retoucher aux dependances.
COPY src ./src
COPY backend ./backend
RUN pip install --no-deps . \
 && find /opt/venv -type d -name '__pycache__' -prune -exec rm -rf {} + \
 && find /opt/venv -type f -name '*.pyc' -delete

# ---------------------------------------------------------------------------
# Etape 2 : image de service
# ---------------------------------------------------------------------------
FROM python:${PYTHON_TAG} AS runtime

ARG BUILD_VERSION=0.1.0
ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="syntra-backend" \
      org.opencontainers.image.description="API de resume automatique Syntra" \
      org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.licenses="MIT"

# SYNTRA_DEVICE=cpu decrit ce que l'image contient : la roue torch installee est
# celle de l'index CPU. Laisser "auto" ferait chercher un CUDA absent.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/var/cache/huggingface \
    SYNTRA_DEVICE=cpu \
    SYNTRA_REGISTRY_ROOT=/srv/registry

# Utilisateur dedie, sans shell et sans home. L'identifiant fixe 10001 permet de
# donner les droits d'un volume monte sans avoir a deviner l'uid.
#
# Le code appartient a root et l'application tourne sous syntra : elle lit ce
# qu'elle execute et ne peut pas le reecrire. Seuls le cache du tokenizer et le
# repertoire de travail lui sont ouverts.
RUN groupadd --system --gid 10001 syntra \
 && useradd --system --uid 10001 --gid 10001 --no-create-home \
      --shell /usr/sbin/nologin syntra \
 && install -d -o syntra -g syntra -m 0755 "${HF_HOME}" \
 && install -d -o root -g root -m 0755 /srv/registry /app

# Le seul artefact repris de l'etape de construction. Le code du projet y est
# deja installe : rien d'autre n'est copie, donc rien d'autre ne peut fuir.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER syntra:syntra

EXPOSE 8000

# Sonde de vivacite, pas de disponibilite. /health repond des que le processus
# tient debout ; un modele qui n'a pas charge se lit sur /ready et ne doit pas
# faire tuer le conteneur. Le prefixe suit le reglage, comme les routes.
# start-period couvre le chargement des poids au demarrage.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:8000'+os.environ.get('SYNTRA_API_V1_PREFIX','/api/v1')+'/health',timeout=4).read()"]

# --no-access-log : l'application journalise elle-meme chaque requete, sans la
# query string, que le log d'acces d'uvicorn ecrirait en clair sur disque.
# Un seul worker : chaque worker chargerait sa propre copie des poids. On monte
# en charge avec des repliques, pas avec des workers.
CMD ["uvicorn", "backend.app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--no-access-log"]
