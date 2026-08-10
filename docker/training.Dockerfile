# Image d'entrainement. Section 28 du cahier des charges.
#
# Elle porte l'extra train : donnees, entrainement, evaluation, tracking. Elle
# est plus lourde que l'image de service, et c'est voulu : les deux images sont
# separees precisement pour que le backend n'embarque pas datasets, mlflow,
# pandas et matplotlib.
#
# Aucun entrainement ne demarre tout seul. La commande par defaut affiche
# l'aide du lanceur : section 29 interdit qu'un docker compose up lance un
# entrainement lourd, et une image dont le CMD entraine rend cette regle
# dependante du fichier compose.
#
# Construction :
#   docker build -f docker/training.Dockerfile -t syntra-training:dev .
#   docker build -f docker/training.Dockerfile \
#     --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu130 \
#     -t syntra-training:cu130 .

ARG PYTHON_TAG=3.12-slim-bookworm

# ---------------------------------------------------------------------------
# Etape 1 : construction de l'environnement
# ---------------------------------------------------------------------------
FROM python:${PYTHON_TAG} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PATH="/opt/venv/bin:${PATH}"

# CPU par defaut. Une image GPU se construit avec l'index cu130, la meme roue
# que celle installee par make install sur le poste de developpement. Le
# conteneur devra alors tourner avec le runtime NVIDIA.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ARG PIP_VERSION=24.3.1
# Voir backend/Dockerfile : la serie 75 est sous PYSEC-2025-49 et PYSEC-2026-3447.
ARG SETUPTOOLS_VERSION=84.0.0
ARG WHEEL_VERSION=0.45.1

WORKDIR /build

RUN python -m venv /opt/venv \
 && pip install --upgrade \
      "pip==${PIP_VERSION}" \
      "setuptools==${SETUPTOOLS_VERSION}" \
      "wheel==${WHEEL_VERSION}"

COPY pyproject.toml README.md ./
RUN mkdir -p src backend \
 && touch src/__init__.py backend/__init__.py \
 && pip install "torch>=2.13,<3.0" --index-url "${TORCH_INDEX}" \
 && pip install ".[train]"

COPY src ./src
COPY backend ./backend
RUN pip install --no-deps . \
 && find /opt/venv -type d -name '__pycache__' -prune -exec rm -rf {} + \
 && find /opt/venv -type f -name '*.pyc' -delete

# ---------------------------------------------------------------------------
# Etape 2 : image d'execution
# ---------------------------------------------------------------------------
FROM python:${PYTHON_TAG} AS runtime

ARG BUILD_VERSION=0.1.0
ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="syntra-training" \
      org.opencontainers.image.description="Chaine d'entrainement et d'evaluation Syntra" \
      org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/var/cache/huggingface

# git est le seul paquet systeme installe. src/tracking/provenance.py enregistre
# la revision de chaque run, et sans la commande il enregistre None : une
# experience tracee sans son commit ne se rejoue pas, ce que la section 14
# interdit. Le nettoyage des listes apt tient dans la meme instruction, sinon le
# layer garde le cache.
RUN apt-get update \
 && apt-get install --no-install-recommends -y git=1:2.39.* \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 10001 syntra \
 && useradd --system --uid 10001 --gid 10001 --no-create-home \
      --shell /usr/sbin/nologin syntra \
 && install -d -o syntra -g syntra -m 0755 "${HF_HOME}" \
 && install -d -o syntra -g syntra -m 0755 /app /app/data /app/reports /app/artifacts

COPY --from=builder /opt/venv /opt/venv

# Les configurations sont des donnees, pas du code installe : elles sont lues
# par chemin relatif depuis le repertoire de travail. Le corpus, les rapports et
# le registre sont montes, jamais copies.
WORKDIR /app
COPY configs ./configs

USER syntra:syntra

# Aucun HEALTHCHECK : un travail par lots n'a pas d'etat "en bonne sante", il a
# un code de sortie. Une sonde qui repondrait toujours OK vaudrait moins que
# rien. Le conteneur est surveille par son code de retour et par ses logs.

ENTRYPOINT ["python"]
CMD ["-m", "src.experiments.run", "--help"]
