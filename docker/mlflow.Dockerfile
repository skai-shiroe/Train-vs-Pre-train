# Serveur de tracking MLflow de la stack locale. Section 29.
#
# L'image officielle de MLflow ne porte ni le pilote PostgreSQL ni le client S3.
# Les installer au demarrage du conteneur donnerait une stack qui depend du
# reseau a chaque up et dont personne ne connait les versions. Elles sont donc
# installees ici, une fois, epinglees.
#
# Une seule etape : rien n'est compile, il n'y a donc aucune toolchain a laisser
# derriere soi. La regle multi-etapes de la section 28.1 porte sur les images
# qui construisent quelque chose.
#
# Construction :
#   docker build -f docker/mlflow.Dockerfile -t syntra-mlflow:dev .

ARG PYTHON_TAG=3.12-slim-bookworm

FROM python:${PYTHON_TAG} AS runtime

# La version du serveur suit celle du client declare par l'extra train de
# pyproject.toml. Un serveur en retard sur son client refuse des appels que le
# lanceur croit valides.
ARG MLFLOW_VERSION=3.15.1
ARG PSYCOPG_VERSION=2.9.10

LABEL org.opencontainers.image.title="syntra-mlflow" \
      org.opencontainers.image.description="Serveur de tracking MLflow de la stack Syntra" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# boto3 porte une borne majeure et non une version exacte : AWS publie un client
# par jour ouvre, l'epingler a l'exact rendrait l'image perimee en une semaine
# sans rien garantir de plus sur le protocole S3.
RUN pip install --upgrade "pip==24.3.1" \
 && pip install \
      "mlflow==${MLFLOW_VERSION}" \
      "psycopg2-binary==${PSYCOPG_VERSION}" \
      "boto3>=1.35,<2.0" \
 && groupadd --system --gid 10002 mlflow \
 && useradd --system --uid 10002 --gid 10002 --no-create-home \
      --shell /usr/sbin/nologin mlflow \
 && install -d -o mlflow -g mlflow -m 0755 /home/mlflow

WORKDIR /home/mlflow
USER mlflow:mlflow

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request;urllib.request.urlopen('http://127.0.0.1:5000/health',timeout=4).read()"]

ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
