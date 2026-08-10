# Déploiement

Le projet produit trois images : le service, la chaîne d'entraînement, et le serveur de tracking de la stack locale. Elles sont décrites par les sections 28 et 29 du cahier des charges.

## État de la vérification

Aucune image n'a été construite à ce jour. Le poste de développement ne dispose pas de Docker.

| Élément | État |
| --- | --- |
| Fichiers de construction et stack locale | Écrits |
| Invariants de la section 28 et de la section 29 | Vérifiés par `tests/unit/test_container_config.py` |
| Commande du `HEALTHCHECK` du backend | Exécutée hors conteneur, contre un uvicorn réel |
| Construction des images | Non exécutée |
| Taille de l'image backend | Non mesurée |
| Scan Trivy | Non exécuté |

La taille de l'image et le résultat du scan seront relevés au premier build, avec les commandes de la section [Mesurer la taille](#mesurer-la-taille) et de la section [Scan de sécurité](#scan-de-securite). Une taille annoncée sans avoir été mesurée serait une valeur inventée, ce que la section 44 traite comme un résultat fabriqué.

## Les trois images

| Image | Fichier | Contenu |
| --- | --- | --- |
| `syntra-backend` | `backend/Dockerfile` | Extra `api` : FastAPI, torch CPU, transformers, ROUGE |
| `syntra-training` | `docker/training.Dockerfile` | Extra `train` : datasets, MLflow, pandas, matplotlib, plus `git` |
| `syntra-mlflow` | `docker/mlflow.Dockerfile` | Serveur MLflow, pilote PostgreSQL, client S3 |

Le backend et l'entraînement sont séparés pour une seule raison : l'image de service ne doit pas embarquer l'outillage de recherche. C'est aussi ce que traduisent les deux extras de `pyproject.toml`.

### Ce que le backend ne contient pas

```text
les poids, qui viennent du Model Registry monté au démarrage
le corpus, qui n'a rien à faire dans une image de service
les tests, la documentation et les fichiers de développement, écartés par .dockerignore
toute valeur secrète : la configuration entre par l'environnement, jamais par un layer
```

Les poids sont montés, pas copiés. Une image qui embarquerait un checkpoint changerait de contenu à chaque promotion de modèle, et pèserait le poids du registre.

### Construction

```bash
make docker-build
make docker-build-training
```

soit :

```bash
docker build -f backend/Dockerfile --build-arg VCS_REF=$(git rev-parse --short HEAD) \
  -t syntra-backend:dev .
```

La construction est en deux étapes. La première installe les dépendances dans un environnement virtuel isolé, la seconde ne reçoit que cet environnement. L'image finale ne contient donc aucune toolchain, aucun cache `pip`, et rien du contexte de build en dehors de ce qui a été installé.

L'ordre des couches suit le cahier des charges : `pyproject.toml` d'abord, le code source ensuite. Un changement dans le code ne réinstalle pas les dépendances.

### Les roues torch sont celles du CPU

Les deux images installent torch depuis `https://download.pytorch.org/whl/cpu`, et l'image backend fixe `SYNTRA_DEVICE=cpu`.

L'index CUDA de `make install` tirerait plusieurs gigaoctets de bibliothèques NVIDIA dans une image qui, sans runtime NVIDIA et sans image de base CUDA, ne les utiliserait jamais. Servir sur GPU est un autre déploiement, pas une autre roue.

L'entraînement, lui, se construit avec l'index CUDA quand la machine cible a un GPU :

```bash
docker build -f docker/training.Dockerfile \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu130 \
  -t syntra-training:cu130 .
```

### Utilisateur non-root

Les trois images tournent sous un compte de service dédié, sans shell et sans home.

Le code appartient à `root` et l'application tourne sous `syntra` : elle lit ce qu'elle exécute et ne peut pas le réécrire. Seuls le cache du tokenizer et son répertoire de travail lui sont ouverts.

L'identifiant est fixé à 10001 dans les deux images Python. Le cache du tokenizer est un volume partagé par le service et l'entraînement : deux identifiants différents le rendraient illisible pour l'un des deux conteneurs.

### Sondes

Le backend déclare un `HEALTHCHECK` sur `/api/v1/health`, avec un `start-period` de 60 secondes qui couvre le chargement des poids.

La sonde interroge `/health` et non `/ready`, et la distinction n'est pas cosmétique. `/health` répond dès que le processus tient debout. `/ready` répond pour les modèles, et un checkpoint corrompu s'y rapporte. Sonder `/ready` ferait redémarrer en boucle une instance parfaitement vivante dont un seul modèle a échoué.

Le préfixe suit le réglage `SYNTRA_API_V1_PREFIX`, comme les routes. Une instance déployée derrière un autre préfixe reste sondée au bon endroit.

La commande de la sonde a été exécutée telle qu'elle est écrite dans le `Dockerfile`, hors conteneur, contre une instance servie par uvicorn avec les options de l'image. Elle sort en 0 sous le préfixe par défaut, en 0 sous un préfixe modifié, et en 1 contre un port fermé. Une sonde qui ne sait pas échouer ne vérifie rien.

L'image d'entraînement ne déclare aucune sonde. Un travail par lots n'a pas d'état de santé, il a un code de sortie : une sonde qui répondrait toujours `OK` vaudrait moins que rien.

### Le service ne journalise pas les URL complètes

La commande par défaut passe `--no-access-log` à uvicorn. Le log d'accès écrit l'URL complète, query string comprise ; un client qui passerait un document en paramètre de requête le verrait arriver sur disque. L'application journalise elle-même chaque requête, sans la query string. La même règle vaut pour `make api`.

Un seul worker. Chaque worker chargerait sa propre copie des poids : on monte en charge avec des répliques, pas avec des workers.

### Mesurer la taille

```bash
make docker-size
```

soit :

```bash
docker image inspect syntra-backend:dev --format '{{.Size}}'
```

Le chiffre relevé sera reporté dans le tableau d'état de cette page. L'ordre de grandeur attendu est dominé par torch CPU et transformers, pas par le code du projet.

## Scan de sécurité {#scan-de-securite}

Trivy scanne l'image construite, après le build et avant tout déploiement.

```bash
make docker-scan
```

soit :

```bash
trivy image --exit-code 1 --severity CRITICAL --ignore-unfixed --format table syntra-backend:dev
trivy image --format json --output reports/trivy.json syntra-backend:dev
```

Règles :

```text
scan systématique de l'image construite
le job échoue si une vulnérabilité CRITICAL est détectée
les niveaux HIGH sont rapportés mais ne bloquent pas par défaut
le rapport JSON est publié en artefact
```

Toute exception vit dans `.trivyignore`, avec la référence CVE et la raison. Le fichier ne contient aujourd'hui aucune exception active. Le scan n'est jamais neutralisé globalement pour faire passer un pipeline.

## Stack locale

```bash
cp .env.example .env
make docker-up
```

`compose.yaml` déclare les quatre services de la section 29, plus deux services de service.

| Service | Rôle | Port hôte |
| --- | --- | --- |
| `backend` | API de résumé | 8000 |
| `mlflow` | Serveur de tracking, artefacts servis par proxy | 5000 |
| `postgres` | Magasin de métadonnées de MLflow | aucun |
| `minio` | Stockage objet des artefacts | 9000, console 9001 |
| `minio-init` | Crée le bucket une fois, puis sort | aucun |
| `training` | Chaîne d'entraînement, derrière le profil `training` | aucun |

Les ports sont paramétrables : `BACKEND_PORT`, `MLFLOW_PORT`, `MINIO_PORT` et `MINIO_CONSOLE_PORT`. Un serveur MLflow lancé hors conteneur occupe déjà le port 5000 sur le poste de développement.

PostgreSQL ne publie aucun port. Seul MLflow lui parle, et il est sur le réseau du projet ; publier 5432 exposerait la base à tout le poste.

### Aucun secret dans le fichier

Les identifiants sont interpolés depuis `.env`, copie de `.env.example`. Une variable absente arrête `docker compose` avec le nom de la variable manquante, plutôt que de démarrer sur un mot de passe par défaut.

Les artefacts MLflow passent par le serveur, qui est lancé avec `--serve-artifacts`. Sans cette option, chaque client aurait besoin des identifiants MinIO, donc du secret, pour télécharger une figure.

### Le registre est monté en lecture seule

```yaml
- ./artifacts/registry:/srv/registry:ro
```

L'instance sert des versions, elle n'en publie pas. La publication est le travail de `python -m src.experiments.publish`, hors conteneur de service.

Le répertoire est vide tant qu'aucun run n'a été publié. Dans cet état, l'API démarre, `/health` répond 200, `/models` renvoie une liste vide et `/ready` répond 503. C'est l'état actuel du projet.

### Le backend ne dépend pas de MLflow

Le service `backend` ne déclare aucun `depends_on`. Le registre par défaut est local : une API qui attendrait une base dont elle n'a pas besoin ne démarrerait pas sur un poste où l'on ne lance que l'API.

### L'entraînement ne démarre jamais tout seul

`docker compose up` ne lance aucun entraînement. La règle tient à deux niveaux, parce qu'un seul serait un accident en attente : le service vit derrière un profil, et l'image, lancée sans compose, affiche l'aide du lanceur au lieu d'entraîner.

```bash
make docker-train CONFIG=configs/experiments/scratch_10.yaml
```

soit :

```bash
docker compose --profile training run --rm training \
  -m src.experiments.run --config configs/experiments/scratch_10.yaml
```

Le service d'entraînement ne redémarre pas après un échec. Un entraînement qui échoue doit rester en échec : le relancer en boucle brûlerait un GPU sur une erreur de configuration.

`git` est le seul paquet système installé dans cette image. `src/tracking/provenance.py` enregistre la révision de chaque run, et sans la commande il enregistre `None` : une expérience tracée sans son commit ne se rejoue pas.

## Ce qui est vérifié sans Docker

`tests/unit/test_container_config.py` lit les fichiers de construction et le fichier compose, et échoue si un invariant disparaît :

```text
la construction reste en plusieurs étapes, chaque étape nommée
aucune image de base ne suit latest
la dernière instruction USER n'est pas root
aucun COPY ni ADD ne prend le contexte entier
aucun ENV ni ARG ne porte une valeur ressemblant à un identifiant
apt nettoie ses listes dans la même instruction que l'installation
le backend déclare une sonde, sur /health et pas sur /ready
la commande de service garde --no-access-log
l'image d'entraînement n'entraîne pas par défaut
les deux images Python partagent le même uid
seul le service training vit derrière un profil
aucun service ne porte un identifiant en dur
le registre est monté en lecture seule
```

Un dernier test vérifie que l'extra `api` déclare tout ce que le backend importe. C'est la panne que rien d'autre n'attrape : un module qui importerait `mlflow` ou `pandas` passerait toute la suite sur le poste de développement, où l'extra `train` est installé, et échouerait au premier démarrage du conteneur. Ce test a trouvé un import de `starlette` que `pyproject.toml` ne déclarait pas.

## Écarts assumés

| Écart | Raison |
| --- | --- |
| `minio/minio` et `minio/mc` suivent `latest` par défaut | MinIO ne publie pas de tag de version stable, seulement des tags de date. `MINIO_IMAGE` et `MC_IMAGE` doivent porter un tag `RELEASE.*` avant tout déploiement |
| Les dépendances applicatives portent une borne majeure, pas une version exacte | Un verrou à l'exact demande un fichier de résolution produit sous Linux, que le poste de développement ne peut pas générer fidèlement. `PIP_CONSTRAINT` reste le point d'entrée d'un tel fichier |
| Aucune image n'a été construite | Docker n'est pas disponible sur le poste. La CI de la section 30 sera le premier build réel |

## Déploiement conteneurisé

Le déploiement de la section 33.2, `deploy/`, `make deploy` et les smoke tests contre l'environnement déployé relèvent de l'étape 17 et ne sont pas encore écrits. `tests/smoke/` existe déjà et attend une `BASE_URL`.
