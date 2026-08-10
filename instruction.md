# PROMPT MAÎTRE - BACKEND NLP + MLOPS + DOCKER + GITLAB

## 1. Rôle

Agis comme une équipe senior composée de :

* Machine Learning Engineer spécialisé en NLP ;
* Research Engineer ;
* Backend Engineer Python ;
* MLOps Engineer ;
* DevOps Engineer GitLab CI/CD ;
* Software Architect ;
* QA Engineer ;
* Security Engineer ;
* Technical Writer.

Ta mission est de concevoir et implémenter **uniquement le backend et la chaîne ML/MLOps** d'un projet académique de niveau Master portant sur la traduction et/ou le résumé automatique.

Le frontend sera développé ultérieurement.

Ne crée donc aucune interface utilisateur graphique à cette étape.

Le backend doit toutefois être conçu pour permettre à un futur frontend de :

* soumettre un texte ;
* choisir la tâche ;
* lancer une inférence ;
* comparer deux modèles ;
* récupérer les métriques ;
* consulter les modèles disponibles ;
* consulter les résultats expérimentaux.

## 1.1 Règle de rédaction transverse

Cette règle s'applique à **tout** ce qui est produit : documentation, README, docstrings, commentaires, messages de commit, messages d'erreur, descriptions de jobs CI et texte de réponse.

```text
ne jamais utiliser de tiret cadratin
ne jamais utiliser de tiret demi-cadratin
```

Remplacer par un deux-points, une virgule, une parenthèse, ou couper en deux phrases.

Le trait d'union ordinaire reste autorisé dans les mots composés et les options de ligne de commande.

Cette contrainte est vérifiée mécaniquement par le linter de prose décrit en section 37.5. Elle n'est pas une préférence esthétique : elle est bloquante.

---

# 2. Objectif scientifique

Le projet doit démontrer une démarche de recherche appliquée rigoureuse.

Les exigences obligatoires sont :

* implémenter un Transformer encodeur-décodeur from scratch ou fortement simplifié ;
* entraîner ce modèle sur un corpus réel ;
* comparer ce modèle à un modèle pré-entraîné ;
* tester le modèle pré-entraîné en zero-shot ;
* fine-tuner le modèle pré-entraîné sur le même sous-ensemble ;
* utiliser exactement le même jeu de test ;
* réaliser une étude d'ablation sur la taille du corpus ;
* utiliser au minimum les proportions 10 %, 50 % et 100 % ;
* mesurer les performances avec BLEU pour la traduction ou ROUGE pour le résumé ;
* produire une analyse qualitative ;
* générer les données nécessaires à une courbe performance vs taille du corpus ;
* permettre de déterminer à partir de quelle taille de corpus le modèle from scratch devient compétitif, si cela se produit.

Ne jamais inventer de résultats.

## 2.1 Décisions verrouillées

Ces décisions sont prises une fois pour toutes. Tout le code, la configuration et la documentation s'y conforment.

```text
tâche              résumé automatique (summarization)
traduction         hors périmètre, aucun code ni métrique de traduction
métriques          ROUGE-1, ROUGE-2, ROUGE-L
métrique reportée  ROUGE-L F-mesure, utilisée pour la courbe performance vs taille du corpus
modèle scratch     Transformer encodeur-décodeur PyTorch, implémenté à la main
modèle pré-entraîné t5-small (Hugging Face), en zero-shot puis fine-tuné
tokenizer          tokenizer T5 partagé par les deux modèles
corpus             XSum (EdinburghNLP/xsum)
langue du corpus   anglais
matériel           GPU NVIDIA local, précision mixte activée
langue du code     anglais, docstrings comprises
langue des docs    français
```

SacreBLEU et BLEU sortent du périmètre : la section 4 les mentionne pour le cas traduction, qui n'est pas retenu.

Le tokenizer T5 est partagé par le modèle from scratch et le modèle pré-entraîné. Ce choix rend la comparaison propre : même vocabulaire, même segmentation, même jeu de test. Il est documenté dans `docs/ml/transformer.md`.

## 2.2 Corpus de travail et honnêteté du « 100 % »

Le corpus XSum complet compte environ 204 000 exemples d'entraînement. Entraîner un Transformer from scratch puis trois fine-tunings sur ce volume dépasse le budget d'un GPU portable de 8 Go.

Le projet définit donc un **corpus de travail**, tiré une seule fois avec une graine fixe :

```text
train        20 000 exemples échantillonnés dans le split train officiel
validation    1 000 exemples échantillonnés dans le split validation officiel
test          1 000 exemples échantillonnés dans le split test officiel
graine        42
```

Les proportions 10 %, 50 % et 100 % de la section 11 s'appliquent à ce corpus de travail, soit 2 000, 10 000 et 20 000 exemples.

Règles d'intégrité :

```text
le corpus de travail est figé par une graine et un checksum, versionnés
« 100 % » désigne toujours le corpus de travail, jamais XSum complet
cette convention est rappelée dans chaque tableau de résultats et sur chaque figure
les splits validation et test sont strictement identiques pour toutes les expériences
aucun exemple du split test n'apparaît dans un split d'entraînement
```

Annoncer un score en laissant croire qu'il porte sur XSum complet serait une donnée inventée au sens de la section 44.

---

# 3. Périmètre

Le projet actuel comprend uniquement :

```text
Backend API
Machine Learning
Training
Evaluation
Experiment Tracking
Model Registry
Data Pipeline
MLOps
Docker
GitLab CI/CD
Tests
Documentation
```

Le projet actuel ne comprend pas :

```text
Frontend React
Interface Web
Design UI
UX
Application mobile
```

Toute logique d'affichage doit être laissée au futur frontend.

---

# 4. Stack principale

Utiliser :

```text
Python 3.12+
PyTorch
FastAPI
Pydantic
Uvicorn
Hugging Face Transformers
SacreBLEU
rouge-score
MLflow
DVC si pertinent
PostgreSQL
MinIO si stockage objet nécessaire
Docker
Docker Compose
GitLab CI/CD
GitLab Pages
GitLab Container Registry
pytest
pytest-cov
black
flake8
ruff
mypy
bandit
trivy
pip-audit
MkDocs Material ou équivalent
```

Répartition des outils qualité :

```text
black       formatage automatique PEP8
flake8      conformité PEP8 et erreurs de style
ruff        linting rapide complémentaire
mypy        typage statique
bandit      SAST Python
trivy       scan de vulnérabilités des images Docker
pip-audit   vulnérabilités des dépendances
```

`black` et `flake8` doivent être configurés de façon cohérente entre eux, par exemple avec une longueur de ligne commune et l'ignore des règles conflictuelles (`E203`, `W503`).

Privilégier une architecture simple, robuste et reproductible.

Ne pas introduire de composants uniquement pour donner une apparence "enterprise".

---

# 5. Architecture globale

Construire une architecture similaire à :

```text
                   ┌───────────────────────┐
                   │     Future Frontend    │
                   └───────────┬───────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │    FastAPI API     │
                     └─────────┬─────────┘
                               │
          ┌────────────────────┴───────────────────┐
          │                                        │
          ▼                                        ▼
┌──────────────────────┐              ┌──────────────────────┐
│ Transformer Scratch  │              │ Pretrained Model     │
│ PyTorch              │              │ T5 / mBART           │
└──────────────────────┘              └──────────────────────┘
          │                                        │
          └────────────────────┬───────────────────┘
                               ▼
                       Evaluation Layer
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
             MLflow                         DVC
                │
        Model Registry / Artifacts
```

---

# 6. Structure du repository

Créer une structure proche de :

```text
.
├── .gitlab/
│   └── ci/
│       ├── common.yml            # anchors, images, cache, règles partagées
│       ├── pipeline-tests.yml    # lint, typecheck, tests, SAST Bandit
│       ├── pipeline-build.yml    # build Docker + scan Trivy
│       ├── pipeline-deploy.yml   # déploiement conteneurisé + smoke tests
│       ├── pipeline-ml.yml       # data, train, evaluate, ablation
│       └── pipeline-docs.yml     # génération et publication GitLab Pages
│
├── deploy/
│   ├── compose.prod.yaml
│   ├── env/
│   └── scripts/
│       ├── deploy.sh
│       └── rollback.sh
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/
│   │   ├── core/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── inference/
│   │   ├── registry/
│   │   ├── monitoring/
│   │   └── main.py
│   ├── tests/
│   └── Dockerfile
│
├── configs/
│   ├── data/
│   ├── model/
│   ├── training/
│   └── experiments/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── docs/
│   ├── architecture/
│   ├── api/
│   ├── ml/
│   ├── experiments/
│   ├── uml/
│   ├── styles/          # règles du linter de prose
│   ├── contributing.md
│   ├── glossary.md
│   └── requirements.txt
│
├── reports/
│   ├── figures/
│   └── results/
│
├── scripts/
│
├── src/
│   ├── data/
│   ├── models/
│   │   ├── scratch/
│   │   └── pretrained/
│   ├── training/
│   ├── evaluation/
│   ├── experiments/
│   ├── metrics/
│   └── utils/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   └── smoke/
│
├── .codespell-ignore
├── .dockerignore
├── .flake8
├── .gitignore
├── .gitlab-ci.yml
├── .markdownlint.yaml
├── .pre-commit-config.yaml
├── .secrets.baseline
├── .trivyignore
├── .vale.ini
├── .yamllint.yaml
├── compose.yaml
├── Makefile
├── mkdocs.yml
├── pyproject.toml
├── README.md
└── .env.example
```

Le dossier `docs/` doit être directement publiable : sa structure est celle du site généré, pas un simple dépôt de notes.

---

# 7. Transformer from scratch

Implémenter explicitement les composants essentiels du Transformer.

Structure recommandée :

```text
src/models/scratch/
├── embeddings.py
├── positional_encoding.py
├── attention.py
├── multi_head_attention.py
├── feed_forward.py
├── encoder_layer.py
├── decoder_layer.py
├── encoder.py
├── decoder.py
├── masks.py
├── transformer.py
└── generation.py
```

Implémenter au minimum :

```text
Token Embedding
Positional Encoding
Scaled Dot Product Attention
Multi Head Attention
Encoder Layer
Decoder Layer
Residual Connections
Layer Normalization
Feed Forward Network
Padding Mask
Causal Mask
Cross Attention
Encoder
Decoder
Output Projection
Autoregressive Generation
```

Ne pas utiliser directement :

```python
torch.nn.Transformer(...)
```

comme implémentation complète du modèle.

L'objectif académique est de démontrer la compréhension de l'architecture.

---

# 8. Formule d'attention

L'implémentation doit suivre explicitement :

```text
Attention(Q, K, V)
=
softmax(QKᵀ / sqrt(d_k))V
```

Ajouter des commentaires expliquant :

* Q ;
* K ;
* V ;
* d_k ;
* masques ;
* division par sqrt(d_k) ;
* concaténation des têtes.

---

# 9. Baseline pré-entraînée

Créer un module séparé :

```text
src/models/pretrained/
├── base.py
├── t5.py
├── mbart.py
└── factory.py
```

Selon la tâche retenue, utiliser un modèle comme :

```text
T5
mT5
mBART
MarianMT
```

Supporter :

```text
zero-shot
fine-tuning
inference
evaluation
```

---

# 10. Data Pipeline

Créer :

```text
src/data/
├── download.py
├── validate.py
├── preprocess.py
├── tokenize.py
├── split.py
├── statistics.py
└── dataset.py
```

Le pipeline doit être entièrement reproductible.

Inclure :

* téléchargement ;
* validation ;
* nettoyage ;
* normalisation ;
* tokenisation ;
* splitting ;
* statistiques ;
* checksums ;
* version du dataset.

Les jeux :

```text
train
validation
test
```

doivent rester strictement identiques pour toutes les comparaisons.

---

# 11. Ablation principale

Construire les expériences :

```text
10 %
50 %
100 %
```

du jeu d'entraînement.

Comparer :

```text
Scratch
Pretrained Zero-Shot
Pretrained Fine-Tuned
```

Exemple :

```text
scratch_10
scratch_50
scratch_100

pretrained_zero_shot

pretrained_ft_10
pretrained_ft_50
pretrained_ft_100
```

Le zero-shot ne dépend normalement pas de la taille du corpus d'entraînement.

Ne pas créer artificiellement trois expériences zero-shot identiques.

---

# 12. Deuxième ablation

Ajouter au moins une autre comparaison simple et pertinente.

Préférence :

```text
nombre de couches Transformer
```

ou :

```text
nombre de têtes d'attention
```

ou :

```text
dimension d_model
```

Limiter volontairement le nombre d'expériences pour rester compatible avec les ressources matérielles.

---

# 13. Configuration des expériences

Chaque expérience doit disposer d'un fichier de configuration.

Exemple :

```text
configs/experiments/
├── scratch_10.yaml
├── scratch_50.yaml
├── scratch_100.yaml
├── pretrained_ft_10.yaml
├── pretrained_ft_50.yaml
└── pretrained_ft_100.yaml
```

Exemple :

```yaml
experiment:
  name: scratch_10
  seed: 42

dataset:
  percentage: 10

model:
  type: scratch
  d_model: 256
  num_heads: 8
  encoder_layers: 4
  decoder_layers: 4
  d_ff: 1024
  dropout: 0.1

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.0001
```

---

# 14. Reproductibilité

Créer une fonction centralisée :

```python
set_seed(seed)
```

configurant :

```text
random
numpy
torch
torch.cuda
```

Les expériences doivent pouvoir être reproduites avec une commande unique.

Exemple :

```bash
python -m src.experiments.run \
  --config configs/experiments/scratch_10.yaml
```

---

# 15. Entraînement

Créer :

```text
src/training/
├── trainer.py
├── callbacks.py
├── checkpoint.py
├── optimizer.py
├── scheduler.py
└── early_stopping.py
```

L'entraînement doit gérer :

```text
train loss
validation loss
gradient clipping
checkpoint
early stopping
scheduler
mixed precision si GPU
resume training
```

---

# 16. Evaluation

Créer :

```text
src/evaluation/
├── evaluator.py
├── translation.py
├── summarization.py
└── qualitative.py
```

Pour traduction :

```text
BLEU
chrF si pertinent
```

Pour résumé :

```text
ROUGE-1
ROUGE-2
ROUGE-L
```

---

# 17. Résultats

Sauvegarder automatiquement les résultats sous :

```text
reports/results/
```

avec par exemple :

```text
experiments.csv
ablation_dataset_size.csv
ablation_architecture.csv
qualitative_examples.json
```

Ne jamais modifier manuellement ces fichiers pour améliorer un résultat.

---

# 18. Graphiques

Produire automatiquement :

```text
reports/figures/performance_vs_dataset_size.png
reports/figures/training_loss.png
reports/figures/validation_loss.png
reports/figures/model_comparison.png
```

Le graphique principal doit permettre de comparer :

```text
Scratch
vs
Fine-Tuned Pretrained
```

selon :

```text
10 %
50 %
100 %
```

---

# 19. MLflow

Tracer automatiquement :

```text
experiment_id
git_commit
seed
dataset_version
dataset_percentage
model
hyperparameters
training_duration
BLEU
ROUGE
checkpoint
hardware
```

Utiliser MLflow pour suivre les expériences.

---

# 20. Model Registry

Le backend ne doit pas dépendre de chemins de fichiers codés en dur.

Créer une abstraction de registre de modèles.

Exemple :

```python
class ModelRegistry:
    def get_model(self, model_name: str, version: str | None = None):
        ...
```

Prévoir :

```text
scratch
pretrained
champion
challenger
```

---

# 21. Backend FastAPI

Créer une API versionnée :

```text
/api/v1
```

Structure :

```text
backend/app/
├── api/
│   └── v1/
│       ├── health.py
│       ├── inference.py
│       ├── models.py
│       ├── experiments.py
│       └── metrics.py
│
├── core/
├── schemas/
├── services/
├── inference/
├── registry/
└── main.py
```

## 21.1 Conventions d'implémentation FastAPI

FastAPI est le seul framework web du projet. Aucun autre framework HTTP ne doit être introduit.

Les conventions suivantes sont obligatoires. Elles garantissent un backend asynchrone, typé, testable et documenté automatiquement.

### Séparation des responsabilités

```text
api/v1/      routeurs seulement : validation, appel du service, mise en forme de la réponse
services/    logique applicative et orchestration, sans aucun import de FastAPI
inference/   exécution des modèles, sans aucun import de FastAPI
registry/    accès au Model Registry
schemas/     modèles Pydantic d'entrée et de sortie
core/        configuration, journalisation, exceptions, dépendances partagées
main.py      création de l'application, lifespan, middlewares, montage des routeurs
```

Un routeur ne contient aucune logique métier ni aucun appel direct à PyTorch.

`services/`, `inference/` et `registry/` doivent rester importables et testables sans démarrer l'application.

Chaque module de `api/v1/` expose un `APIRouter` avec son `prefix` et ses `tags`. `main.py` se limite à l'assemblage.

### Typage et validation

```text
toute signature de fonction est annotée, mypy passe en mode strict sur backend/app
toute entrée et toute sortie d'endpoint est un modèle Pydantic v2, jamais un dict brut
chaque route déclare un response_model et un status_code explicites
les contraintes métier sont portées par le schéma : max_length du texte, langues, tâche
les valeurs fermées (task, model, language) sont des Enum, jamais des chaînes libres
les Enum de schéma sont la source unique consommée par la documentation générée
```

Exemple de contrat, aligné sur les sections 23 et 24 :

```python
class PredictRequest(BaseModel):
    """Requête d'inférence sur un modèle unique."""

    text: str = Field(min_length=1, max_length=5000)
    task: TaskEnum = TaskEnum.SUMMARIZATION
    model: ModelNameEnum
    language: LanguageEnum = LanguageEnum.EN
    max_summary_tokens: int | None = Field(default=None, ge=8, le=256)
```

### Asynchronisme

L'inférence PyTorch est bloquante. Elle ne doit jamais s'exécuter directement dans la boucle d'événements.

```text
async def      endpoints, accès réseau, accès base de données, appels MLflow
def            fonctions pures, transformations, calculs de métriques
threadpool     toute inférence de modèle et tout appel PyTorch bloquant
```

Déporter explicitement le travail bloquant :

```python
from anyio import to_thread

output = await to_thread.run_sync(inference_service.predict, request)
```

Un `time.sleep`, un `requests.get` ou un `model.generate` appelé directement dans une coroutine est un défaut bloquant.

Sérialiser les accès concurrents à un même modèle si le modèle n'est pas réentrant, et documenter le choix.

### Cycle de vie et injection de dépendances

Le chargement des modèles suit le flux de la section 27 et s'exécute dans un `lifespan`, jamais dans un événement `on_event` déprécié ni à chaque requête.

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Charge les modèles au démarrage et libère les ressources à l'arrêt."""
    app.state.registry = await build_registry(get_settings())
    yield
    await app.state.registry.close()
```

Règles :

```text
aucune instance globale de modèle créée à l'import d'un module
les services sont fournis par Depends, jamais par une variable globale mutable
get_settings est mis en cache avec lru_cache
les dépendances sont surchargeables par dependency_overrides dans les tests
l'échec de chargement d'un modèle au démarrage est journalisé et rend /ready non prêt
```

`/api/v1/health` est une sonde de vivacité : elle ne dépend d'aucun modèle ni d'aucune base.

`/api/v1/ready` est une sonde d'aptitude : elle vérifie que les modèles attendus sont chargés.

### Configuration

La configuration passe uniquement par `pydantic-settings`, dans `core/`.

```text
aucune lecture directe de os.environ hors de la classe Settings
aucune valeur d'infrastructure codée en dur, y compris les URL MLflow, PostgreSQL et MinIO
les secrets utilisent SecretStr et ne sont jamais journalisés
.env.example reste synchronisé avec Settings, contrôle mécanique décrit en section 38.4
CORS piloté par la configuration, jamais fixé à une liste d'origines en dur
```

### Erreurs

Les erreurs suivent le format unique de la section 26.

```text
une énumération unique de codes d'erreur, source de vérité pour le code et la documentation
une exception applicative de base, levée par les services, ignorante du HTTP
un exception_handler unique qui traduit cette exception en réponse JSON normalisée
un handler dédié pour RequestValidationError, retournant le code INVALID_INPUT
un handler global qui renvoie INFERENCE_FAILED sans jamais exposer de trace
HTTPException réservée aux cas triviaux, avec le même corps de réponse
```

Correspondance minimale :

```text
INVALID_INPUT       422
INVALID_TASK        422
INVALID_LANGUAGE    422
MODEL_NOT_FOUND     404
MODEL_NOT_READY     503
TIMEOUT             504
INFERENCE_FAILED    500
```

Aucune réponse d'erreur ne doit renvoyer un corps non conforme à ce format.

### Middlewares et observabilité

```text
identifiant de corrélation par requête, propagé dans les logs et renvoyé en en-tête
journalisation structurée en JSON : méthode, chemin, statut, durée, identifiant de requête
mesure de la latence d'inférence, exposée dans latency_ms des réponses predict et compare
timeout explicite sur l'inférence, traduit en code TIMEOUT
GZipMiddleware si les charges utiles le justifient
aucun contenu de texte utilisateur journalisé intégralement, seulement sa longueur
```

### Contrat OpenAPI

```text
title, version et description renseignés sur l'application
tags documentés et stables, un tag par domaine fonctionnel
operation_id explicite et stable sur chaque route, pour la génération de client
exemples de requête et de réponse fournis via json_schema_extra des schémas
réponses d'erreur déclarées dans responses, avec le modèle d'erreur normalisé
pagination par limit et offset, encapsulée dans un schéma de page réutilisable
```

La spécification est exportée par `scripts/export_openapi.py` et comparée en CI, comme décrit en section 38.4.

Toute évolution d'un schéma ou d'une route impose la régénération de l'OpenAPI versionné.

### Tests de l'API

```text
httpx avec ASGITransport, ou TestClient, sans serveur réel pour les tests unitaires
dependency_overrides pour injecter un registre et un service d'inférence de test
un test par code d'erreur de la section 26
un test vérifiant que la réponse d'erreur respecte le format normalisé
un test de contrat vérifiant que l'OpenAPI généré contient les endpoints obligatoires
les modèles réels ne sont jamais chargés dans les tests unitaires
```

### Interdictions

```text
charger un modèle dans le corps d'un endpoint
exécuter une inférence bloquante dans une coroutine sans threadpool
retourner un dict non typé depuis un endpoint
utiliser un état global mutable en dehors de app.state initialisé par le lifespan
définir allow_origins à l'étoile lorsque allow_credentials est activé
exposer les endpoints de documentation sans contrôle en environnement de production
servir l'application avec le serveur de développement en conteneur de production
```

---

# 22. Endpoints obligatoires

Implémenter au minimum :

```http
GET /api/v1/health
GET /api/v1/ready

GET /api/v1/models

POST /api/v1/predict

POST /api/v1/compare

GET /api/v1/experiments

GET /api/v1/experiments/{experiment_id}

GET /api/v1/metrics
```

---

# 23. Endpoint predict

Exemple :

```http
POST /api/v1/predict
```

Payload :

```json
{
  "text": "Texte source a resumer, plusieurs phrases.",
  "task": "summarization",
  "model": "scratch",
  "language": "en",
  "max_summary_tokens": 64
}
```

Réponse :

```json
{
  "model": "scratch",
  "model_version": "1",
  "task": "summarization",
  "output": "Le resume genere.",
  "input_tokens": 218,
  "output_tokens": 27,
  "latency_ms": 97
}
```

---

# 24. Endpoint compare

Créer :

```http
POST /api/v1/compare
```

Payload :

```json
{
  "text": "Texte source a resumer, plusieurs phrases.",
  "task": "summarization",
  "language": "en",
  "reference": "Resume de reference, optionnel."
}
```

Réponse :

```json
{
  "scratch": {
    "model_version": "1",
    "output": "Resume produit par le modele from scratch.",
    "latency_ms": 110,
    "metrics": { "rouge1": 0.31, "rouge2": 0.09, "rougeL": 0.24 }
  },
  "pretrained": {
    "model_version": "3",
    "output": "Resume produit par T5 fine-tune.",
    "latency_ms": 65,
    "metrics": { "rouge1": 0.42, "rouge2": 0.19, "rougeL": 0.35 }
  }
}
```

Le champ `reference` est optionnel. S'il est absent, le bloc `metrics` est omis plutôt que rempli de zéros : une métrique non calculable ne doit jamais ressembler à une métrique calculée.

Les valeurs de l'exemple ci-dessus sont illustratives et ne proviennent d'aucune expérience. Elles ne doivent jamais être recopiées dans un tableau de résultats.

---

# 25. Préparation du futur frontend

Même si aucun frontend n'est implémenté maintenant, prévoir :

```text
OpenAPI
Swagger
schemas propres
CORS configurable
erreurs standardisées
pagination
versionnement API
```

Le futur frontend doit pouvoir être développé sans modifier l'architecture ML interne.

---

# 26. Gestion des erreurs

Standardiser les erreurs.

Exemple :

```json
{
  "error": {
    "code": "MODEL_NOT_AVAILABLE",
    "message": "Le modèle demandé n'est pas disponible."
  }
}
```

Prévoir notamment :

```text
INVALID_INPUT
MODEL_NOT_FOUND
MODEL_NOT_READY
INFERENCE_FAILED
INVALID_LANGUAGE
INVALID_TASK
TIMEOUT
```

---

# 27. Chargement des modèles

Ne jamais charger les modèles à chaque requête.

Utiliser :

```text
Application startup
        ↓
Model Registry
        ↓
Model Loader
        ↓
Models cached in memory
        ↓
Inference
```

Supporter éventuellement le chargement lazy.

---

# 28. Docker

Créer au minimum :

```text
backend/Dockerfile
docker/training.Dockerfile
```

Respecter impérativement :

```text
multi-stage build
image de base légère (python:3.12-slim ou distroless)
utilisateur non-root dédié
.dockerignore complet
HEALTHCHECK
dépendances épinglées
aucun secret dans l'image ni dans les layers
```

## 28.1 Optimisation des images

Le Dockerfile doit être optimisé, pas seulement fonctionnel :

```text
étape builder séparée pour compiler et installer les dépendances
seuls les artefacts nécessaires sont copiés dans l'étape finale
aucune toolchain de build dans l'image finale
ordre des layers optimisé pour le cache (dépendances avant code source)
une seule couche par groupe d'opérations apt/pip
nettoyage des caches (pip, apt) dans la même instruction RUN
COPY sélectif plutôt que COPY . lorsque c'est possible
```

Structure attendue :

```dockerfile
FROM python:3.12-slim AS builder
# installation des dépendances dans un venv isolé

FROM python:3.12-slim AS runtime
# copie du venv, création de l'utilisateur non-root, HEALTHCHECK, CMD
USER appuser
```

L'image backend finale doit rester raisonnablement légère.

Documenter la taille obtenue dans `docs/deployment.md`.

Ne pas embarquer les checkpoints lourds dans l'image : ils sont récupérés depuis le Model Registry, MLflow ou MinIO au démarrage.

## 28.2 Scan de sécurité des images

Intégrer **Trivy** dans le pipeline, après le build de l'image.

Règles :

```text
scan systématique de l'image construite
le job doit échouer si une vulnérabilité CRITICAL est détectée
les niveaux HIGH sont rapportés mais ne bloquent pas par défaut
le rapport est publié en artefact
```

Exemple de commande :

```bash
trivy image \
  --exit-code 1 \
  --severity CRITICAL \
  --ignore-unfixed \
  --format table \
  "$IMAGE_TAG"
```

Toute exception doit être explicitement justifiée dans un fichier `.trivyignore` commenté, avec la référence CVE et la raison.

Ne jamais neutraliser globalement le scan pour faire passer un pipeline.

---

# 29. Docker Compose

Créer un `compose.yaml` avec :

```text
backend
mlflow
postgres
minio
```

Le service d'entraînement peut être lancé par profil ou à la demande.

Exemple :

```text
docker compose --profile training run training
```

Ne pas lancer automatiquement des entraînements lourds avec :

```text
docker compose up
```

---

# 30. GitLab CI/CD modulaire

Pour assurer la maintenabilité, le `.gitlab-ci.yml` racine doit rester **léger** : il ne contient que les stages, les variables globales et les `include`.

Aucun job ne doit être défini directement dans le fichier racine.

```yaml
stages:
  - lint
  - test
  - security
  - build
  - data
  - train
  - evaluate
  - ablation
  - deploy
  - docs

include:
  - local: .gitlab/ci/common.yml
  - local: .gitlab/ci/pipeline-tests.yml
  - local: .gitlab/ci/pipeline-build.yml
  - local: .gitlab/ci/pipeline-deploy.yml
  - local: .gitlab/ci/pipeline-ml.yml
  - local: .gitlab/ci/pipeline-docs.yml
```

Répartition des responsabilités :

```text
common.yml          images, cache pip, anchors YAML, règles réutilisables
pipeline-tests.yml  pre-commit, black --check, flake8, ruff, mypy, pytest, Bandit (SAST)
pipeline-build.yml  build de l'image Docker, push registry, scan Trivy
pipeline-deploy.yml déploiement de l'image puis smoke tests
pipeline-ml.yml     data, train, evaluate, ablation (manuels ou conditionnels)
pipeline-docs.yml   docs:lint, docs:sync-check, build MkDocs, GitLab Pages
```

Chaque fichier inclus doit être autonome et lisible seul.

Factoriser les éléments répétés via `common.yml` (anchors, `extends`, `.rules` partagées) plutôt que par copier-coller.

---

# 31. Pipeline

Créer les stages :

```text
validate
lint
test
security
build
data
train
evaluate
ablation
deploy
package
release
docs
```

Ordre logique du flux principal :

```text
lint  →  test  →  security (Bandit)  →  build  →  security image (Trivy)  →  deploy  →  smoke tests  →  docs
```

Le pipeline doit échouer immédiatement (`fail fast`) sur :

```text
hook pre-commit en échec sur le dépôt complet
formatage non conforme (black --check)
violation flake8
faute détectée par codespell
erreur bloquante du linter de prose
tiret cadratin ou demi-cadratin présent dans le dépôt
échec de test
couverture inférieure à 80 %
vulnérabilité CRITICAL détectée par Trivy
smoke test en échec après déploiement
documentation générée désynchronisée du code
lien mort détecté par mkdocs build --strict
```

---

# 32. Politique des entraînements

Ne jamais lancer systématiquement les entraînements lourds à chaque commit.

Les jobs :

```text
train:scratch
train:pretrained
ablation:data-size
ablation:architecture
```

doivent être :

```text
manual
scheduled
ou conditionnels
```

selon les changements.

Exemple :

```yaml
rules:
  - changes:
      - src/models/**/*
      - src/training/**/*
      - configs/experiments/**/*
```

---

# 33. GitLab artifacts

Publier automatiquement :

```text
reports/results/*.csv
reports/figures/*.png
reports/junit.xml
reports/coverage.xml
reports/bandit.json
reports/trivy.json
metrics.json
```

Les gros checkpoints doivent aller dans :

```text
MLflow
MinIO
DVC remote
```

et non directement dans Git.

---

# 33.1 Feedback continu dans les Merge Requests

Les résultats de la CI doivent être visibles **directement dans la Merge Request**, sans avoir à ouvrir les logs.

Configurer les rapports natifs GitLab :

```yaml
test:unit:
  stage: test
  script:
    - pytest --junitxml=reports/junit.xml
             --cov=src --cov=backend/app
             --cov-report=xml:reports/coverage.xml
             --cov-report=term
             --cov-fail-under=80
  coverage: '/^TOTAL\s+\d+\s+\d+\s+(\d+%)$/'
  artifacts:
    when: always
    paths:
      - reports/
    reports:
      junit: reports/junit.xml
      coverage_report:
        coverage_format: cobertura
        path: reports/coverage.xml
```

Exigences :

```text
JUnit      widget de tests dans la MR, échecs listés nommément
Cobertura  annotation de la couverture ligne par ligne dans le diff de la MR
coverage   pourcentage global affiché sur la MR et dans le badge du projet
artifacts  when: always pour publier les rapports même en cas d'échec
```

Ajouter également, si pertinent :

```text
rapport SAST GitLab pour Bandit
rapport Container Scanning GitLab pour Trivy
```

de manière à faire remonter les vulnérabilités dans l'onglet sécurité de la MR.

---

# 33.2 Déploiement

Aucun fournisseur cloud n'est utilisé. Le déploiement reste conteneurisé et portable.

La cible de déploiement est un hôte Docker, décrit dans `deploy/compose.prod.yaml`, qui consomme l'image taggée publiée sur le GitLab Container Registry.

Le job de déploiement (`.gitlab/ci/pipeline-deploy.yml`) doit :

```text
récupérer l'image taggée du registry, jamais reconstruire au déploiement
valider la configuration avec docker compose config
appliquer le déploiement via deploy/scripts/deploy.sh
attendre que le healthcheck du conteneur soit vert
enchaîner les smoke tests sur l'URL de l'environnement
```

Règles :

```text
le tag déployé est immuable : commit SHA, pas latest
aucun secret en clair : uniquement des variables CI protégées et masquées
le déploiement en production est manuel (when: manual)
le déploiement échoue explicitement plutôt que de laisser un état incohérent
un échec des smoke tests doit faire échouer le job de déploiement
un script de rollback vers le tag précédent doit exister et être testé
```

Utiliser les environnements GitLab pour tracer les déploiements :

```yaml
deploy:staging:
  stage: deploy
  environment:
    name: staging
    url: $STAGING_URL
```

Si aucun hôte de déploiement n'est disponible dans le contexte académique, le job doit :

```text
rester défini et fonctionnel
cibler un environnement éphémère lancé par docker compose dans le pipeline
exécuter réellement les smoke tests contre ce conteneur
ne jamais être simulé ni marqué artificiellement en succès
```

Ce mode dégradé est documenté explicitement dans `docs/deployment.md`, avec la raison.

---

# 34. Tests unitaires

Tester au minimum :

```text
attention
multi-head attention
masks
positional encoding
encoder
decoder
generation
metrics
tokenization
FastAPI schemas
```

---

# 35. Tests d'intégration

Tester :

```text
dataset -> tokenizer
tokenizer -> model
model -> generation
generation -> metric
registry -> model
model -> backend
```

---

# 36. Tests E2E backend

Créer au minimum :

```text
POST /compare
    ↓
Scratch inference
    ↓
Pretrained inference
    ↓
JSON response
```

Aucun navigateur n'est nécessaire à cette étape.

---

# 36.1 Smoke tests

Créer `tests/smoke/` : un jeu de tests fonctionnels **rapides** exécutés contre un environnement réellement déployé, et non contre un mock.

Doivent vérifier au minimum :

```text
GET  /api/v1/health   répond 200
GET  /api/v1/ready    répond 200 et les modèles sont chargés
GET  /api/v1/models   liste au moins un modèle
POST /api/v1/predict  retourne une sortie non vide et une latence mesurée
```

Contraintes :

```text
cible pilotée par une variable d'environnement (BASE_URL)
aucune dépendance à des données locales
exécution en moins d'une minute
marqués pytest -m smoke pour être exclus du run unitaire
```

Un échec de smoke test après déploiement doit faire échouer le pipeline.

---

# 37. Qualité et Shift Left

La qualité est validée **au plus tôt** : localement via `pre-commit`, puis rejouée à l'identique en CI. Les mêmes commandes doivent produire le même verdict dans les deux contextes.

Utiliser :

```text
black
flake8
ruff
mypy
pytest
pytest-cov
bandit
pip-audit
pre-commit
```

## 37.1 Formatage et style (PEP8)

```text
black         formatage imposé, vérifié en CI avec black --check --diff
flake8        conformité PEP8, complexité, imports inutilisés
ruff          linting complémentaire rapide
```

Le job de lint doit échouer si un fichier n'est pas formaté.

Ne jamais reformater automatiquement en CI : la CI constate, le développeur corrige.

## 37.2 Couverture

Seuil minimal **obligatoire de 80 %**, appliqué mécaniquement :

```bash
pytest --cov=src --cov=backend/app \
       --cov-report=xml:reports/coverage.xml \
       --cov-report=term-missing \
       --cov-fail-under=80
```

Le seuil est un plancher, pas un objectif : privilégier des tests pertinents sur l'attention, les masques, la génération, les métriques et les endpoints, plutôt que du remplissage pour atteindre le chiffre.

Ne pas exclure de modules de la couverture pour atteindre artificiellement le seuil. Toute exclusion dans la configuration doit être justifiée par un commentaire.

## 37.3 Sécurité statique (SAST)

Analyser tout le code Python avec **Bandit** :

```bash
bandit -r src backend/app -f json -o reports/bandit.json
bandit -r src backend/app -ll
```

Règles :

```text
le job échoue sur les findings de sévérité HIGH
les findings MEDIUM sont rapportés et doivent être traités ou justifiés
toute exception utilise # nosec avec un commentaire expliquant pourquoi
le rapport JSON est publié en artefact
```

Compléter avec `pip-audit` sur les dépendances.

## 37.4 Pre-commit hooks

Les hooks sont la première barrière du Shift Left : ce qui peut être détecté sur le poste du développeur ne doit pas consommer un runner CI.

Installer les hooks à l'installation du projet :

```bash
pre-commit install
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push
```

`make install` doit poser ces hooks automatiquement.

### Hooks au stage `pre-commit`

Rapides, exécutés sur les fichiers modifiés uniquement :

```text
black                     formatage Python
flake8                    conformité PEP8
ruff                      linting complémentaire
mypy                      typage du code applicatif
bandit                    SAST sur src et backend/app
codespell                 fautes d'orthographe dans le code et la documentation
markdownlint              qualité du Markdown
yamllint                  qualité des fichiers YAML, y compris .gitlab/ci
check-yaml                validité syntaxique YAML
check-json                validité syntaxique JSON
check-toml                validité syntaxique TOML
check-added-large-files   refus des fichiers volumineux, seuil explicite
check-merge-conflict      marqueurs de conflit oubliés
end-of-file-fixer         newline finale
trailing-whitespace       espaces en fin de ligne
detect-private-key        clés privées commitées
detect-secrets            secrets et credentials, avec baseline versionnée
nbstripout                sortie des notebooks si des notebooks existent
```

Un hook doit rester sous quelques secondes. Tout ce qui est lent va au stage `pre-push`.

### Hooks au stage `commit-msg`

Imposer un format de message de commit exploitable, par exemple Conventional Commits :

```text
feat, fix, docs, test, refactor, perf, build, ci, chore
```

Le hook rejette un message hors format. Le format retenu doit être documenté dans `docs/contributing.md`.

### Hooks au stage `pre-push`

Vérifications plus coûteuses, avant d'engager un runner :

```text
pytest -m "not slow and not smoke"
pytest --cov-fail-under=80 en mode rapide si le temps le permet
hadolint sur les Dockerfile
vérification de synchronisation de la documentation (voir section 38.4)
```

### Interdiction de contourner

```text
--no-verify n'est pas une pratique acceptée
la CI rejoue les mêmes hooks : contourner localement ne fait que déplacer l'échec
```

### Rejeu en CI

Un job dédié dans `pipeline-tests.yml` rejoue l'intégralité des hooks sur tout le dépôt :

```yaml
lint:pre-commit:
  stage: lint
  script:
    - pre-commit run --all-files --show-diff-on-failure
```

Cela garantit que les fichiers non touchés par le commit restent conformes et que les versions des hooks sont bien épinglées dans `.pre-commit-config.yaml`.

Les révisions des hooks doivent être épinglées, jamais laissées en branche flottante. Mettre à jour avec `pre-commit autoupdate` de façon délibérée et versionnée.

---

# 37.5 Contrôle orthographique et grammatical

Le projet est un livrable académique : la qualité rédactionnelle de la documentation, des docstrings et des messages d'erreur fait partie du rendu.

## Orthographe

Utiliser `codespell` sur l'ensemble du dépôt :

```bash
codespell src backend docs README.md \
  --skip="*.lock,*.svg,*.png,data/*,.git" \
  --ignore-words=.codespell-ignore
```

Les faux positifs, notamment le vocabulaire technique et les termes français dans un dépôt majoritairement anglophone, vont dans `.codespell-ignore`, jamais dans une désactivation globale.

## Grammaire et style

Contrôler la documentation en langue naturelle avec un linter de prose, `Vale` de préférence, ou LanguageTool.

```text
docs/**/*.md
README.md
CHANGELOG.md
```

Configurer `.vale.ini` avec :

```text
la langue de rédaction retenue pour le projet
un style de base, par exemple Google ou Microsoft, adapté au contexte académique
un vocabulaire projet, pour ne pas signaler Transformer, tokenizer, embedding, BLEU, ROUGE
un niveau d'alerte distinguant erreur bloquante et suggestion
```

Règles de sévérité :

```text
error        orthographe, accord, ponctuation cassée, terme interdit  → bloque le pipeline
warning      style, longueur de phrase, voix passive                 → rapporté sans bloquer
suggestion   préférences rédactionnelles                             → informatif
```

## Cohérence terminologique

Maintenir un glossaire dans `docs/glossary.md` et faire respecter la terminologie par le linter de prose.

Exemples de règles :

```text
un seul terme pour un concept : "modèle from scratch", pas d'alternance avec "modèle maison"
casse cohérente des noms propres : PyTorch, FastAPI, GitLab, MLflow, Hugging Face, Docker
pas d'abréviation non définie à sa première occurrence
```

## Conventions de rédaction

Ces règles s'appliquent à toute la documentation, aux commentaires et aux messages destinés à l'utilisateur :

```text
ne pas utiliser de tiret cadratin ni de tiret demi-cadratin
utiliser deux points, une virgule, une parenthèse ou une phrase séparée
phrases courtes, une idée par phrase
voix active
présent de l'indicatif pour décrire le comportement du système
pas de superlatif marketing ni de formulation promotionnelle
tout chiffre annoncé doit être vérifiable dans le dépôt
```

Ajouter une règle de linter de prose interdisant explicitement les caractères de tiret long, afin que la contrainte soit vérifiée mécaniquement et non par relecture.

## Intégration

```text
codespell et le linter de prose tournent en hook pre-commit
un job docs:lint dans pipeline-docs.yml les rejoue sur tout le dépôt
un échec de niveau error bloque la publication GitLab Pages
```

---

# 38. Documentation "As Code"

La documentation vit dans le dépôt, évolue avec le code et est **publiée automatiquement** : le dossier `/docs` doit générer un site statique via **GitLab Pages**.

Créer au minimum :

```text
README.md

docs/index.md
docs/architecture/backend.md
docs/architecture/mlops.md
docs/architecture/ci-cd.md
docs/ml/transformer.md
docs/ml/training.md
docs/ml/evaluation.md
docs/experiments/ablation.md
docs/api/openapi.md
docs/reproducibility.md
docs/deployment.md
docs/security.md
docs/testing.md
```

## 38.1 Génération du site

Utiliser MkDocs (thème Material) ou un générateur équivalent, configuré par `mkdocs.yml` à la racine.

Le site doit contenir :

```text
la documentation technique du backend
l'architecture du projet et les diagrammes UML
la description du pipeline ML
la description du pipeline CI/CD
la documentation d'API générée depuis OpenAPI
la procédure de reproduction des expériences
la politique de sécurité (Bandit, Trivy, gestion des exceptions)
```

## 38.2 Publication GitLab Pages

Le job de publication est défini dans `.gitlab/ci/pipeline-docs.yml` :

```yaml
pages:
  stage: docs
  script:
    - pip install -r docs/requirements.txt
    - mkdocs build --strict --site-dir public
  artifacts:
    paths:
      - public
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
```

Exigences :

```text
le répertoire de sortie doit s'appeler public
le build s'exécute en mode --strict : tout lien mort casse le job
la publication n'a lieu que sur la branche par défaut
les MR construisent le site sans le publier, pour valider le build
```

Automatiser au maximum : les diagrammes Mermaid, la spécification OpenAPI et les résultats d'expériences doivent être injectés dans le site plutôt que recopiés à la main.

Une documentation qui ne se construit plus est un échec de pipeline, pas un détail.

## 38.3 Documentation lint

Un job `docs:lint` dans `pipeline-docs.yml` valide la documentation avant sa construction :

```text
markdownlint          structure, niveaux de titres, listes, longueur de ligne
codespell             orthographe
linter de prose       grammaire, style, terminologie, interdiction des tirets longs
lychee ou équivalent  liens externes morts, en mode non bloquant si le réseau est instable
mkdocs build --strict liens internes morts, pages orphelines, navigation incohérente
```

## 38.4 Synchronisation documentation et code

Une documentation fausse est plus nuisible qu'une documentation absente. La synchronisation doit être vérifiée mécaniquement, pas par discipline.

### Sources générées, jamais recopiées

Les éléments suivants sont produits à partir du code et injectés dans le site à chaque build :

```text
spécification OpenAPI      exportée depuis l'application FastAPI
référence des endpoints    générée depuis OpenAPI
référence du code Python   générée depuis les docstrings, via mkdocstrings
configuration disponible   générée depuis les modèles Pydantic Settings
codes d'erreur             générés depuis l'énumération unique définie en section 26
tableaux de résultats      lus depuis reports/results/*.csv
figures                    lues depuis reports/figures/*.png
arborescence du dépôt      générée, non maintenue à la main
```

Aucun de ces contenus ne doit être maintenu manuellement en Markdown. Toute duplication manuelle finit par diverger.

### Détection de dérive

Le pipeline doit échouer quand la documentation ne reflète plus le code.

Mécanisme pour l'OpenAPI :

```bash
python -m scripts.export_openapi --output /tmp/openapi.json
diff <(jq -S . docs/api/openapi.json) <(jq -S . /tmp/openapi.json)
```

Le job échoue si le fichier versionné diffère de la spécification régénérée. Le développeur relance `make docs-sync` et commite le résultat.

Appliquer le même principe de régénération et comparaison à :

```text
la liste des endpoints documentés
la liste des variables d'environnement de .env.example face aux Settings Pydantic
la liste des cibles du Makefile documentées dans le README
la liste des expériences dans configs/experiments face à docs/experiments/ablation.md
```

### Règles de fraîcheur

```text
une modification d'un endpoint impose une mise à jour de la documentation d'API
une modification d'un schéma Pydantic impose la régénération de l'OpenAPI
une modification de l'architecture impose la mise à jour du diagramme correspondant
une nouvelle expérience impose une entrée dans docs/experiments
un ajout de dépendance impose une mention dans docs/deployment.md si elle est runtime
```

Un job `docs:sync-check` peut signaler les Merge Requests qui modifient `backend/app/api/` ou `src/models/` sans toucher `docs/`. Ce contrôle est un avertissement, pas un blocage automatique, afin de ne pas encourager les modifications cosmétiques de documentation pour faire passer un pipeline.

### Docstrings

```text
toute fonction publique, classe et module de src et backend/app porte une docstring
style unique sur tout le projet, par exemple Google ou NumPy
les docstrings des composants du Transformer expliquent la formule implémentée
un contrôle de couverture de docstrings, par exemple interrogate, avec un seuil explicite
la couverture de docstrings est rapportée en artefact
```

### Commande unique

```bash
make docs-sync
```

régénère l'ensemble des contenus dérivés, puis :

```bash
make docs
```

construit le site en mode strict. Les deux commandes doivent être exécutables hors ligne, une fois les dépendances installées.

---

# 39. UML

Créer avec Mermaid ou PlantUML :

```text
diagramme de composants
diagramme de séquence d'inférence
diagramme de déploiement
pipeline ML
pipeline CI/CD
```

Les diagrammes doivent être versionnés en texte (Mermaid de préférence) et rendus dans le site GitLab Pages, jamais stockés uniquement en image exportée.

---

# 40. Commandes

Créer un Makefile avec :

```bash
make install
make hooks
make format
make lint
make spell
make prose
make typecheck
make security
make test
make test-unit
make test-integration
make test-e2e
make test-smoke
make coverage

make data
make train-scratch
make train-pretrained
make evaluate
make ablation

make api
make docker-build
make docker-scan
make docker-up
make docker-down

make docs
make docs-lint
make docs-sync
make docs-serve
make deploy

make ci
make reproduce
```

Correspondance attendue :

```text
make hooks        pre-commit install pour les trois stages
make format       black + correction automatique
make lint         black --check, flake8, ruff, yamllint, markdownlint
make spell        codespell sur le code et la documentation
make prose        linter de prose sur docs et README
make typecheck    mypy
make security     bandit + pip-audit
make coverage     pytest avec --cov-fail-under=80 et rapports xml
make docker-scan  trivy image, échec si CRITICAL
make docs-sync    régénère OpenAPI, référence de code et contenus dérivés
make docs-lint    markdownlint, codespell, prose, liens morts
make docs         mkdocs build --strict
make ci           reproduit localement la séquence complète de la CI
```

`make hooks` est appelé par `make install` : un dépôt cloné doit être conforme dès le premier commit.

`make ci` doit permettre à un développeur de reproduire le verdict du pipeline **avant** de pousser.

---

# 41. Commande de reproduction

Créer une commande :

```bash
make reproduce
```

qui permet de reproduire :

```text
préparation des données
expériences principales
évaluation
ablation
agrégation des résultats
graphiques
```

Si certaines étapes nécessitent un GPU important, prévoir un mode :

```text
full
quick
```

Le mode `quick` sert uniquement à vérifier le pipeline.

Les résultats scientifiques doivent provenir du mode complet.

---

# 42. GitHub et GitLab

Le livrable académique impose un repository GitHub.

Utiliser donc la stratégie suivante :

```text
GitLab
    ↓
développement
CI/CD
MLOps
Docker Registry
experiments
    ↓
mirror
    ↓
GitHub
repository de soumission
```

Le GitHub final doit contenir tout le code source et la documentation nécessaires à la reproduction.

Aucun résultat essentiel ne doit dépendre exclusivement d'un accès privé GitLab.

---

# 43. Priorité d'implémentation

Respecter strictement cet ordre :

```text
1. Requirements
2. Architecture
3. Repository bootstrap
4. Qualité et tests (pre-commit, black, flake8, pytest, bandit)
5. Data pipeline
6. Transformer from scratch
7. Tests du Transformer
8. Training pipeline
9. Baseline pretrained
10. Evaluation
11. Ablation
12. MLflow
13. Model Registry
14. FastAPI
15. Docker et scan Trivy
16. GitLab CI/CD modulaire
17. Déploiement conteneurisé et smoke tests
18. Documentation et GitLab Pages
19. Validation finale
```

Ne pas commencer par Docker ou GitLab avant que le cœur ML soit correctement structuré.

L'outillage qualité (étape 4) est en revanche mis en place **dès le bootstrap** : c'est le principe du Shift Left, les tests et les linters accompagnent le code plutôt que de le suivre.

---

# 43.1 Correspondance avec les étapes du cahier des charges

```text
Étape 1  Développement et Qualité Code (Shift Left)
         → sections 34, 35, 36, 36.1, 37, 37.4, 37.5

Étape 2  Conteneurisation et Sécurité des Artefacts
         → sections 28, 28.1, 28.2

Étape 3  Déploiement conteneurisé
         → sections 29, 33.2

Étape 4  Pipeline CI/CD Modulaire
         → sections 30, 31, 32

Étape 5  Feedback Continu et Documentation As Code
         → sections 33, 33.1, 38, 38.1, 38.2, 38.3, 38.4

Transverse  Règle de rédaction, interdiction des tirets longs
         → sections 1.1, 37.5
```

Aucune de ces exigences n'est optionnelle : elles font partie de la Definition of Done.

---

# 44. Intégrité scientifique

Utiliser :

```text
NOT_RUN
```

pour une expérience non exécutée.

Utiliser :

```text
FAILED
```

pour une expérience ayant échoué.

Utiliser :

```text
MOCK
```

uniquement pour les tests techniques.

Ne jamais transformer une donnée fictive en résultat scientifique.

---

# 45. Definition of Done

Le backend est considéré terminé uniquement si :

* le Transformer from scratch est réellement implémenté ;
* les composants d'attention sont testés ;
* l'entraînement fonctionne ;
* le modèle pré-entraîné fonctionne en zero-shot ;
* le fine-tuning fonctionne ;
* les splits sont identiques ;
* les expériences 10 %, 50 % et 100 % sont reproductibles ;
* BLEU ou ROUGE est calculé ;
* l'ablation est automatisée ;
* les résultats sont sauvegardés ;
* les graphiques sont produits automatiquement ;
* MLflow trace les expériences ;
* les modèles sont versionnés ;
* FastAPI expose les modèles ;
* `/predict` fonctionne ;
* `/compare` fonctionne ;
* Docker fonctionne ;
* le Dockerfile est multi-stage, l'image est légère et tourne en non-root ;
* Trivy scanne l'image et le pipeline échoue sur une vulnérabilité CRITICAL ;
* Bandit s'exécute sur l'ensemble du code sans finding HIGH non justifié ;
* `black` et `flake8` passent sans erreur ;
* les hooks pre-commit sont installés, épinglés et rejoués intégralement en CI ;
* `codespell` et le linter de prose passent sans erreur bloquante ;
* aucun tiret cadratin ni demi-cadratin ne subsiste dans le dépôt ;
* la documentation générée est synchronisée avec le code, vérifié mécaniquement ;
* l'OpenAPI versionné est identique à celui régénéré depuis l'application ;
* la couverture de tests est supérieure ou égale à 80 %, vérifiée automatiquement ;
* les rapports JUnit et Cobertura remontent visuellement dans les Merge Requests ;
* le `.gitlab-ci.yml` est modulaire et ne contient aucun job en propre ;
* le déploiement conteneurisé aboutit et les smoke tests passent réellement ;
* le site de documentation se construit et est publié sur GitLab Pages ;
* les pipelines GitLab passent ;
* les tests passent ;
* aucune donnée scientifique n'est inventée ;
* la documentation permet à un autre développeur de reprendre le projet ;
* le backend est prêt à être consommé ultérieurement par un frontend.

---

# 46. Règle finale

Ne développe aucun frontend.

Lorsque le besoin d'interface apparaît, créer uniquement :

* les contrats API ;
* les schemas Pydantic ;
* les endpoints ;
* les exemples de payload ;
* la documentation OpenAPI.

Le frontend sera traité comme un projet ou un lot séparé.

Le résultat final attendu pour cette phase est donc :

**un backend NLP expérimental, reproductible, conteneurisé, exposé par API et industrialisé avec une chaîne MLOps et GitLab CI/CD.**
