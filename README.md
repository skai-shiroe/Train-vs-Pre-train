# Syntra

Backend NLP expérimental de résumé automatique. Le projet compare un Transformer encodeur-décodeur implémenté à la main en PyTorch à un modèle pré-entraîné T5, en zero-shot puis fine-tuné, sur un jeu de test strictement identique.

Projet académique de niveau Master. Le périmètre couvre le backend, la chaîne ML, MLOps, Docker et l'intégration continue GitHub Actions. Aucun frontend n'est développé dans ce lot.

## Démarrage rapide

```bash
git clone <url-du-depot> syntra && cd syntra
py -3.12 -m venv .venv && source .venv/Scripts/activate   # Linux, macOS : .venv/bin/activate
make install
make reproduce MODE=quick
```

Quatre commandes depuis un dépôt fraîchement cloné. La dernière enchaîne toute la chaîne scientifique : elle construit le corpus s'il manque, joue les neuf expériences plafonnées à deux pas d'optimisation, agrège les tableaux et trace les figures. Elle vérifie que la chaîne tourne sur ce poste ; elle ne produit aucun résultat, et le dit. Comptez une minute une fois le corpus construit, mesuré sur GPU RTX 5060 portable.

Le premier appel télécharge XSum et construit le corpus de travail, ce qui domine le temps total. Les appels suivants sautent cette étape : `make data` est idempotent, et la chaîne affiche le `dataset_version` du corpus qu'elle a lu.

```bash
make ci                     # la séquence de vérification complète du pipeline
make api                    # l'API sur http://127.0.0.1:8000, documentation sur /docs
make reproduce MODE=full    # la campagne réelle, plusieurs heures de GPU
```

Sous Windows, `make` s'appelle depuis Git Bash et non depuis PowerShell. La raison et les trois façons d'ouvrir un tel terminal sont dans [Prérequis](#prérequis).

## Décisions verrouillées

| Sujet | Choix |
| --- | --- |
| Tâche | Résumé automatique |
| Métriques | ROUGE-1, ROUGE-2, ROUGE-L |
| Modèle from scratch | Transformer encodeur-décodeur PyTorch |
| Modèle pré-entraîné | `t5-small` |
| Tokenizer | Tokenizer T5, partagé par les deux modèles |
| Corpus | XSum, sous-ensemble figé de 20 000 exemples d'entraînement |
| Matériel | GPU NVIDIA local |

## Prérequis

```text
Python 3.12
GPU NVIDIA avec pilote récent, optionnel mais recommandé
Docker et Docker Compose, pour la stack locale
GNU Make
```

Sous Windows, `make` n'est pas fourni par le système :

```powershell
winget install ezwinports.make
```

Le PATH n'est mis à jour qu'au shell suivant.

Il faut ensuite **appeler `make` depuis Git Bash, pas depuis PowerShell**. Les recettes de ce `Makefile` sont écrites pour un shell POSIX : `command -v`, `test -n`, `awk`, `rm -rf`, groupes d'accolades. GNU Make ne retient un shell POSIX que s'il trouve `sh.exe` dans le PATH, sinon il retombe sur `cmd.exe`. Or l'installation courante de Git pour Windows ne publie que `C:\Program Files\Git\cmd`, où `sh.exe` ne se trouve pas : il vit dans `C:\Program Files\Git\usr\bin`. Depuis PowerShell, la première cible venue échoue donc ainsi :

```text
'grep' n'est pas reconnu en tant que commande interne ou externe
make: *** [Makefile:61: help] Error 255
```

Depuis Git Bash, la même cible passe, et le `Makefile` retrouve seul l'interpréteur du `.venv`. Trois façons d'y arriver :

```text
VS Code       menu déroulant du terminal, profil Git Bash
un terminal   & "C:\Program Files\Git\bin\bash.exe" -l
l'explorateur clic droit sur le dossier, Afficher plus d'options, Git Bash Here
```

Un terminal hérite du PATH du processus qui l'a lancé. VS Code transmet celui qu'il portait à son propre démarrage, et « Recharger la fenêtre » ne le rafraîchit pas : après l'installation, il faut quitter VS Code puis le rouvrir, sans quoi `make: command not found` persiste dans un terminal pourtant neuf. Pour débloquer la session en cours sans rien fermer :

```bash
export PATH="$PATH:$HOME/AppData/Local/Microsoft/WinGet/Packages/ezwinports.make_Microsoft.Winget.Source_8wekyb3d8bbwe/bin"
```

Publier `C:\Program Files\Git\usr\bin` dans le PATH lève aussi la limite, au prix d'exposer les outils Unix de Git à tout le système.

> Ne pas traiter ce point en fixant `SHELL` dans le `Makefile`. Les jobs GitHub Actions appellent les mêmes cibles sur des runners Linux : un chemin Windows codé en dur y casserait le pipeline pour arranger un poste.

## Installation

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
make install
```

`make install` installe PyTorch depuis l'index CUDA 12.8, puis les dépendances, puis pose les hooks `pre-commit`, `commit-msg` et `pre-push`.

> Les cartes RTX 50 (architecture Blackwell, `sm_120`) exigent les roues PyTorch CUDA 12.8. L'index PyPI par défaut ne convient pas.

Vérifier l'installation :

```bash
make ci
```

## Structure

```text
backend/app/     application FastAPI : api, core, schemas, services, inference, registry
src/             code de recherche : data, models, training, evaluation, experiments, metrics, tracking, utils
configs/         configuration des données, des modèles, de l'entraînement et des expériences
data/            corpus, versionné par DVC et non par git
reports/         résultats et figures produits automatiquement
docs/            documentation publiée sur GitHub Pages, rapport compris
notebooks/       vérification de l'environnement et analyse exploratoire du corpus
scripts/         outils de contrôle et d'export
tests/           tests unitaires, d'intégration, de bout en bout et smoke
deploy/          déploiement conteneurisé
.github/         pipeline modulaire, un workflow par étape
```

## Commandes

### Installation et hooks

| Cible | Effet |
| --- | --- |
| `make install` | Installe les dépendances et pose les hooks |
| `make hooks` | Pose les hooks `pre-commit`, `commit-msg` et `pre-push` |

### Qualité

| Cible | Effet |
| --- | --- |
| `make format` | Formate avec `black` et applique les corrections de `ruff` |
| `make lint` | `black --check`, `flake8`, `ruff`, `yamllint`, contrôle des tirets longs |
| `make spell` | Orthographe avec `codespell` |
| `make prose` | Grammaire, style et terminologie avec Vale |
| `make typecheck` | Typage statique avec `mypy` |
| `make security` | `bandit` et `pip-audit` |

### Tests

| Cible | Effet |
| --- | --- |
| `make test` | Tous les tests sauf les smoke tests |
| `make test-unit` | Tests unitaires |
| `make test-integration` | Tests d'intégration |
| `make test-e2e` | Tests de bout en bout sur l'API |
| `make test-smoke BASE_URL=...` | Smoke tests contre un environnement déployé |
| `make coverage` | Tests avec le seuil de couverture de 80 % |

### Chaîne ML

| Cible | Effet |
| --- | --- |
| `make data` | Télécharge, valide et prépare le corpus de travail |
| `make train-scratch` | Entraîne le Transformer from scratch sur 100 % du corpus |
| `make train-pretrained` | Fine-tune T5 sur 100 % du corpus |
| `make evaluate` | Évalue tous les modèles sur le jeu de test commun |
| `make ablation` | Rejoue les ablations taille de corpus et architecture |
| `make figures` | Trace les quatre figures de la section 18 |
| `make reproduce MODE=quick` | Vérifie la chaîne complète sur un budget plafonné, sans produire de résultat |
| `make reproduce MODE=full` | Rejoue la chaîne scientifique complète |
| `make mlflow-ui` | Sert l'interface MLflow du magasin local sur <http://localhost:5000> |

Deux commandes complètent la chaîne, hors `make` parce qu'elles portent sur une expérience précise :

| Commande | Effet |
| --- | --- |
| `python -m src.tracking.log --all` | Renvoie vers MLflow les enregistrements déjà écrits |
| `python -m src.experiments.publish --experiment NOM` | Publie un run complet dans le registre de modèles |

Le lanceur trace chaque expérience terminée sans qu'on le lui demande. `--no-tracking` le désactive.

### API et conteneurs

| Cible | Effet |
| --- | --- |
| `make api` | Lance l'API en local avec rechargement automatique |
| `make docker-build` | Construit l'image backend |
| `make docker-build-training` | Construit l'image d'entraînement |
| `make docker-size` | Mesure la taille de l'image backend construite |
| `make docker-scan` | Scanne l'image, échoue sur une vulnérabilité CRITICAL |
| `make docker-up` | Démarre la stack locale, sans entraînement |
| `make docker-train CONFIG=...` | Lance un entraînement dans la stack |
| `make docker-down` | Arrête la stack locale |

La stack locale demande un fichier `.env` : `cp .env.example .env`. Les trois images, les ports publiés et ce qui est vérifié sans Docker sont décrits dans [la documentation de déploiement](docs/deployment.md).

L'API expose `/api/v1` : `health`, `ready`, `models`, `predict` et `compare`. Le contrat est décrit dans [la documentation de l'API](docs/api/index.md) et versionné dans `docs/api/openapi.json`, régénéré par `make docs-sync`.

### Documentation et déploiement

| Cible | Effet |
| --- | --- |
| `make docs-sync` | Régénère les contenus dérivés du code |
| `make report-sync` | Régénère les tableaux du rapport depuis les enregistrements de runs |
| `make docs-lint` | Orthographe, Markdown et contrôle de synchronisation |
| `make docs` | Construit le site en mode strict |
| `make docs-serve` | Sert la documentation en local |
| `make deploy` | Déploie l'image taggée |

### Agrégats

| Cible | Effet |
| --- | --- |
| `make ci` | Reproduit localement la séquence complète du pipeline |
| `make clean` | Supprime les caches et les rapports générés |

## Intégrité scientifique

Aucune expérience n'a été exécutée à ce jour. Aucun résultat n'est publié.

Une expérience non exécutée porte le statut `NOT_RUN`, une expérience en échec le statut `FAILED`, et `MOCK` est réservé aux tests techniques. Une valeur factice ne devient jamais un résultat scientifique.

Le « 100 % » du corpus désigne le sous-ensemble de travail de 20 000 exemples, pas XSum complet. Cette convention est rappelée sur chaque tableau et chaque figure.

## État d'avancement

| Étape | Contenu | État |
| --- | --- | --- |
| 1 à 4 | Requirements, architecture, bootstrap, outillage qualité | Fait |
| 5 | Data pipeline, corpus de travail construit | Fait |
| 6 et 7 | Transformer from scratch et ses tests | Fait |
| 8 à 11 | Entraînement, baseline, évaluation, ablations | Fait, les neuf expériences mesurées |
| 12 et 13 | MLflow et Model Registry | Fait, neuf runs tracés, deux modèles publiés |
| 14 | API FastAPI | Fait, `champion` et `challenger` servis |
| 15 | Docker et scan Trivy | Écrit, aucune image construite |
| 16 et 17 | CI/CD GitHub Actions, déploiement | Pipeline écrit et validé localement, jamais exécuté sur un runner ; déploiement à faire |
| 18 | Documentation et GitHub Pages | En cours |

Toutes les cibles de la chaîne ML sont opérationnelles, `make reproduce` compris. Le mode `quick` a été joué de bout en bout sur ce poste le 9 août 2026 : cinq étapes, 51 secondes, neuf enregistrements `PARTIAL` écrits sous `reports/quick/` et aucun fichier touché hors de ce répertoire. Le mode `full` rejoue la campagne réelle.

## Résultats

Les neuf expériences ont tourné le 8 août 2026, sur GPU NVIDIA GeForce RTX 5060 portable, pour 127 minutes de calcul cumulé. Toutes portent le statut `OK`.

<!-- syntra:begin headline -->
<!-- Généré par python -m src.experiments.fragments. Ne pas éditer à la main. -->

| Modèle | Corpus | ROUGE-L | IC 95 % |
| --- | --- | --- | --- |
| `t5-small` fine-tuné | 100 % | **0,2295** | [0,2227, 0,2365] |
| `t5-small` fine-tuné | 50 % | 0,2191 | [0,2129, 0,2256] |
| `t5-small` fine-tuné | 10 % | 0,1843 | [0,1782, 0,1900] |
| from scratch | 100 % | 0,1634 | [0,1581, 0,1686] |
| from scratch | 50 % | 0,1550 | [0,1499, 0,1600] |
| `t5-small` zero-shot | sans objet | 0,1366 | [0,1328, 0,1406] |
| from scratch | 10 % | 0,1250 | [0,1207, 0,1293] |
<!-- syntra:end headline -->

Le modèle pré-entraîné fine-tuné sur 2 000 exemples devance le Transformer from scratch entraîné sur 20 000, et l'écart entre les deux familles reste stable autour de 40 % sur toute la plage. L'ablation d'architecture, elle, ne montre aucun gain à augmenter la profondeur.

Ces chiffres sont reproductibles par un tiers : les quatre mesures `t5-small` ont été refaites sous la révision `df1b051c`, épinglée dans les fichiers `pretrained_*`. Voir [Conformité aux requirements](docs/conformity.md).

## Documentation

Le [rapport](docs/report.md) est le document à lire en premier : il présente l'architecture du Transformer from scratch, la courbe de performance contre la taille du corpus, et il répond à la question de savoir à partir de quelle taille le modèle from scratch devient compétitif.

L'analyse exploratoire qui fixe les réglages d'entraînement est dans `notebooks/01_eda_xsum.ipynb`, dont les conclusions durables sont recopiées sur la page [Corpus](docs/ml/data.md). Elle demande le groupe optionnel `eda` : `pip install -e ".[eda]"`.

Le notebook `notebooks/00_environment_check.ipynb` se lance avant tout le reste : il vérifie que ce poste peut exécuter la chaîne, et sur quoi. Il est décrit sur la page [Reproductibilité](docs/reproducibility.md).

La documentation complète se construit avec `make docs` et se consulte avec `make docs-serve`. Elle couvre la reproductibilité, les tests, la sécurité, la contribution et le glossaire terminologique.

## Rédaction

Les tirets cadratin et demi-cadratin sont interdits dans tout le dépôt. La règle est vérifiée mécaniquement par `scripts/check_dashes.py`, en hook et en CI. Elle est bloquante.
