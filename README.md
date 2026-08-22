# Syntra

Chaîne ML expérimentale de résumé automatique. Le projet compare un Transformer encodeur-décodeur implémenté à la main en PyTorch à un modèle pré-entraîné T5, en zero-shot puis fine-tuné, sur un jeu de test strictement identique.

Le périmètre couvre la chaîne scientifique : corpus, modèles, entraînement, évaluation, ablations et traçage MLflow. Ni backend, ni conteneurs, ni frontend.

## Démarrage rapide

```bash
git clone <url-du-depot> syntra && cd syntra
py -3.12 -m venv .venv && source .venv/Scripts/activate   # Linux, macOS : .venv/bin/activate
make install
make reproduce MODE=quick
```

Quatre commandes depuis un dépôt fraîchement cloné. La dernière enchaîne toute la chaîne scientifique : elle construit le corpus s'il manque, joue les neuf expériences plafonnées à deux pas d'optimisation, agrège les tableaux et trace les figures. Elle vérifie que la chaîne tourne sur ce poste ; elle ne produit aucun résultat, et le dit. Comptez une minute une fois le corpus construit, mesuré sur GPU RTX 5060 portable.

Le premier appel télécharge CNN/DailyMail et construit le corpus de travail, ce qui domine le temps total : l'archive fait 1,3 Go. Les appels suivants sautent cette étape : `make data` est idempotent, et la chaîne affiche le `dataset_version` du corpus qu'elle a lu.

```bash
make coverage               # la suite de tests, seuil de couverture compris
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
| Corpus | CNN/DailyMail 3.0.0, sous-ensemble figé de 20 000 exemples d'entraînement |
| Matériel | GPU NVIDIA local |

## Prérequis

```text
Python 3.12
GPU NVIDIA avec pilote récent, optionnel mais recommandé
GNU Make
```

Sous Windows, `make` n'est pas fourni par le système :

```powershell
winget install ezwinports.make
```

Le PATH n'est mis à jour qu'au shell suivant.

Il faut ensuite **appeler `make` depuis Git Bash, pas depuis PowerShell**. Les recettes de ce `Makefile` sont écrites pour un shell POSIX : `test -n`, `awk`, `rm -rf`, `mkdir -p`. GNU Make ne retient un shell POSIX que s'il trouve `sh.exe` dans le PATH, sinon il retombe sur `cmd.exe`. Or l'installation courante de Git pour Windows ne publie que `C:\Program Files\Git\cmd`, où `sh.exe` ne se trouve pas : il vit dans `C:\Program Files\Git\usr\bin`. Depuis PowerShell, la première cible venue échoue donc ainsi :

```text
'grep' n'est pas reconnu en tant que commande interne ou externe
make: *** [Makefile:27: help] Error 255
```

Depuis Git Bash, la même cible passe. Trois façons d'y arriver :

```text
VS Code       menu déroulant du terminal, profil Git Bash
un terminal   & "C:\Program Files\Git\bin\bash.exe" -l
l'explorateur clic droit sur le dossier, Afficher plus d'options, Git Bash Here
```

Un terminal hérite du PATH du processus qui l'a lancé. VS Code transmet celui qu'il portait à son propre démarrage, et « Recharger la fenêtre » ne le rafraîchit pas : après l'installation, il faut quitter VS Code puis le rouvrir, sans quoi `make: command not found` persiste dans un terminal pourtant neuf. Pour débloquer la session en cours sans rien fermer :

```bash
export PATH="$PATH:$HOME/AppData/Local/Microsoft/WinGet/Packages/ezwinports.make_Microsoft.Winget.Source_8wekyb3d8bbwe/bin"
```

Le `Makefile` appelle l'interpréteur du `.venv`. Si cet environnement est absent ou incomplet, les cibles s'appellent directement : `python -m src.experiments.run --config ...`.

## Installation

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
make install
```

`make install` installe PyTorch depuis l'index CUDA 13.0, puis les dépendances de la chaîne et de la suite de tests.

> Les cartes RTX 50 (architecture Blackwell, `sm_120`) exigent les roues PyTorch CUDA. L'index PyPI par défaut ne convient pas.

Vérifier l'installation :

```bash
make test
```

## Structure

```text
.
├── Makefile                     toutes les cibles de la chaîne, appelées depuis Git Bash
├── pyproject.toml               dépendances et configuration des outils
├── GUIDE.md                     parcours de lecture du code, du corpus à MLflow
├── RAPPORT.md                   le rapport scientifique et ses résultats
├── src/                         code de recherche
│   ├── data/                    téléchargement, validation, tokenisation, corpus figé
│   ├── models/
│   │   ├── scratch/             le Transformer écrit à la main, 15 modules
│   │   └── pretrained/          l'adaptateur t5-small
│   ├── training/                boucle d'entraînement, optimiseur, arrêt anticipé
│   ├── evaluation/              génération et évaluation sur le jeu de test commun
│   ├── metrics/                 ROUGE et intervalles de confiance
│   ├── experiments/             lanceur, registre des neuf expériences, ablations, figures
│   ├── tracking/                enregistrement des runs et envoi vers MLflow
│   └── utils/                   graine aléatoire, périphérique, markdown
├── configs/
│   ├── data/                    cnn_dailymail.yaml : corpus et tokenizer
│   ├── experiments/             les neuf expériences, une par fichier
│   ├── model/                   vide, les hyperparamètres vivent dans les expériences
│   └── training/                vide, pour la même raison
├── data/                        corpus de travail, reconstruit par make data
│   ├── raw/                     CNN/DailyMail tel que téléchargé
│   ├── interim/                 étapes intermédiaires
│   ├── processed/               le corpus figé de 20 000 exemples
│   └── external/                ressources tierces
├── reports/
│   ├── results/                 un enregistrement par run
│   ├── figures/                 les quatre figures du rapport
│   └── _generated/              les fragments injectés dans ce README et le rapport
├── notebooks/                   00 environnement, 01 corpus, 02 entraînement, 03 Transformer
├── scripts/                     measure_padding.py et compare_archive.py
└── tests/
    ├── unit/                    46 fichiers de test
    └── integration/             8 fichiers de test

Le contenu de data/, de reports/results/ et de reports/quick/ n'est pas versionné :
il est reconstruit par make data et par la chaîne d'expériences.

Créés à l'usage, jamais versionnés :

.venv/       environnement Python appelé par le Makefile
runs/        checkpoints d'entraînement, plusieurs Go
mlflow.db    magasin MLflow servi par make mlflow-ui
mlruns/      artefacts MLflow associés
```

## Commandes

### Environnement

| Cible | Effet |
| --- | --- |
| `make install` | Installe PyTorch, les dépendances et le paquet en mode éditable |

### Tests

| Cible | Effet |
| --- | --- |
| `make test` | Toute la suite |
| `make test-unit` | Tests unitaires |
| `make test-integration` | Tests d'intégration |
| `make coverage` | Tests avec le seuil de couverture de 80 % |

### Chaîne ML

| Cible | Effet |
| --- | --- |
| `make data` | Télécharge, valide et prépare le corpus de travail |
| `make train-scratch` | Entraîne le Transformer from scratch sur 100 % du corpus |
| `make train-pretrained` | Fine-tune T5 sur 100 % du corpus |
| `make evaluate` | Évalue la baseline zero-shot sur le jeu de test commun |
| `make ablation` | Rejoue les ablations taille de corpus et architecture |
| `make figures` | Trace les quatre figures dans `reports/figures` |
| `make reproduce MODE=quick` | Vérifie la chaîne complète sur un budget plafonné, sans produire de résultat |
| `make reproduce MODE=full` | Rejoue la chaîne scientifique complète |
| `make mlflow-ui` | Sert l'interface MLflow du magasin local sur <http://localhost:5000> |

Une commande complète la chaîne, hors `make` parce qu'elle porte sur des enregistrements déjà écrits :

| Commande | Effet |
| --- | --- |
| `python -m src.tracking.log --all` | Renvoie vers MLflow les enregistrements déjà écrits |

Le lanceur trace chaque expérience terminée sans qu'on le lui demande. `--no-tracking` le désactive.

### Artefacts dérivés

| Cible | Effet |
| --- | --- |
| `make report-sync` | Régénère les tableaux de résultats depuis les enregistrements de runs, et les réinjecte dans ce README et dans le rapport |
| `make corpus-sync` | Régénère les tableaux du corpus depuis le manifeste, et les réinjecte dans le rapport |
| `make clean` | Supprime les caches et les rapports générés |

## Intégrité scientifique

Les tableaux de ce README et du rapport sont générés depuis les enregistrements de runs par `make report-sync`, puis injectés entre marqueurs. Une nouvelle campagne les réécrit. Tant qu'une expérience n'a pas tourné, sa ligne existe et porte `NOT_RUN` : le tableau ne raccourcit pas, il dit ce qui manque.

Une expérience non exécutée porte le statut `NOT_RUN`, une expérience en échec le statut `FAILED`, et `MOCK` est réservé aux tests techniques. Seuls les enregistrements `OK` entrent dans les tableaux.

Le « 100 % » du corpus désigne le sous-ensemble de travail de 20 000 exemples, pas CNN/DailyMail complet. Cette convention est rappelée sur chaque tableau et chaque figure.

## État d'avancement

| Étape | Contenu | État |
| --- | --- | --- |
| 1 à 4 | Requirements, architecture, bootstrap | Fait |
| 5 | Data pipeline, corpus de travail construit | Fait |
| 6 et 7 | Transformer from scratch et ses tests | Fait |
| 8 à 11 | Entraînement, baseline, évaluation, ablations | Chaîne complète, campagne CNN/DailyMail en cours |
| 12 et 13 | Traçage MLflow | Chaîne complète, magasin remis à zéro avec le changement de corpus |
| 18 | Rapport | `RAPPORT.md`, sections de résultats en attente de la campagne |

Toutes les cibles de la chaîne ML sont opérationnelles, `make reproduce` compris. Le mode `full` joue la campagne réelle ; le mode `quick` vérifie la chaîne sur un budget plafonné et n'écrit rien hors de `reports/quick/`.

## Résultats

Le corpus a changé : les résultats XSum ont été supprimés et les neuf expériences sont relancées sur CNN/DailyMail. Le tableau ci-dessous est régénéré depuis les enregistrements de runs et porte `NOT_RUN` tant que la campagne n'a pas écrit.

<!-- syntra:begin headline -->
<!-- Généré par python -m src.experiments.fragments. Ne pas éditer à la main. -->

| Modèle | Corpus | ROUGE-L | IC 95 % | Statut |
| --- | --- | --- | --- | --- |
| `t5-small` fine-tuné | 10 % |  |  | `NOT_RUN` |
| `t5-small` fine-tuné | 100 % |  |  | `NOT_RUN` |
| `t5-small` fine-tuné | 50 % |  |  | `NOT_RUN` |
| `t5-small` zero-shot | sans objet |  |  | `NOT_RUN` |
| from scratch | 10 % |  |  | `NOT_RUN` |
| from scratch | 100 % |  |  | `NOT_RUN` |
| from scratch | 50 % |  |  | `NOT_RUN` |
<!-- syntra:end headline -->

Les quatre mesures `t5-small` sont prises sous la révision `df1b051c`, épinglée dans les fichiers `pretrained_*`, et le corpus sous `dataset_version = 00c0ee4e` : les deux voyagent dans chaque enregistrement de run.

## Documentation

Deux documents, deux usages.

Le [guide](GUIDE.md) explique **comment le code fonctionne**. Il suit un batch du fichier brut jusqu'au tableau de comparaison : corpus, tokenisation, Transformer couche par couche, étape d'entraînement, métriques, MLflow. C'est la porte d'entrée pour un nouveau contributeur.

Le [rapport](RAPPORT.md) présente **ce que les expériences ont montré** : le corpus, l'architecture, le protocole d'évaluation, la courbe de performance contre la taille du corpus, et il répond à la question de savoir à partir de quelle taille le modèle from scratch devient compétitif.

L'analyse exploratoire qui fixe les réglages d'entraînement est dans `notebooks/01_eda_cnn_dailymail.ipynb`. Elle demande le groupe optionnel `eda` : `pip install -e ".[eda]"`.

Le notebook `notebooks/00_environment_check.ipynb` se lance avant tout le reste : il vérifie que ce poste peut exécuter la chaîne, et sur quoi.

Le notebook [notebooks/03_transformer_walkthrough.ipynb](notebooks/03_transformer_walkthrough.ipynb) fait traverser le Transformer à un vrai batch en affichant la forme des tenseurs à chaque étape. Il accompagne la section 3 du guide, tourne sur CPU en une minute et n'écrit rien.

Le notebook `notebooks/02_training.ipynb` lance une expérience à la fois et trace ses courbes de perte. Il appelle `run_one`, la fonction que `python -m src.experiments.run` et `make reproduce` appellent aussi, et son mode `quick` écrit sous `reports/quick/` avec les mêmes plafonds. Une campagne complète, elle, se lance depuis un terminal : `make reproduce MODE=full`.
