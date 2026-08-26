# Syntra

Chaîne ML expérimentale de résumé automatique. Le projet compare trois familles sur un jeu de test strictement identique : un Transformer encodeur-décodeur écrit bloc par bloc, l'architecture `t5-small` initialisée aléatoirement, et la même `t5-small` chargée avec ses poids pré-entraînés, en zero-shot puis fine-tunée.

Le périmètre couvre la chaîne scientifique : corpus, modèles, entraînement, évaluation, ablations et traçage MLflow.

## Démarrage rapide

```bash
git clone <url-du-depot> syntra && cd syntra
py -3.12 -m venv .venv && source .venv/Scripts/activate   # Linux, macOS : .venv/bin/activate
make install
make reproduce MODE=quick
```

La dernière commande enchaîne toute la chaîne scientifique : elle construit le corpus s'il manque, joue les douze expériences plafonnées à deux pas d'optimisation, agrège les tableaux et trace les figures. Elle vérifie que la chaîne tourne sur ce poste ; elle ne produit aucun résultat, et le dit. Comptez une minute une fois le corpus construit, mesuré sur GPU RTX 5060 portable.

Le premier appel télécharge CNN/DailyMail et construit le corpus de travail, ce qui domine le temps total : l'archive fait 1,3 Go. Les appels suivants sautent cette étape : `make data` est idempotent, et la chaîne affiche le `dataset_version` du corpus qu'elle a lu.

```bash
make coverage               # la suite de tests, seuil de couverture compris
make reproduce MODE=full    # la campagne réelle, plusieurs heures de GPU
```

Sous Windows, `make` s'appelle depuis Git Bash et non depuis PowerShell ; la raison est dans [Prérequis](#prérequis). Pour rester sous PowerShell, `make.ps1` porte les mêmes cibles :

```powershell
.\make.ps1                       # la liste des cibles
.\make.ps1 install
.\make.ps1 reproduce -Mode quick
```

## Décisions verrouillées

| Sujet | Choix |
| --- | --- |
| Tâche | Résumé automatique |
| Métriques | ROUGE-1, ROUGE-2, ROUGE-L |
| Modèle from scratch | Transformer encodeur-décodeur écrit bloc par bloc, à la forme de `t5-small` |
| Témoin d'initialisation | Architecture `t5-small`, poids initialisés aléatoirement |
| Modèle pré-entraîné | `t5-small` |
| Tokenizer | Tokenizer T5, partagé par les trois familles |
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

Depuis Git Bash, la même cible passe. Pour l'ouvrir :

```text
VS Code       menu déroulant du terminal, profil Git Bash
un terminal   & "C:\Program Files\Git\bin\bash.exe" -l
l'explorateur clic droit sur le dossier, Afficher plus d'options, Git Bash Here
```

Un terminal hérite du PATH du processus qui l'a lancé. VS Code transmet celui qu'il portait à son propre démarrage, et « Recharger la fenêtre » ne le rafraîchit pas : après l'installation, il faut quitter VS Code puis le rouvrir, sans quoi `make: command not found` persiste dans un terminal pourtant neuf. Pour débloquer la session en cours sans rien fermer :

```bash
export PATH="$PATH:$HOME/AppData/Local/Microsoft/WinGet/Packages/ezwinports.make_Microsoft.Winget.Source_8wekyb3d8bbwe/bin"
```

### Rester sous PowerShell

Rien de ce qui précède n'est nécessaire si l'on passe par `make.ps1`, qui porte les mêmes cibles sans dépendre ni de GNU Make ni d'un shell POSIX :

```powershell
.\make.ps1                       # la liste des cibles
.\make.ps1 reproduce -Mode full  # equivaut a make reproduce MODE=full
```

Le `Makefile` reste la référence : c'est lui qui tourne en CI. Les deux fichiers listent les mêmes cibles, et une cible ajoutée d'un côté doit l'être de l'autre.

### Quand le venv manque

Les deux points d'entrée appellent l'interpréteur du `.venv`, et retombent sur le `python` du PATH quand il est absent. **Ce repli est un piège si le PATH porte un Python hors de la plage `>=3.12,<3.14`** : `install` déverserait alors PyTorch dans le Python système, et `test` échouerait sur un `ModuleNotFoundError` qui ne dit pas pourquoi. `make.ps1` vérifie la version et s'arrête ; le `Makefile`, lui, ne le voit pas. Devant un import manquant, vérifier `.venv/Scripts/python.exe` avant toute autre hypothèse, et recréer l'environnement plutôt qu'emprunter un interpréteur voisin :

```powershell
py -3.12 -m venv .venv
.\make.ps1 install
```

## Installation

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
make install
```

`make install` installe PyTorch depuis l'index CUDA 13.0, puis les dépendances de la chaîne et de la suite de tests.

Les dépendances sont déclarées une fois, dans `pyproject.toml`. Les trois fichiers `requirements*.txt` en sont le reflet, pour qui préfère `pip install -r` ; `tests/unit/test_requirements.py` échoue si les deux divergent.

| Fichier | Contenu |
| --- | --- |
| `requirements.txt` | les dépendances d'exécution de la chaîne |
| `requirements-dev.txt` | y ajoute pytest, pytest-cov et ruff |
| `requirements-eda.txt` | y ajoute JupyterLab, ipykernel et nbconvert |

> `pip install -r requirements.txt` installe le **PyTorch CPU** publié sur PyPI. Sur un poste GPU, passer par la cible `install`, qui tire d'abord PyTorch depuis l'index CUDA.
>
> Les cartes RTX 50 (architecture Blackwell, `sm_120`) exigent les roues PyTorch CUDA. L'index PyPI par défaut ne convient pas.

Vérifier l'installation :

```bash
make test
```

## Structure

```text
.
├── Makefile                     toutes les cibles de la chaîne, appelées depuis Git Bash
├── make.ps1                     les mêmes cibles, appelées depuis PowerShell
├── pyproject.toml               dépendances et configuration des outils
├── requirements.txt             reflet des dépendances de pyproject.toml
├── requirements-dev.txt         y ajoute l'outillage de test
├── requirements-eda.txt         y ajoute la chaîne Jupyter des carnets
├── .env.example                 modèle de configuration du magasin MLflow, sans secret
├── GUIDE.md                     parcours de lecture du code, du corpus à MLflow
├── RAPPORT.md                   le rapport scientifique et ses résultats
├── src/                         code de recherche
│   ├── data/                    téléchargement, validation, tokenisation, corpus figé
│   ├── models/
│   │   ├── scratch/             le Transformer écrit bloc par bloc
│   │   ├── pretrained/          t5-small, chargé avec ou sans ses poids
│   │   └── generation.py        la configuration de décodage, commune aux trois familles
│   ├── training/                boucle d'entraînement, optimiseur, arrêt anticipé
│   ├── evaluation/              génération et évaluation sur le jeu de test commun
│   ├── metrics/                 ROUGE et intervalles de confiance
│   ├── experiments/             lanceur, registre des douze expériences, ablations, figures
│   ├── tracking/                enregistrement des runs et envoi vers MLflow
│   └── utils/                   graine aléatoire, périphérique, markdown
├── configs/
│   ├── data/                    cnn_dailymail.yaml : corpus et tokenizer
│   ├── experiments/             les douze expériences, une par fichier
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
├── notebooks/                   00 environnement, 01 corpus, 02 entraînement, 03 le T5 forme par forme
├── scripts/                     measure_padding.py
└── tests/
    ├── unit/                    44 fichiers de test
    └── integration/             8 fichiers de test

Le contenu de data/, de reports/results/ et de reports/quick/ n'est pas versionné :
il est reconstruit par make data et par la chaîne d'expériences.

Créés à l'usage, jamais versionnés :

.venv/         environnement Python appelé par le Makefile
.env           URI du magasin MLflow, avec son mot de passe
runs/          checkpoints d'entraînement, plusieurs Go
mlartifacts/   artefacts MLflow, modèles compris, environ 1,5 Go par campagne
```

## Commandes

Les cibles ci-dessous sont écrites pour `make`, depuis Git Bash. Sous PowerShell, `make.ps1` porte les mêmes : `make <cible>` s'écrit `.\make.ps1 <cible>`, et `MODE=full` devient `-Mode full`.

```powershell
.\make.ps1 test
.\make.ps1 reproduce -Mode full
```

### Environnement

| Cible | Effet |
| --- | --- |
| `make install` | Installe PyTorch, les dépendances et le paquet en mode éditable |
| `make kernel` | Enregistre le kernel Jupyter du dépôt et installe la chaîne Jupyter |

### Qualité

| Cible | Effet |
| --- | --- |
| `make lint` | Passe ruff sur le code, les tests et les carnets |

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
| `make train-scratch` | Entraîne le Transformer écrit à la main sur 100 % du corpus |
| `make train-random` | Entraîne `t5-small` initialisé aléatoirement sur 100 % du corpus |
| `make train-pretrained` | Fine-tune T5 sur 100 % du corpus |
| `make evaluate` | Évalue la baseline zero-shot sur le jeu de test commun |
| `make ablation` | Rejoue les ablations taille de corpus et architecture |
| `make figures` | Trace les quatre figures dans `reports/figures` |
| `make reproduce MODE=quick` | Vérifie la chaîne complète sur un budget plafonné, sans produire de résultat |
| `make reproduce MODE=full` | Rejoue la chaîne scientifique complète |
| `make mlflow-ui` | Sert l'interface MLflow du magasin configuré sur <http://localhost:5000> |

Deux commandes vivent hors de `make`, parce qu'elles portent sur ce qui est déjà écrit :

| Commande | Effet |
| --- | --- |
| `python -m src.tracking.log --all` | Renvoie vers MLflow les enregistrements déjà écrits |
| `python -m src.tracking.purge --all` | Compte les runs du magasin ; les supprime avec `--yes` |

Rejouer une campagne depuis zéro laisse la précédente dans le magasin à côté de la nouvelle : deux
réponses par expérience, et rien dans l'interface ne dit laquelle le rapport cite. `purge` vide le
magasin avant la relance. Sans `--yes` il ne fait que compter, parce qu'une fois `reports/results/`
effacé le magasin est le seul endroit où une mesure existe encore.

Le lanceur trace chaque expérience terminée sans qu'on le lui demande. `--no-tracking` le désactive.

### Magasin MLflow

Les métadonnées des runs vont dans une base **PostgreSQL** dédiée, les artefacts et les modèles dans un répertoire. L'URI porte un mot de passe et reste donc hors du dépôt :

```bash
cp .env.example .env      # puis remplir MLFLOW_TRACKING_URI
```

Sans `.env`, MLflow retombe sur un fichier SQLite local et le lanceur affiche lequel des deux magasins il a obtenu, mot de passe masqué, avant la première expérience.

**Ce repli joue sur l'absence de configuration, pas sur un serveur injoignable.** `resolve_tracking_uri` rend `None` quand rien ne configure d'URI, et c'est ce `None` qui laisse MLflow choisir SQLite. Avec un `.env` en place et la base éteinte, il n'y a pas de repli : `make mlflow-ui` attend puis échoue sur un timeout. La base vit sur une autre machine du réseau local, qui doit donc tourner. Pour vérifier avant de lancer quoi que ce soit :

```powershell
Test-NetConnection -ComputerName <hôte> -Port 5432
```

Le magasin SQLite local reste consultable pendant ce temps, avec ce qu'il porte des campagnes passées :

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root mlartifacts
```

Une campagne lancée alors que la base est injoignable tourne quand même : un échec de traçage n'interrompt jamais un run. Seul le dépôt dans le magasin est perdu, et `python -m src.tracking.log --all` le rattrape une fois le serveur revenu.

Chaque run complet dépose le modèle qu'il a mesuré dans le magasin, enregistré sous `syntra-<expérience>` :

```python
import mlflow

t5 = mlflow.transformers.load_model("models:/syntra-pretrained_ft_100/1")
maison = mlflow.pytorch.load_model("models:/syntra-scratch_100/1")
```

Les deux saveurs ne sont pas interchangeables : `mlflow.transformers` porte les deux branches `t5-small` avec leur tokenizer, `mlflow.pytorch` porte le Transformer écrit à la main, qui n'a pas de saveur à lui.

### Artefacts dérivés

| Cible | Effet |
| --- | --- |
| `make report-sync` | Régénère les tableaux de résultats depuis les enregistrements de runs, et les réinjecte dans ce README et dans le rapport |
| `make corpus-sync` | Régénère les tableaux du corpus depuis le manifeste, et les réinjecte dans le rapport |
| `make clean` | Supprime les caches et les rapports générés |

## Statuts et conventions

Une expérience non exécutée porte le statut `NOT_RUN`, une expérience en échec le statut `FAILED`, et `MOCK` est réservé aux tests techniques. Un répertoire de run laissé par une autre architecture que celle déclarée aujourd'hui porte `STALE_CONFIG` et ne fournit aucun score. Seuls les enregistrements `OK` entrent dans les tableaux, qui ne raccourcissent jamais : une ligne `NOT_RUN` dit ce qui manque.

Le « 100 % » du corpus désigne le sous-ensemble de travail de 20 000 exemples, pas CNN/DailyMail complet. Cette convention est rappelée sur chaque tableau et chaque figure.

## Résultats

Trois familles sont comparées à budget de paramètres égal : le Transformer écrit à la main pèse 60 575 744 paramètres, les deux `t5-small` en pèsent 60 506 624, soit 0,11 % d'écart. Seul le couple `random_t5_*` contre `pretrained_ft_*` partage l'architecture elle-même, et c'est donc le seul dont l'écart s'attribue au pré-entraînement seul. Les quatre runs pré-entraînés sont mesurés ; les huit autres sont déclarés et pas encore exécutés, ce que le tableau dit ligne par ligne.

<!-- syntra:begin headline -->
<!-- Généré par python -m src.experiments.fragments. Ne pas éditer à la main. -->

| Modèle | Corpus | ROUGE-L | IC 95 % |
| --- | --- | --- | --- |
| `t5-small` fine-tuné | 100 % | **0,2915** | [0,2836, 0,2997] |
| `t5-small` fine-tuné | 50 % | 0,2894 | [0,2818, 0,2973] |
| `t5-small` fine-tuné | 10 % | 0,2867 | [0,2784, 0,2945] |
| `t5-small` zero-shot | sans objet | 0,2751 | [0,2672, 0,2829] |
| `t5-small` aléatoire | 100 % | 0,1172 | [0,1145, 0,1200] |
| Transformer from scratch | 100 % | 0,1156 | [0,1129, 0,1184] |
| Transformer from scratch | 50 % | 0,1014 | [0,0987, 0,1042] |
| `t5-small` aléatoire | 50 % | 0,0985 | [0,0961, 0,1011] |
| `t5-small` aléatoire | 10 % | 0,0771 | [0,0745, 0,0798] |
| Transformer from scratch | 10 % | 0,0343 | [0,0327, 0,0357] |
<!-- syntra:end headline -->

Les quatre mesures `t5-small` sont prises sous la révision `df1b051c`, épinglée dans les fichiers `pretrained_*`, et le corpus sous `dataset_version = 00c0ee4e` : les deux voyagent dans chaque enregistrement de run.

## Documentation

Le [guide](GUIDE.md) explique **comment le code fonctionne**. Il suit un batch du fichier brut jusqu'au tableau de comparaison : corpus, tokenisation, T5 couche par couche, étape d'entraînement, métriques, MLflow.

Le [rapport](RAPPORT.md) présente le corpus, les trois familles, le protocole d'évaluation et l'état des résultats. Les conclusions comparatives seront établies quand les runs `scratch_*` et `random_t5_*` auront tourné.

Les quatre carnets demandent le groupe optionnel `eda`, et le kernel du dépôt :

```bash
make kernel                 # .\make.ps1 kernel sous PowerShell
```

Cette cible installe la chaîne Jupyter puis enregistre un kernel nommé `train-vs-pre-train`, affiché **Train-vs-Pre-train (.venv)** dans le sélecteur de VS Code. C'est celui qu'il faut choisir. Le kernel `python3` que Jupyter propose à côté n'est pas équivalent : son `argv` est un `python` nu, résolu depuis le PATH au lancement, donc pas nécessairement celui du dépôt. Les carnets épinglent le bon dans leur métadonnée, et `tests/unit/test_notebooks.py` échoue si l'éditeur les réassigne, ce qu'il fait dès que le kernel du dépôt n'est plus enregistré.

L'analyse exploratoire qui fixe les réglages d'entraînement est dans `notebooks/01_eda_cnn_dailymail.ipynb`.

Le notebook `notebooks/00_environment_check.ipynb` se lance avant tout le reste : il vérifie que ce poste peut exécuter la chaîne, et sur quoi.

Le notebook [notebooks/03_transformer_walkthrough.ipynb](notebooks/03_transformer_walkthrough.ipynb) fait traverser le Transformer à un vrai batch en affichant la forme des tenseurs à chaque étape. Il accompagne la section 3 du guide, tourne sur CPU en une minute et n'écrit rien.

Le notebook `notebooks/02_training.ipynb` lance une campagne et la donne à suivre. `EXPERIMENTS` nomme celles à jouer, dans l'ordre voulu, ou `None` pour les douze déclarées, dans l'ordre de la campagne : le Transformer écrit à la main d'abord, puis `t5-small` initialisé aléatoirement, des plus petites proportions de corpus aux plus grandes, puis `t5-small` zero-shot, puis ses fine-tunes ; c'est le défaut, avec `MODE = "full"`, donc un Run All lance la campagne complète. Chaque expérience ouvre une bannière `i/N`, ses pas s'écrivent au fil de l'eau, et une ligne la referme ; un tableau final aligne les douze sur leur statut, leur durée et leur ROUGE-L. Une expérience qui échoue est enregistrée `FAILED` sans interrompre les suivantes, et une interruption au clavier laisse le bilan s'imprimer sur ce qui a tourné. Les sections 5 à 8 détaillent ensuite une seule expérience, que `FOCUS` désigne.

Il appelle `run_one`, la fonction que `python -m src.experiments.run` et `make reproduce` appellent aussi, et son mode `quick` écrit sous `reports/quick/` avec les mêmes plafonds.

Il sait aussi décrire une campagne qu'il n'a pas lancée : après un `make reproduce MODE=full` au terminal, exécuter les sections 1 à 3 puis sauter à la section 5 suffit ; elle relit les enregistrements de `reports/results/`, et rien n'est réentraîné.

> Une campagne complète dure plusieurs heures et meurt avec le noyau : fermer VS Code l'emporte. Pour la lancer sans cette contrainte, passer par un terminal : `make reproduce MODE=full`, ou `.\make.ps1 reproduce -Mode full`.
