# Traçabilité MLflow

Section 19 du cahier des charges.

## Composants

```text
src/tracking/
├── provenance.py   le commit et l'état de l'arbre de travail
├── payload.py      un enregistrement de run traduit en paramètres et métriques
├── client.py       le seul module qui importe MLflow
└── log.py          renvoi vers le magasin des enregistrements déjà écrits
```

Le lanceur d'expériences écrit `run.json`, puis le trace. L'ordre compte : le résultat d'une expérience est le fichier sur le disque, le magasin de tracking en est le miroir.

## Ce qui est tracé

La section 19 liste douze champs. Tous sont déjà dans l'enregistrement de run, ce qui rend la traduction mécanique.

| Champ de la section 19 | Origine dans `run.json` | Forme MLflow |
| --- | --- | --- |
| `experiment_id` | `experiment` | paramètre et nom du run |
| `git_commit` | `provenance.git_commit` | tag |
| `seed` | `config.experiment.seed` | paramètre |
| `dataset_version` | `dataset.version` | paramètre |
| `dataset_percentage` | `dataset.percentage` | paramètre |
| `model` | `model` | paramètres et tags |
| `hyperparameters` | `config.training`, `config.evaluation` | paramètres |
| `training_duration` | `training.duration_seconds` | métrique |
| `BLEU` | hors périmètre | absent |
| `ROUGE` | `evaluation.report.rouge` | métriques |
| `checkpoint` | `training.best_checkpoint` | paramètre |
| `hardware` | `hardware` | tags |

Les tags portent le préfixe `syntra.`, qui ne peut pas entrer en collision avec ceux que MLflow pose lui-même.

### Pourquoi BLEU est absent

La section 2.1 met la traduction et BLEU hors périmètre, et verrouille les métriques sur ROUGE-1, ROUGE-2 et ROUGE-L. Une colonne BLEU remplie de zéros serait une mesure inventée au sens de la section 44. Remplie d'un ROUGE, elle serait pire.

### Configuration ou environnement

Un paramètre est ce que l'expérience a déclaré, et ce qu'une reprise devrait répéter. Le matériel, le commit et l'état de l'arbre de travail décrivent l'endroit où le run s'est déroulé : ce sont des tags. Un run qui rapporte les mêmes paramètres sur un autre GPU reste la même expérience.

## Le commit est enregistré, pas recalculé

`run.json` porte un bloc `provenance` écrit au moment du run :

```json
"provenance": {
  "git_commit": "unknown",
  "git_branch": "master",
  "git_dirty": "true"
}
```

Le relire plus tard plutôt que le recalculer est ce qui empêche un enregistrement renvoyé la semaine suivante de porter le commit de ce jour-là au lieu de celui qui a produit le score.

Ce que `git` ne peut pas répondre vaut `unknown`, jamais une valeur plausible. Un dépôt dont le premier commit n'est pas encore fait, un export sans historique et une machine sans `git` donnent tous les trois un run qui mérite d'être tracé et un commit qui n'existe pas. C'est exactement l'état de ce dépôt aujourd'hui.

Un arbre de travail modifié est enregistré, pas refusé. Refuser de tracer un run parti d'une modification non commitée sortirait la mesure du magasin, ce qui est l'inverse du but de la section 19. `git_dirty` vaut `unknown` quand la question n'a pas pu être posée, ce qui n'est pas la même réponse qu'un arbre propre.

## Un run qui n'est pas une mesure

Trois familles de métriques, sous trois conditions distinctes :

| Famille | Condition |
| --- | --- |
| Durée totale du run | toujours |
| Perte de validation, époques, pas, durée d'entraînement | dès qu'une boucle d'entraînement a tourné |
| ROUGE, longueurs, prédictions vides | seulement pour un run complet |

Un magasin de tracking se lit trié par score. Une répétition sur cinquante documents dans ce classement est précisément le nombre présenté mais non mesuré qu'interdit la section 44.

Un run qui n'est pas complet porte aussi son statut dans son nom : `scratch_100:partial`, `scratch_100:failed`. Une expérience en échec est tracée elle aussi, avec son message d'erreur en tag et aucune métrique de qualité : ce qu'une campagne a produit inclut ce qui a cassé.

Le zero-shot ne déclare aucune proportion de corpus. La valeur est absente, pas nulle : le modèle n'a pas été entraîné sur rien, il n'a pas été entraîné. Un zéro le placerait à l'origine de la courbe d'ablation.

## Les artefacts

Quatre fichiers sont copiés dans le magasin quand ils existent : `run.json`, `metrics.json`, `history.json`, `qualitative.json`.

`predictions.jsonl` n'y est pas. Il porte une ligne par document évalué, il est déjà publié comme artefact de pipeline, et rien ne le lit depuis le magasin.

Le checkpoint non plus. Il est tracé par son chemin. Les poids sont l'artefact de la section 20 et appartiennent au [registre de modèles](registry.md), qui est ce que l'API lit. Recopier plusieurs centaines de mégaoctets par run produirait une seconde copie du même fichier, à l'endroit où rien ne le charge.

## Le magasin local

Sans configuration, MLflow 3 résout son URI vers une base SQLite du répertoire courant. Un entraînement hors ligne est donc tracé sans qu'un serveur soit nécessaire, et `docker compose up mlflow` n'est pas un prérequis.

Un URI en `file://` est le seul qui ne fonctionne pas : MLflow 3 a placé le backend de tracking sur système de fichiers en mode maintenance et lève plutôt que d'y écrire. Un magasin local est une base SQLite, un magasin partagé est un serveur.

`src/tracking/client.py` ne lit jamais l'environnement lui-même. C'est MLflow qui résout `MLFLOW_TRACKING_URI`, ce qui satisfait la section 21.1 : aucun module hors `backend/app/core/config.py` ne lit `os.environ`.

## Consulter le magasin

```bash
make mlflow-ui

# La même chose sans make, depuis la racine du dépôt
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

L'interface répond sur [http://localhost:5000](http://localhost:5000). Elle lit `mlflow.db` à la racine du dépôt, le fichier que le lanceur vient d'écrire, et ne dépend d'aucun conteneur. C'est aussi l'adresse par défaut du réglage `mlflow_tracking_uri`, donc celle que le reste du projet attend quand un serveur est en jeu plutôt qu'un fichier.

Le magasin est passé explicitement à la commande, alors qu'elle sait le deviner, et c'est le seul argument qui compte :

| Ce qui résout l'URI | Sans configuration | Quand un `./mlruns` existe |
| --- | --- | --- |
| `mlflow ui` | `sqlite:///mlflow.db` | bascule sur `./mlruns` |
| `src/tracking/client.py` | `sqlite:///mlflow.db` | `sqlite:///mlflow.db` |

Les deux résolutions ne sont pas les mêmes. Un répertoire `mlruns/` laissé par un autre outil suffirait à faire lire à l'interface un magasin que personne n'alimente, pendant que le traçage continue d'écrire à côté. L'écran serait vide et la conclusion serait que le traçage ne marche pas.

Le serveur n'écoute que sur `127.0.0.1`, ce qui est le défaut de `mlflow ui` et se change avec précaution : le magasin ne porte aucune authentification, et il expose le commit, le matériel et les chemins de checkpoints de chaque run.

Un magasin neuf n'affiche qu'une expérience `Default` vide. C'est l'état attendu ici : la base est créée à la première ouverture de l'interface, pas remplie par elle.

## Un échec de tracking ne fait pas échouer un run

Un serveur injoignable, un disque plein ou une incompatibilité de version transformeraient sinon six heures d'entraînement en plantage survenu après la prise de mesure. Chaque envoi passe par `log_safely`, qui signale l'échec sur la sortie d'erreur et rend la main.

La commande de renvoi, elle, sort en code 1 quand un envoi échoue : envoyer est la seule chose qu'elle fait.

## Commandes

```bash
# Tracé automatiquement par le lanceur
python -m src.experiments.run --config configs/experiments/scratch_100.yaml

# Sans tracking
python -m src.experiments.run --config configs/experiments/scratch_100.yaml --no-tracking

# Vers un serveur
python -m src.experiments.run --all --tracking-uri http://localhost:5000

# Renvoyer ce qui est déjà sur le disque
python -m src.tracking.log --all
python -m src.tracking.log --experiment scratch_10
```

Une expérience déclarée qui n'a jamais tourné est nommée dans le compte rendu avec son statut `NOT_RUN` et ne produit aucun run dans le magasin. Une ligne pour une expérience que personne n'a exécutée mettrait un run inventé à l'endroit précis où un lecteur va compter ce qui est fait.

Renvoyer un enregistrement crée un nouveau run dans le magasin. Rien n'est écrasé et rien n'est dédupliqué : mettre à jour silencieusement un run antérieur réécrirait ce que le magasin disait hier.

## État du magasin

Le magasin contient neuf runs, un par expérience déclarée, groupés sous l'expérience `syntra-summarization` et tous au statut `OK`. Aucun run `PARTIAL` ni `FAILED` n'y figure, la campagne n'en a produit aucun.

Les cinq runs `scratch_*` viennent de la campagne du 8 août 2026. Les quatre runs `pretrained_*` ont été refaits le 9 août sous la révision `df1b051c` de `t5-small` : le registre refuse une révision non épinglée, et il lit le bloc modèle de l'enregistrement, pas celui de la configuration courante.

Le magasin local est la base SQLite `mlflow.db` à la racine du dépôt. L'interface se sert en lui passant explicitement cette base, faute de quoi elle bascule sur `./mlruns` et affiche un magasin vide :

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Suivre un run pendant qu'il tourne

`src/tracking/live.py` ouvre le run avant l'entraînement et y pousse ce que la boucle mesure : `step_loss`, `step_running_loss`, `step_learning_rate` et `step_grad_norm` tous les `log_every_steps` pas, puis `epoch_train_loss` et `epoch_validation_loss` à chaque fin d'époque. L'enregistrement ferme ce même run à la fin, donc une expérience reste un run et non un run ouvert à côté d'un miroir terminé.

**Le flux et le résultat ne sont pas la même série.** Ce que la boucle envoie porte les préfixes `step_` et `epoch_` ; ce que l'enregistrement écrit en fermant s'appelle `train_loss` et `validation_loss` et vient du record seul. Les deux ne partagent pas d'abscisse, la première étant indexée par pas d'optimisation et la seconde par indice d'époque. Les confondre tracerait deux axes comme une seule courbe et laisserait lire une observation comme un résultat.

**Un run ouvert n'est pas un résultat.** Il ne porte aucun score tant qu'il n'est pas fermé, de la même façon qu'un run `PARTIAL` n'en porte pas. C'est la seule règle que le suivi relâche : le magasin peut contenir un run dont le dépôt n'a pas encore l'enregistrement. Ce qu'il ne relâche pas est l'origine des nombres finaux, qui reste l'enregistrement et rien d'autre.

**Un échec du suivi n'échoue pas le run.** Même contrat que le miroir, appliqué par appel. Un magasin qui devient injoignable en cours d'entraînement cesse d'être sollicité au lieu d'écrire un avertissement par pas mesuré.

`history.json`, lui, n'est toujours écrit qu'à la fin de l'entraînement. Voir la page [Entraînement](training.md).
