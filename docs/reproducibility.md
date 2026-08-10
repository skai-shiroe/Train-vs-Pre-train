# Reproductibilité

Une expérience non reproductible n'est pas un résultat. Cette page décrit les mécanismes qui garantissent qu'un tiers obtient les mêmes chiffres.

## Graine unique

`src/utils/seed.py` expose une fonction unique appelée une seule fois par expérience, avant la construction des chargeurs de données et du modèle.

```python
from src.utils.seed import set_seed

state = set_seed(42)
```

Elle configure les quatre sources de hasard listées au cahier des charges :

```text
random
numpy
torch
torch.cuda
```

Elle fixe également `PYTHONHASHSEED`, pour que l'ordre des ensembles et des dictionnaires reste stable entre deux processus.

### Mode déterministe

Le mode déterministe est actif par défaut. Il force cuDNN en mode déterministe et désactive l'autotuner.

```python
set_seed(42, deterministic=False)  # runs exploratoires uniquement
```

Le mode non déterministe est plus rapide mais ses résultats ne sont pas publiables. Ne l'utiliser que pour des essais dont les chiffres ne sont pas reportés.

### Travailleurs du DataLoader

Avec `num_workers` supérieur à zéro, chaque processus doit être ré-ensemencé. `seed_worker` est passé au `DataLoader` :

```python
from torch.utils.data import DataLoader

from src.utils.seed import seed_worker

loader = DataLoader(dataset, num_workers=4, worker_init_fn=seed_worker)
```

## Corpus figé

Le corpus de travail est tiré une seule fois de XSum avec la graine 42, puis figé par un checksum versionné. Sa composition est décrite sur la [page d'accueil](index.md).

Règles d'intégrité :

```text
les splits validation et test sont identiques pour toutes les expériences
aucun exemple du split test n'apparaît dans un split d'entraînement
« 100 % » désigne toujours le corpus de travail, jamais XSum complet
```

## Matériel tracé

`src/utils/device.py` expose `describe_hardware()`, dont la sortie est enregistrée dans MLflow avec chaque run. Une durée d'entraînement sans le matériel associé n'est pas interprétable.

```text
torch_version
cuda_available
cuda_version
gpu_name
gpu_count
gpu_capability
```

La précision mixte n'est activée que sur GPU : sur CPU elle ajoute du coût de conversion sans gain.

## Contrôle préalable de l'environnement

Les mécanismes décrits sur cette page garantissent qu'une même chaîne donne les mêmes chiffres. Ils ne garantissent pas qu'un poste donné sait l'exécuter. Le notebook `notebooks/00_environment_check.ipynb` répond à cette seconde question, et il est le premier à lancer sur une machine neuve :

```bash
jupyter lab notebooks/00_environment_check.ipynb   # ou l'ouvrir dans l'éditeur
```

Il ne recopie aucune contrainte : la borne de version vient de `requires-python`, les dépendances des groupes de `pyproject.toml`, le matériel de `describe_hardware()`, les empreintes du manifeste du corpus, la provenance de `describe_provenance()`. Ce qu'il vérifie est donc ce que le projet déclare, et non une seconde source de vérité qui finirait par diverger.

Il classe ses constats en trois verdicts, et la distinction entre les deux derniers est la seule qui compte :

```text
OK       la vérification passe
ALERTE   un manque, qui ne bloque qu'une partie du travail
ECHEC    une incohérence, qui invalide un résultat ou en produira un faux
```

Un corpus absent est une alerte : il empêche d'entraîner, pas de servir l'API. Une version installée hors des bornes déclarées est un échec, parce que le plancher `torch>=2.13` répond à une vulnérabilité et non à une préférence. Trois vérifications sont des tests plutôt que des déclarations : les générateurs aléatoires doivent redonner le même tirage après un même ensemencement, les empreintes des splits sont recalculées avec la fonction qui les a écrites, et un produit matriciel est effectivement exécuté sur le périphérique retenu, un `import torch` réussi ne prouvant rien d'une installation CUDA incomplète.

Le notebook ne joint aucun service et n'écrit aucun fichier. Il tourne donc hors ligne, ce qui est la situation où il sert. Ce qu'il ne couvre pas est énoncé dans sa dernière section : le téléchargement de `t5-small` à sa révision épinglée, la mémoire nécessaire à un `batch_size` donné, et les hooks de qualité, qui portent sur le code quand lui porte sur ce qui l'entoure.

## Commande unique

Chaque expérience est décrite par un fichier de configuration et se rejoue par une commande unique :

```bash
python -m src.experiments.run --config configs/experiments/scratch_10.yaml
```

Le fichier est la seule chose que le lanceur lit. Le format et les règles qu'il fait respecter sont décrits sur la page [Expériences et ablations](experiments/index.md).

La graine du run est appliquée avant la construction des chargeurs et du modèle, et elle est aussi celle du rééchantillonnage qui borne la moyenne. Deux exécutions d'un même fichier donnent donc le même score, propriété vérifiée par un test.

## Chaîne complète

La chaîne entière se rejoue par une commande, dans l'un de deux modes :

```bash
make reproduce MODE=quick   # vérifie la mécanique, ne produit aucun résultat
make reproduce MODE=full    # la campagne, plusieurs heures de GPU
```

`src/experiments/reproduce.py` enchaîne les cinq étapes de la section 41, dans cet ordre :

```text
data            construit le corpus figé, ou signale celui déjà présent
experiments     joue les neuf fichiers de configs/experiments, évaluation comprise
tables          agrège les enregistrements en tableaux de registre et d'ablation
figures         trace les quatre figures de la section 18
documentation   régénère les tableaux Markdown du rapport et de la page Corpus
```

L'évaluation n'est pas une étape séparée : chaque expérience est mesurée par le run qui l'a produite, sur le split que son fichier déclare, ce qui garantit que les poids mesurés sont ceux que le run a sélectionnés.

**Une étape qui échoue arrête la chaîne.** Agréger les tableaux après une campagne interrompue publierait les chiffres de la campagne précédente sous le code courant. Le récapitulatif nomme l'étape fautive, conserve celles déjà faites, et le code de sortie est 1.

**L'étape `data` est sautée quand le corpus est là.** `make data` est idempotent : le reconstruire redonne les mêmes fichiers et le même `dataset_version`, au prix du téléchargement et de la tokenisation. `--rebuild-data` force la reconstruction. Le `dataset_version` est affiché dans les deux cas, parce que c'est lui qui dit sur quel corpus le reste de la chaîne a tourné.

### Ce que le mode `quick` vérifie, et ce qu'il ne mesure pas

Le mode `quick` est un test de tuyauterie, pas une petite campagne. Il rejoue les neuf expériences déclarées, chacune sur un budget plafonné :

| Grandeur | Valeur en mode `quick` |
| --- | --- |
| Pas d'optimisation | 2, via `max_steps` |
| Exemples d'entraînement | 32 |
| Exemples de validation | 16 |
| Documents de test évalués | 8, sur un split dont la taille réelle reste enregistrée |
| Faisceau et budget de génération | 1 faisceau, 16 tokens |

Ce plafonnement n'est pas un réglage de confort : `max_steps` marque le run comme plafonné, la limite d'évaluation le marque comme partiel, et les deux mécanismes existaient avant cette commande. **Les neuf enregistrements sortent en `PARTIAL`, donc aucun tableau ni aucune figure ne les reprend.** La section 44 traite un nombre présenté mais non mesuré comme un résultat fabriqué ; la réponse ici n'est pas de produire un petit nombre, c'est d'en produire un qu'aucun tableau n'accepte.

Le mode `quick` écrit tout sous `reports/quick/` et ses points de contrôle sous `runs/quick/`, y compris la page dans laquelle il injecte le tableau d'en-tête, qui est une copie du README. Il ne trace rien dans MLflow. Une vérification qui écraserait les neuf runs mesurés, ou les tableaux du rapport, coûterait plus qu'elle ne prouve.

Mesuré sur ce poste le 9 août 2026, corpus déjà construit, GPU RTX 5060 portable :

```text
data             SKIPPED    0.0 s   dataset_version 259d8397ce78
experiments      DONE      50.3 s   9 expériences, 9 PARTIAL
tables           DONE       0.1 s   4 tableaux
figures          DONE       0.9 s   4 figures
documentation    DONE       0.1 s   13 tableaux générés
```

Les résultats scientifiques proviennent exclusivement du mode `full`, qui n'applique aucun plafond et écrit là où le rapport, le site et le contrôle de dérive lisent.

## Traçabilité MLflow

Chaque run enregistre au minimum :

```text
experiment_id
git_commit
seed
dataset_version
dataset_percentage
model
hyperparameters
training_duration
ROUGE-1, ROUGE-2, ROUGE-L
checkpoint
hardware
```

Ces champs sont écrits dans `run.json`, puis envoyés à MLflow tels quels. Le commit est lu au moment du run et conservé dans l'enregistrement : le recalculer à l'envoi ferait porter à une mesure d'aujourd'hui le commit du jour où elle a été renvoyée. Le détail du mappage est sur la page [Traçabilité MLflow](ml/tracking.md).

## État de la reproductibilité

Les neuf expériences portent toutes le statut `OK` : les cinq runs `scratch_*` viennent de la campagne du 8 août 2026, les quatre runs `pretrained_*` ont été refaits le 9 août sous une révision épinglée. Les tableaux de `reports/results/` et les quatre figures de `reports/figures/` sont produits depuis ces neuf enregistrements.

Conformément à la règle d'intégrité scientifique, une expérience non exécutée porte le statut `NOT_RUN`, jamais une valeur estimée. Une expérience interrompue par une erreur porte `FAILED` et conserve son message. Aucun des neuf runs n'est dans ce cas.

**La reproductibilité par un tiers est acquise.** Les quatre expériences `pretrained_*` ont longtemps tourné avec `revision: null`, donc sur la révision `t5-small` que Hugging Face servait ce jour-là, sans que le dépôt en garde la trace. Elles ont été refaites sous la révision `df1b051c49625cf57a3d0d8d3863ed4d13564fe4`, désormais épinglée dans les quatre fichiers d'expérience. Le point de départ des poids est donc figé, et les mécanismes décrits sur cette page ne garantissent plus seulement que deux exécutions donnent le même score sur le même poste : ils le garantissent aussi pour un tiers, à distance de temps.
