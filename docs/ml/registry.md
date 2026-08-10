# Registre de modèles

Section 20 du cahier des charges.

## Pourquoi

Le backend ne doit dépendre d'aucun chemin de fichier codé en dur. Un modèle est donc décrit par un document, pas par un emplacement : l'API demande `champion` et reçoit une identité, un endroit d'où lire les poids, et la mesure qui a justifié leur publication.

## Composants

```text
backend/app/registry/
├── models.py    ce qu'est une version, et ce qui la rend valide
├── base.py      le contrat dont dépend l'API
├── local.py     le registre stocké dans un répertoire
└── factory.py   la sélection du backend depuis les réglages

src/experiments/publish.py   ce qui remplit le registre à partir d'un run
```

`backend/app/registry/` n'importe pas torch. Résoudre `champion` coûte une lecture de fichier, ce qui permet à l'API de décider quoi charger avant de le payer, conformément à la section 27.

## Disposition sur le disque

```text
artifacts/registry/
├── index.json                  ce que sert chaque nom, où pointe chaque alias
├── scratch/
│   └── v1/
│       ├── version.json        identité, provenance et scores
│       └── weights.pt          le dictionnaire d'état, rien d'autre
└── pretrained/
    └── v1/
        ├── version.json        identité, provenance et scores
        └── weights.pt          le checkpoint fine-tuné, réduit de même
```

Les deux versions portent un fichier de poids parce que les deux viennent d'un checkpoint : `pretrained:v1` est un `t5-small` fine-tuné, donc des poids que le projet a produits. Une version publiée depuis la baseline zero-shot n'aurait, elle, que son `version.json` : ses poids se retrouvent par l'identifiant et la révision, et les recopier dupliquerait ce que le hub sert déjà.

`index.json` ne porte que ce qui ne se déduit pas : la version par défaut de chaque nom et la cible de chaque alias. Y lister aussi les versions créerait une seconde source de vérité, et un inventaire en désaccord avec les répertoires est le genre de dérive que rien ne détecte avant l'échec d'une résolution.

## Les quatre noms de la section 20

`scratch` et `pretrained` sont des noms de modèles. `champion` et `challenger` sont des alias de déploiement. Les deux se demandent de la même façon :

```python
registry.get_model("scratch")            # la version que ce nom sert
registry.get_model("scratch", "v2")      # une version précise
registry.get_model("champion")           # ce que l'alias désigne
```

Un modèle ne peut pas être enregistré sous un nom d'alias : la résolution deviendrait ambiguë.

Le côté de la comparaison décide du nom, pas le nom de l'expérience. `scratch_10` et `scratch_100` sont deux versions d'un même modèle, pas deux modèles.

## Les règles

**Une version identifie exactement des poids, ou ce n'est pas une version.** Un checkpoint local est identifié par son chemin et par l'empreinte SHA-256 calculée à la publication. Un modèle tiré de Hugging Face est identifié par son identifiant et sa révision.

**Une révision non épinglée est refusée.** La section 14 ne tolère cela qu'en dehors d'une expérience reportée, et une version enregistrée est l'inverse : c'est ce que l'API sert. Un identifiant sans révision se résout vers ce que le hub publie ce jour-là, donc les poids changeraient sans que la version change. La règle s'applique aussi à un checkpoint fine-tuné : les poids sont les nôtres, mais l'architecture dans laquelle ils sont chargés est téléchargée depuis le même identifiant.

**Une version publiée n'est jamais modifiée.** La publication alloue l'identifiant suivant et refuse d'écrire dans un répertoire qui existe déjà. Une version est ce que désigne une mesure ; changer son contenu changerait silencieusement ce à quoi se rapporte un score publié. Une version supprimée à la main ne libère pas son numéro non plus.

**Publier ne change pas ce qui est servi.** Une nouvelle version se range à côté des autres jusqu'à ce que quelque chose la promeuve. L'exception est la première version d'un nom, qui devient la version servie parce qu'il n'y avait rien à déplacer. Sans cette règle, copier un checkpoint dans le registre échangerait le modèle derrière une API en marche.

**Un alias désigne une version exacte, jamais la plus récente.** Si `champion` signifiait « la dernière version du modèle champion », publier changerait la production sans que personne ne promeuve quoi que ce soit.

**Les poids servis sont les poids mesurés.** L'empreinte enregistrée à la publication est vérifiée avant que le fichier ne soit rendu. La vérification est active par défaut : une garantie désactivée par défaut n'est pas une garantie.

**Une entrée de registre est une copie, pas une référence.** `runs/` est un espace de travail que la prochaine exécution de la même expérience écrase.

## Ce qui est publié, et ce qui ne l'est pas

Seul un run complet peut être publié. Un run `PARTIAL` ou `FAILED` ne porte pas de mesure ; le registre est l'endroit où un lecteur demande quel modèle a produit un résumé et ce qu'il valait, et une version dont la réponse est `PARTIAL` n'a pas de réponse.

Le checkpoint est réduit au dictionnaire d'état. Les moments de l'optimiseur, la position du scheduler et l'état des générateurs aléatoires existent pour qu'un run reprenne ; un modèle servi ne reprend jamais, et les transporter triplerait la taille de chaque version.

La mesure voyage avec les poids : `experiment`, `dataset_version`, `git_commit` et les scores ROUGE sont recopiés de l'enregistrement de run. Sans eux, une entrée de registre est un fichier anonyme.

## Commandes

```bash
# Publier un run complet
python -m src.experiments.publish --experiment scratch_100

# Publier et promouvoir en une fois
python -m src.experiments.publish --experiment scratch_100 --alias champion

# Publier une nouvelle version et la servir immédiatement
python -m src.experiments.publish --experiment scratch_100 --activate
```

`--activate` et `--alias` sont les deux seuls moyens de changer ce qui est servi. Revenir en arrière consiste à réactiver une version antérieure : elle est toujours là.

## Réglages

| Variable | Effet |
| --- | --- |
| `SYNTRA_REGISTRY_BACKEND` | `local` ou `mlflow` |
| `SYNTRA_REGISTRY_ROOT` | racine du registre local, `artifacts/registry` par défaut |

## État du registre

Les deux côtés de la comparaison sont publiés, chacun depuis le run à 100 % de sa famille :

| Nom | Version | Expérience | ROUGE-L | Sert par défaut |
| --- | --- | --- | --- | --- |
| `pretrained` | `v1` | `pretrained_ft_100` | 0,2295 | `v1` |
| `scratch` | `v1` | `scratch_100` | 0,1634 | `v1` |

| Alias | Désigne |
| --- | --- |
| `champion` | `pretrained:v1` |
| `challenger` | `scratch:v1` |

Le champion est le `t5-small` fine-tuné parce que c'est ce que la mesure dit : les intervalles bootstrap des deux versions sont disjoints, [0,2227 ; 0,2365] contre [0,1581 ; 0,1686]. Servir le modèle from scratch par défaut aurait fait répondre l'API avec le moins bon des deux, ce qu'aucune ligne du registre n'aurait signalé.

Le côté pré-entraîné a longtemps manqué, et il manquait pour une raison que le registre applique lui-même : l'enregistrement des runs `pretrained_*` portait `revision: null`. Épingler le fichier d'expérience ne suffisait pas, parce que la publication lit le bloc modèle de l'enregistrement et non celui de la configuration courante. Les quatre runs ont donc été refaits sous la révision `df1b051c49625cf57a3d0d8d3863ed4d13564fe4`.

## Ce qui n'est pas fait

**Le backend `mlflow` n'est pas écrit.** Le sélectionner lève au démarrage, avec le nom du réglage qui l'a produit. Se rabattre sur le registre local répondrait à la demande avec des poids venus d'un autre magasin, et l'API rapporterait un modèle dont la provenance n'est pas celle que le déploiement a demandée.

**Le chargeur de la section 27 arrive avec l'API.** Le registre résout une identité et un chemin ; construire un module torch à partir de cela appartient au lot suivant.
