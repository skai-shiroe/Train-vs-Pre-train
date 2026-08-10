# Conformité aux requirements

Cette page met en regard chaque exigence de l'énoncé et l'endroit du dépôt qui la satisfait. Elle sépare deux questions que la lecture d'un dépôt confond facilement : le code est-il capable de produire ce qui est demandé, et l'a-t-il produit.

La réponse courte : les quatre requirements techniques sont implémentés, couverts par les tests, et exécutés. Les neuf expériences déclarées portent toutes le statut `OK` : les cinq runs `scratch_*` viennent de la campagne du 8 août 2026, les quatre runs `pretrained_*` ont été refaits le 9 août sous une révision épinglée. Ce qui reste ouvert tient à l'outillage et non aux résultats, et la dernière section le dit sans l'atténuer.

## Technos suggérées

| Attendu | État | Où |
| --- | --- | --- |
| PyTorch, implémentation seq2seq | Fait | `src/models/scratch/` |
| `transformers` pour la baseline pré-entraînée | Fait | `src/models/pretrained/t5.py` |
| `rouge-score` | Fait | `src/metrics/rouge.py` |
| `sacrebleu` | Hors périmètre | voir ci-dessous |

**SacreBLEU n'est pas installé, et c'est le choix attendu.** L'énoncé demande BLEU pour la traduction ou ROUGE pour le résumé. La tâche retenue est le résumé automatique, donc la métrique est ROUGE. Installer SacreBLEU pour ne jamais l'appeler ajouterait une dépendance dans l'image d'entraînement sans ajouter une mesure. Le périmètre est verrouillé sur la [page d'accueil](index.md).

## Requirement 1 : Transformer encodeur-décodeur from scratch

Le modèle est écrit à la main. Aucun module Transformer prêt à l'emploi de PyTorch n'apparaît dans `src/models/scratch/` : ni `nn.Transformer`, ni `nn.TransformerEncoderLayer`, ni `nn.MultiheadAttention`, ni la fonction d'attention fusionnée de `torch.nn.functional`.

L'attention est calculée explicitement, produit scalaire, mise à l'échelle, masquage puis `softmax` :

```python
d_k = query.size(-1)
scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k**0.5)
scores = scores.masked_fill(~mask, masked_fill_value(scores.dtype))
weights = F.softmax(scores, dim=-1)
```

Le découpage en têtes, la projection de sortie, l'encodage positionnel sinusoïdal, les masques causaux et de padding, les couches d'encodeur et de décodeur sont chacun dans leur module. Le détail des choix est sur la page [Transformer from scratch](ml/transformer.md).

L'entraînement porte sur un sous-ensemble du corpus, comme demandé : `scratch_10`, `scratch_50` et `scratch_100` couvrent 2 000, 10 000 et 20 000 exemples du corpus de travail.

## Requirement 2 : pré-entraîné en zero-shot puis fine-tuné

`t5-small` est chargé par `T5ForConditionalGeneration.from_pretrained`. Les deux modes demandés sont deux expériences distinctes :

--8<-- "_generated/pretrained.md"

**Le fine-tuning porte sur tout le modèle.** Aucun paramètre n'est gelé : `src/models/pretrained/base.py` ne contient aucun `requires_grad = False`. Un fine-tuning qui n'entraînerait que la tête de sortie mesurerait autre chose que ce que l'énoncé demande, et rien dans le score ne le montrerait.

**Le sous-ensemble est le même que celui du modèle from scratch.** Les fichiers pré-entraînés et les fichiers from scratch pointent vers `configs/data/xsum.yaml`, déclarent les mêmes proportions, et sont évalués sur le même split `test` de 1 000 exemples. Le tokenizer T5 est partagé par les deux modèles. C'est ce qui rend l'écart attribuable au modèle plutôt qu'au protocole.

**La révision du hub est épinglée.** Les quatre fichiers `pretrained_*` fixent `revision: df1b051c49625cf57a3d0d8d3863ed4d13564fe4`. Les poids fine-tunés sont les nôtres, mais l'architecture dans laquelle ils sont chargés est téléchargée depuis le hub : sans révision fixée, elle change sans que rien dans le dépôt ne change. Les quatre runs `pretrained_*` ont été refaits sous cette révision, parce que le registre lit le bloc modèle de l'enregistrement et non celui de la configuration courante.

## Requirement 3 : ablation sur la taille du corpus

L'étude `dataset_size` regroupe sept runs, soit les deux familles de modèles aux trois proportions demandées, plus la référence zero-shot.

--8<-- "_generated/families.md"

Le zero-shot n'apparaît pas dans cette table. Ses poids ne dépendent pas de la taille du corpus d'entraînement, donc il donne une mesure et non trois ; son score figure dans la table du requirement 2 ci-dessus.

L'ablation répond à la question posée : **l'écart entre les deux familles ne se referme pas quand le corpus grandit.** Le modèle pré-entraîné fine-tuné sur 2 000 exemples devance le modèle from scratch entraîné sur 20 000.

Trois propriétés rendent la comparaison exploitable, et elles sont décrites en détail sur la page [Expériences et ablations](experiments/index.md) :

1. Les sous-ensembles sont emboîtés, donc la taille varie sans que la composition de l'échantillon varie.
2. Le budget d'époques est constant sur les trois proportions, donc une seule grandeur bouge.
3. L'agrégation refuse d'écrire un tableau dont deux runs n'ont pas été mesurés avec le même décodage et les mêmes réglages de ROUGE.

Une seconde ablation, sur la profondeur de l'encodeur et du décodeur, dépasse ce que l'énoncé exige. Elle est décrite au même endroit.

## Requirement 4 : métrique et analyse qualitative

ROUGE-1, ROUGE-2 et ROUGE-L sont calculés par `rouge-score`, l'implémentation de référence, jamais réécrite. ROUGE-L en F-mesure est la valeur portée par la courbe.

Deux garde-fous accompagnent la moyenne. Une prédiction vide vaut zéro et reste dans la moyenne, parce que retirer les documents sur lesquels un modèle a échoué relèverait sa moyenne pour avoir échoué. Le nombre de prédictions vides est reporté à côté du score, ce qui sépare un score faible d'un modèle cassé. La moyenne est bornée par un intervalle de confiance bootstrap, calculé sous la graine du run.

L'analyse qualitative est produite par chaque run et rassemblée entre les runs. La sélection retient le meilleur cas, le pire cas, et un tirage aléatoire dans le reste.

**Le tirage n'est pas un ornement.** Les deux extrêmes sont les deux exemples les moins représentatifs de la distribution : un modèle peut avoir un beau meilleur cas, un pire cas catastrophique, et un centre plat qui est ce qu'il fait réellement. La sélection est déterministe, ce qui rend l'analyse vérifiable par un tiers.

Les neuf sélections sont rassemblées dans `reports/results/qualitative_examples.json`, les mêmes documents de test pour les neuf modèles. Aucun des neuf n'a produit de résumé vide.

Ce que la lecture des exemples apporte et que le score ne dit pas : le zero-shot ne résume pas, il recopie. Ses sorties font en moyenne 36,4 mots contre 21,3 pour la référence, et son pire cas reprend un fragment d'habillage de la page source au lieu du contenu. Les modèles entraînés, from scratch comme fine-tunés, se calent entre 17,3 et 19,4 mots. C'est la mesure de ce que le fine-tuning apprend ici, le format plutôt que la langue.

Le détail est sur la page [Évaluation](ml/evaluation.md).

**Un entraînement est observable pendant qu'il tourne.** `src/tracking/live.py` ouvre le run MLflow avant la première époque et y pousse la perte, le taux d'apprentissage et la norme du gradient ; `main` appelle `logging.basicConfig`, donc la console affiche la même progression. `history.json` reste écrit à la fin, parce que c'est un artefact de run et non un flux.

## Reproduction par un tiers

Après clonage, quatre commandes suffisent : créer l'environnement, `make install`, puis `make reproduce MODE=quick` pour vérifier que la chaîne tourne sur ce poste, et `make reproduce MODE=full` pour rejouer la campagne. `src/experiments/reproduce.py` enchaîne les cinq étapes de la section 41, corpus compris, et s'arrête à la première qui échoue.

**Le mode `quick` ne produit aucun résultat, et c'est sa définition.** Il plafonne chaque run à deux pas d'optimisation et l'évalue sur huit documents, ce qui suffit à sortir les neuf enregistrements en `PARTIAL` : aucun tableau, aucune figure ne les reprend. Il écrit sous `reports/quick/`, ne touche ni au rapport ni aux neuf runs mesurés, et ne trace rien dans MLflow. Mesuré le 9 août 2026 sur ce poste, corpus déjà construit : cinq étapes, 51 secondes.

Le détail des deux modes, et ce que chacun garantit, est sur la page [Reproductibilité](reproducibility.md).
