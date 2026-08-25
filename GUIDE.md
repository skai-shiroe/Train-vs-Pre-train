# Guide de lecture

Ce guide suit un batch de données du fichier brut jusqu'au tableau de comparaison, en nommant à chaque étape le fichier qui fait le travail. Il ne remplace pas le [rapport](RAPPORT.md), qui présente les résultats : il explique le code qui les produit.

```text
corpus  ->  tokenisation  ->  Transformer  ->  loss  ->  backward  ->  optimiseur
                                                                          |
   comparaison  <-  MLflow  <-  métriques  <-  validation  <-  checkpoint <+
```

Le code est en anglais, ce guide en français. Chaque section renvoie au fichier concerné : le guide donne l'intention et la forme des tenseurs, le code donne le détail. En cas de doute, le code a raison.

Pour voir les tenseurs réels traverser le modèle, exécutez [notebooks/03_transformer_walkthrough.ipynb](notebooks/03_transformer_walkthrough.ipynb) : il reprend les étapes de la section 3 en affichant les formes à chaque passage.

Les valeurs numériques citées sont celles de `scratch_100`, l'expérience de référence, déclarée dans [configs/experiments/scratch_100.yaml](configs/experiments/scratch_100.yaml).

---

## 1. Du dataset au corpus figé

**Répertoire :** [src/data/](src/data/) — **Commande :** `make data`

CNN/DailyMail 3.0.0 compte 287 113 articles d'entraînement, 13 368 de validation et 11 490 de test. Le projet en fige 22 000 sous la graine 42 : 20 000 pour l'entraînement, 1 000 pour la validation, 1 000 pour le test. Le « 100 % » des ablations désigne ces 20 000 exemples, jamais le corpus complet.

La configuration `3.0.0` est épinglée dans [configs/data/cnn_dailymail.yaml](configs/data/cnn_dailymail.yaml). Le hub publie trois versions du corpus sous le même identifiant, et seule la 3.0.0 expose les entités nommées en clair : la laisser au choix du hub reviendrait à laisser un score changer sans qu'une ligne du dépôt bouge.

**Pourquoi ne pas tout prendre.** Le corpus complet multiplierait par 14 le coût de chaque run, et la campagne des neuf expériences avec lui. À 20 000 exemples, le modèle from scratch voit déjà ~3 400 fois moins de texte que `t5-small` en a vu en pré-entraînement ; à 287 113 il en verrait ~240 fois moins. L'écart change d'amplitude, pas de nature. La contrepartie est la couverture du vocabulaire, seule grandeur qui dépende du nombre d'exemples : mesurée à 76,0 % du tokenizer ici et à 77,0 % en portant le corpus à 30 000, elle croît trop lentement pour que la conclusion en dépende.

**Pourquoi un tirage plutôt qu'une troncature.** Garder les 20 000 premiers exemples reprendrait l'ordre du fichier amont, dont rien ne garantit qu'il soit aléatoire. La graine casse cet ordre de façon rejouable, et les empreintes du manifeste le vérifient. Sa valeur, 42, n'a aucune propriété : ce qui compte est qu'elle soit fixée dans la configuration. Les trois splits reçoivent 42, 43 et 44, de sorte que changer `train_size` ne déplace ni la validation ni le test.

| Étape | Fichier | Ce qui se passe |
| --- | --- | --- |
| Téléchargement | [download.py](src/data/download.py) | Récupère CNN/DailyMail depuis Hugging Face |
| Nettoyage | [preprocess.py](src/data/preprocess.py) | Écarte les documents trop courts ou trop longs, normalise l'Unicode |
| Tirage | [split.py](src/data/split.py) | Tire les trois splits sous graine, calcule leurs empreintes |
| Écriture | [dataset.py](src/data/dataset.py) | Écrit trois `.jsonl` et un `manifest.json` |
| Contrôle | [validate.py](src/data/validate.py) | Vérifie qu'aucun document ne fuit d'un split à l'autre |

Un exemple est un triplet minimal, [example.py](src/data/example.py) :

```python
Example(
    example_id="95175a26fb68f3a151d452f845ee0f41570e9984",
    source="L'article complet...",
    target="Les highlights de référence.",
)
```

Les highlights arrivent en plusieurs lignes, une puce par phrase. Le nettoyage les joint en une seule ligne, ce que le JSONL impose ; c'est aussi ce qui rend ROUGE-L plus sévère ici que le ROUGE-Lsum publié, point développé en section 5.

Deux propriétés comptent pour la suite.

**Le corpus est figé, pas régénéré.** `manifest.json` porte un `dataset_version`, empreinte des trois splits. Chaque run enregistre cette empreinte. Deux scores calculés sous des empreintes différentes ne se comparent pas, et le dépôt le sait.

**Les sous-ensembles d'ablation sont emboîtés.** 10 % est un préfixe de 50 %, lui-même préfixe de 100 %. Tirés indépendamment, un écart entre deux points de la courbe mélangerait l'effet de la taille et celui de la composition de l'échantillon.

---

## 2. Tokenisation, `Dataset`, `DataLoader`, batchs

**Fichiers :** [src/data/tokenize.py](src/data/tokenize.py), [src/data/dataset.py](src/data/dataset.py), [src/training/sampler.py](src/training/sampler.py)

### Le tokenizer

Un modèle ne lit pas du texte, il lit des entiers. Le tokenizer T5 découpe le texte en sous-mots et rend un identifiant par sous-mot. **Les deux modèles du projet partagent ce tokenizer** : même vocabulaire, même découpage, même troncature. Un écart de score ne peut donc venir que des modèles.

```text
"summarize: Three men have been arrested."

21603   10    5245    1076   43      118     10195       5     1
summarize  :   Three   men   have   been   arrested     .   </s>
```

Neuf identifiants pour sept mots et deux ponctuations. Sur cet exemple chaque mot tient en un sous-mot, ce qui n'est pas garanti : un mot rare est découpé en plusieurs morceaux, et c'est ce qui permet au vocabulaire de rester fini. Le `1` final est le token de fin de séquence. Le vocabulaire compte 32 100 entrées, et le corpus en fait apparaître 24 480.

Le préfixe `"summarize: "` vient de la convention T5, qui multiplexe les tâches par un préfixe textuel. Le modèle from scratch le reçoit aussi, pour que l'entrée soit strictement identique.

### Ce que produit un batch

[`encode_batch`](src/data/tokenize.py) rend un `EncodedBatch` de quatre tenseurs :

| Champ | Forme | Rôle |
| --- | --- | --- |
| `input_ids` | `(8, S)` | L'article, tronqué à 512 tokens |
| `attention_mask` | `(8, S)` | 1 sur les vrais tokens, 0 sur le padding |
| `target_ids` | `(8, T)` | Le résumé, tronqué à 128 tokens, padding conservé |
| `labels` | `(8, T)` | Le résumé, padding remplacé par `-100` |

`S` et `T` sont les longueurs du plus long élément du batch, pas les plafonds : le padding est dynamique.

**Pourquoi `target_ids` et `labels` séparément.** `target_ids` sert à construire l'entrée du décodeur par décalage (section 3.9), `labels` sert à calculer la loss. La valeur `-100` est celle que `cross_entropy` ignore : sans elle, le modèle serait entraîné à prédire du padding, et la loss récompenserait le silence.

**Les deux plafonds n'ont pas le même prix.** 512 tokens en source coupent 85 % des articles et laissent la moitié du texte hors de l'encodeur ; 128 tokens en cible n'en coupent que 5,3 %. Le premier est le compromis dominant du projet, détaillé en section 1 du rapport.

### Le sampler groupé par longueur

[`LengthGroupedSampler`](src/training/sampler.py) trie les exemples par longueur avant de former les batchs. La mesure est dans le fichier lui-même, obtenue par [scripts/measure_padding.py](scripts/measure_padding.py) :

```text
batch 8   mélangé : 3,8 % de padding      groupé : 1,0 %
```

**Sur ce corpus, le sampler rapporte peu, et il faut le dire.** 85 % des articles atteignent le plafond de 512 tokens, donc dès la taille de batch 4 tous les batchs sont à 512 : le padding dynamique s'est déjà effondré en padding fixe, mais ce qu'il laisse à récupérer ne fait que 3,8 %. Le groupement en reprend 2,8 points. Il est conservé parce qu'il coûte un tri par fenêtre et ne peut pas nuire, pas parce qu'il serait ici l'optimisation décisive — celle-ci serait d'allonger la troncature, et elle est bornée par le budget GPU.

Le groupement conserve un aléa entre les mega-batchs, pour que l'ordre change à chaque époque, et place le batch le plus large en tête : le pic mémoire de l'époque est payé au premier pas, donc une configuration qui ne tient pas dans les 8 Go échoue tout de suite plutôt qu'après vingt minutes.

---

## 3. Le Transformer, de bout en bout

**Répertoire :** [src/models/scratch/](src/models/scratch/) — un concept par fichier.

Configuration de référence : `d_model=256`, `num_heads=8`, donc `d_head=32`, 4 couches d'encodeur, 4 de décodeur, `d_ff=1024`. Vocabulaire T5 : 32 100 entrées. Total : 15,6 M paramètres, dont 52,7 % dans la seule table d'embedding.

Le tableau des formes, pour un batch de 8, un article de `S` tokens et un résumé de `T` tokens :

| Étape | Forme en sortie |
| --- | --- |
| `input_ids` | `(8, S)` |
| Embedding | `(8, S, 256)` |
| + encodage positionnel | `(8, S, 256)` |
| Projections Q, K, V | `(8, S, 256)` |
| Découpage en têtes | `(8, 8, S, 32)` |
| Scores d'attention | `(8, 8, S, S)` |
| Contexte | `(8, 8, S, 32)` |
| Recomposition des têtes | `(8, S, 256)` |
| Sortie de l'encodeur | `(8, S, 256)` |
| Sortie du décodeur | `(8, T, 256)` |
| Logits | `(8, T, 32100)` |

`(8, 8, S, S)` est le seul tenseur dont la taille croît comme le **carré** de la longueur, et c'est pourquoi la troncature à 512 tokens coûte ce qu'elle coûte.

**La mise à l'échelle par `sqrt(d_model)` n'est pas cosmétique.** La table est initialisée avec un écart-type de `d_model ** -0.5`, soit environ 0,0625 ici. L'encodage positionnel qu'on va lui ajouter vit dans `[-1, 1]`. Sans le facteur `sqrt(256) = 16`, le signal de position écraserait le signal de token.

**Le décalage à droite.** [`shift_target_right`](src/models/scratch/transformer.py) construit l'entrée du décodeur en décalant la cible d'une position et en préfixant le token de départ. La position `t` du décodeur contient donc le token `t - 1` de la cible : prédire le token `t` ne demande que ce que le masque causal autorise à voir. Le tokenizer T5 n'a pas de token de début de séquence, donc l'identifiant de padding joue ce rôle, exactement comme T5 lui-même.

**Poids liés.** Quand `tie_embeddings` est vrai, la projection réutilise la transposée de la table d'embedding. Cela économise `vocab_size * d_model` paramètres, soit 8,2 M ici, et régularise un modèle entraîné sur peu de données en forçant la vue d'entrée et la vue de sortie d'un token à s'accorder.

**La génération est une boucle.** À l'entraînement, tout le résumé est fourni d'un coup. À la génération, rien n'est fourni : le modèle produit un token, se le redonne, et recommence.

| Fonction | Ce qu'elle fait |
| --- | --- |
| `greedy_search` | Prend le token le plus probable à chaque pas |
| `beam_search` | Garde les `num_beams` hypothèses les plus probables, 4 ici |

---

## 4. L'entraînement

**Répertoire :** [src/training/](src/training/) — **Commandes :** `make train-scratch`, `make train-pretrained`

**Le lissage de labels** retire 10 % de la masse au token de référence et l'étale sur le vocabulaire. Le modèle est ainsi pénalisé s'il devient trop confiant, ce qui réduit le surapprentissage sur un petit corpus.

**La loss est pondérée par les tokens, pas par les batchs.** Un batch contient un nombre variable de tokens cibles : une moyenne simple sur les batchs pondérerait un résumé court comme un résumé long, et la courbe d'entraînement ne serait plus comparable à celle de validation.

**Le taux d'apprentissage monte puis descend.** Ici : taux maximal `3e-4`, `warmup_ratio=0.06`, soit 6 % des pas totaux consacrés à la montée, puis une décroissance linéaire.

**Une époque se termine par une validation**, qui calcule la loss de validation avec la même mesure pondérée par tokens, décide si le checkpoint est le meilleur, et alimente l'arrêt anticipé.

**Les poids évalués sont relus depuis le meilleur checkpoint.** Quand l'arrêt anticipé retient une époque antérieure, l'objet en fin d'entraînement n'est pas celui qui a le meilleur score. Le lanceur reconstruit le modèle depuis le checkpoint avant de le mesurer : le score publié appartient aux poids que le run a sélectionnés, et un checkpoint illisible échoue là plutôt que silencieusement.

---

## 5. Les métriques

**Fichiers :** [src/metrics/rouge.py](src/metrics/rouge.py), [src/evaluation/evaluator.py](src/evaluation/evaluator.py)

ROUGE mesure le recouvrement entre le résumé produit et le résumé de référence.

| Variante | Ce qu'elle compte |
| --- | --- |
| ROUGE-1 | Unigrammes communs |
| ROUGE-2 | Bigrammes communs |
| ROUGE-L | Plus longue sous-séquence commune |

ROUGE-L est la métrique rapportée : elle tolère les réordonnancements, ce qu'un résumé fait constamment.

**ROUGE-L, et non ROUGE-Lsum.** `rougeLsum` découpe la référence sur ses sauts de ligne et apparie chaque phrase séparément ; ROUGE-L exige une seule sous-séquence traversant toute la paire. Sur une référence CNN/DailyMail de trois à quatre phrases, la seconde est nettement plus sévère, et les chiffres publiés sur ce corpus sont des ROUGE-Lsum. Aucun score de ce projet ne s'y compare : toutes les comparaisons faites ici sont internes, même métrique et même jeu de test pour tous les modèles.

**L'ordre des arguments compte.** `RougeScorer.score` prend la référence en premier et la prédiction en second. Les intervertir laisse la F-mesure inchangée, donc l'erreur survit à tout test écrit sur F seule, pendant que précision et rappel échangent silencieusement leurs places. [`score_example`](src/metrics/rouge.py) fixe l'ordre une fois, et un test asymétrique l'épingle.

**Chaque score porte un intervalle de confiance.** Le calcul rééchantillonne 1 000 fois les documents notés et rend l'intervalle de percentiles à 95 %. Un score seul ne dit pas s'il diffère de son voisin, et deux intervalles qui se recouvrent ne permettent pas de conclure. Le rapport applique cette règle, y compris quand elle l'empêche de conclure.

L'évaluation est faite sur les mêmes 1 000 documents de test pour toutes les expériences, sans quoi les scores ne se compareraient pas.

---

## 6. MLflow : les runs et les modèles

**Répertoire :** [src/tracking/](src/tracking/) — **Interface :** `make mlflow-ui`, sur <http://localhost:5000>

MLflow répond à une question : quel run a produit ce score, avec quelle configuration, sur quel corpus, depuis quel commit, et où sont ses poids.

### Le magasin

Les métadonnées vont dans une base **PostgreSQL** dédiée. L'URI porte un mot de passe, donc elle ne vit pas dans le dépôt :

| Fichier | Rôle |
| --- | --- |
| `.env.example` | La forme de l'URI, versionnée, sans valeur |
| `.env` | La valeur, ignorée par git |
| [store.py](src/tracking/store.py) | Résout l'URI, l'environnement d'abord, le fichier ensuite |

Le lanceur affiche le magasin obtenu avant la première expérience, mot de passe masqué. Une campagne qui aurait tracé dans un fichier SQLite local au lieu de la base partagée le dit à la première seconde, plutôt que d'être découverte six heures plus tard.

Sans configuration, la résolution rend `None` et MLflow retombe sur un fichier SQLite local. C'est ce qui permet à `make reproduce` de tourner sur un clone frais sans base de données.

La distinction compte : ce repli répond à une **absence de configuration**, pas à un serveur injoignable. Un `.env` en place désigne la base PostgreSQL quoi qu'il arrive, et `make mlflow-ui` échoue sur un timeout si la machine qui l'héberge est éteinte. Un run, lui, survit à ce cas — voir *Un échec de tracking ne fait jamais échouer un run* plus bas — et `python -m src.tracking.log --all` renvoie après coup ce qui n'a pas pu partir.

### Ce qui est enregistré, quand

| Moment | Ce qui part vers MLflow | Fichier |
| --- | --- | --- |
| Avant l'entraînement | Ouverture du run, sous le nom de l'expérience | [live.py](src/tracking/live.py) |
| Pendant | `step_loss`, `step_learning_rate`, `step_grad_norm`, puis `epoch_train_loss` et `epoch_validation_loss` | [live.py](src/tracking/live.py) |
| Après l'évaluation | Le modèle mesuré, et son entrée au registre | [model.py](src/tracking/model.py) |
| À la fermeture | Paramètres, métriques finales, tags, artefacts | [payload.py](src/tracking/payload.py) |

**Paramètres** : ce que l'expérience a déclaré et qu'un rejeu devrait répéter — graine, proportion de corpus, architecture, hyperparamètres d'entraînement et de décodage.

**Métriques** : ROUGE-1, ROUGE-2, ROUGE-L, durée d'entraînement, meilleure loss de validation.

**Tags** : où le run a tourné — GPU, version de torch, commit git, propreté de l'arbre de travail.

**Artefacts** : `run.json`, `metrics.json`, `history.json`, `qualitative.json`, et le modèle.

### Les modèles

Un modèle par run complet, dans la saveur qui lui correspond :

| Famille | Saveur MLflow | Ce que ça embarque |
| --- | --- | --- |
| `t5-small` | `mlflow.transformers` | Poids, tokenizer, configuration de génération |
| from scratch | `mlflow.pytorch` | L'objet sérialisé, plus `src` via `code_paths` |

Chacun est enregistré sous `syntra-<expérience>` et se recharge par ce nom :

```python
import mlflow
model = mlflow.pytorch.load_model("models:/syntra-scratch_100/1")
```

Trois règles, écrites dans [model.py](src/tracking/model.py) :

**La base ne porte aucun poids.** PostgreSQL stocke les métadonnées et un pointeur ; les fichiers vont sous `MLFLOW_ARTIFACT_ROOT`. Compter environ 1,5 Go pour une campagne complète.

**Ce sont les poids évalués qui sont déposés**, relus depuis le meilleur checkpoint, pas l'objet en fin d'entraînement. Quand l'arrêt anticipé a retenu une époque antérieure, les deux diffèrent.

**Seul un run `OK` entre au registre.** Un run `PARTIAL` a vu deux pas d'optimisation et huit documents de test : ses poids existent et ne veulent rien dire, et une entrée au registre est exactement ce que quelqu'un recharge plus tard sans lire le statut à côté.

### Trois règles qui expliquent le reste du code

**Un échec de tracking ne fait jamais échouer un run.** Le résultat d'une expérience est l'enregistrement sur disque ; le magasin en est un miroir. Un serveur injoignable ne doit pas transformer six heures d'entraînement en plantage après que la mesure a été prise. Tout passe par `log_safely` et `log_model_safely`.

**Rien n'est enregistré qui n'ait été écrit d'abord.** Le lanceur écrit `run.json`, puis le trace. Le magasin ne contient donc jamais un run dont le dépôt n'a pas trace. Les séries live sont l'exception assumée, et elles portent d'autres noms : `step_*` et `epoch_*` sont ce que la boucle a observé, les métriques finales viennent de l'enregistrement seul.

**Le tracking se désactive, jamais ne se simule.** `--no-tracking` sélectionne [`NullTracker`](src/tracking/client.py), qui renvoie `None` et le dit.

### Rejouer une expérience

Un run porte son `git_commit`, son `dataset_version` et sa configuration complète. Reproduire consiste à revenir au commit, vérifier que l'empreinte du corpus correspond, et relancer :

```bash
python -m src.experiments.run --config configs/experiments/scratch_100.yaml
python -m src.tracking.log --all      # renvoie des enregistrements déjà écrits
python -m src.tracking.purge --all    # compte les runs du magasin, les supprime avec --yes
```

---

## 7. Comparer les expérimentations

**Répertoire :** [src/experiments/](src/experiments/) — le seul module que ce guide ne détaille pas.

Il porte la machinerie de campagne : lecture des fichiers d'expérience, boucle sur les neuf runs, ablations, tableaux, figures. Ce n'est pas de l'apprentissage automatique, c'est de l'orchestration, et on peut comprendre tout le reste sans l'ouvrir.

Ce qu'il faut en savoir tient en quatre commandes :

| Commande | Effet |
| --- | --- |
| `make reproduce MODE=quick` | Joue toute la chaîne en une minute, plafonnée. Ne produit aucun résultat, et le dit |
| `make reproduce MODE=full` | Rejoue la campagne réelle, plusieurs heures |
| `make ablation` | Rejoue les deux ablations et écrit leurs tableaux |
| `make report-sync` | Régénère les tableaux du rapport depuis les enregistrements |

Un enregistrement porte un statut, qui décide de son entrée dans les tableaux :

| Statut | Sens |
| --- | --- |
| `OK` | Mesure complète, entre dans les tableaux et au registre de modèles |
| `PARTIAL` | Run plafonné ou évalué sur un sous-ensemble. N'entre dans aucun tableau |
| `FAILED` | L'expérience a levé. L'erreur est enregistrée |
| `NOT_RUN` | Déclarée, jamais exécutée |

L'agrégation ne lit que les enregistrements `OK` porteurs d'une évaluation, condition écrite dans [`record.py`](src/experiments/record.py).

---

## 8. Par où commencer

Dans cet ordre, en une soirée :

0. `make kernel` — enregistrer le kernel du dépôt, sans quoi les carnets tournent sur l'interpréteur du PATH. Sous PowerShell, `.\make.ps1 kernel`.
1. `notebooks/00_environment_check.ipynb` — ce poste peut-il exécuter la chaîne, et sur quoi.
2. `make data` — construire le corpus, une fois.
3. [notebooks/01_eda_cnn_dailymail.ipynb](notebooks/01_eda_cnn_dailymail.ipynb) — les mesures qui fixent les plafonds et le budget de décodage.
4. [notebooks/03_transformer_walkthrough.ipynb](notebooks/03_transformer_walkthrough.ipynb) — voir un batch réel traverser le modèle, forme par forme.
5. `notebooks/02_training.ipynb` en `MODE = "quick"` — rejouer les neuf expériences en une minute et voir la mécanique de bout en bout.
