# Guide de lecture

Ce guide suit un batch de données du fichier brut jusqu'au tableau de comparaison, en nommant à chaque étape le fichier qui fait le travail. Le [rapport](RAPPORT.md) présente les résultats, ce guide explique le code qui les produit.

```text
corpus  ->  tokenisation  ->  Transformer  ->  loss  ->  backward  ->  optimiseur
                                                                          |
   comparaison  <-  MLflow  <-  métriques  <-  validation  <-  checkpoint <+
```

Le code est en anglais, ce guide en français. Chaque section renvoie au fichier concerné : le guide donne l'intention et la forme des tenseurs, le code donne le détail.

Pour voir les tenseurs réels traverser le modèle, exécutez [notebooks/03_transformer_walkthrough.ipynb](notebooks/03_transformer_walkthrough.ipynb) : il reprend les étapes de la section 3 en affichant les formes à chaque passage.

Les valeurs numériques citées sont celles de `scratch_100`, l'expérience de référence, déclarée dans [configs/experiments/scratch_100.yaml](configs/experiments/scratch_100.yaml).

---

## 1. Du dataset au corpus figé

**Répertoire :** [src/data/](src/data/) · **Commande :** `make data`

CNN/DailyMail 3.0.0 compte 287 113 articles d'entraînement, 13 368 de validation et 11 490 de test. Le projet en fige 22 000 sous la graine 42 : 20 000 pour l'entraînement, 1 000 pour la validation, 1 000 pour le test. Le « 100 % » des ablations désigne ces 20 000 exemples, jamais le corpus complet.

La configuration `3.0.0` est épinglée dans [configs/data/cnn_dailymail.yaml](configs/data/cnn_dailymail.yaml). Le hub publie trois versions du corpus sous le même identifiant, et seule la 3.0.0 expose les entités nommées en clair : la laisser au choix du hub reviendrait à laisser un score changer sans qu'une ligne du dépôt bouge.

**Pourquoi ne pas tout prendre.** Le corpus complet multiplierait par 14 le coût de chaque run, et la campagne des douze expériences avec lui. À 20 000 exemples, un modèle parti de zéro voit déjà ~3 400 fois moins de texte que `t5-small` en a vu en pré-entraînement ; à 287 113 il en verrait ~240 fois moins. L'écart change d'amplitude, pas de nature.

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

**Le corpus est figé, pas régénéré.** `manifest.json` porte un `dataset_version`, empreinte des trois splits. Chaque run enregistre cette empreinte. Deux scores calculés sous des empreintes différentes ne se comparent pas.

**Les sous-ensembles d'ablation sont emboîtés.** 10 % est un préfixe de 50 %, lui-même préfixe de 100 %. Tirés indépendamment, un écart entre deux points de la courbe mélangerait l'effet de la taille et celui de la composition de l'échantillon.

---

## 2. Tokenisation, `Dataset`, `DataLoader`, batchs

**Fichiers :** [src/data/tokenize.py](src/data/tokenize.py), [src/data/dataset.py](src/data/dataset.py), [src/training/sampler.py](src/training/sampler.py)

### Le tokenizer

Un modèle ne lit pas du texte, il lit des entiers. Le tokenizer T5 découpe le texte en sous-mots et rend un identifiant par sous-mot. **Les trois familles du projet partagent ce tokenizer** : même vocabulaire, même découpage, même troncature. Un écart de score ne peut donc venir que des modèles.

```text
"summarize: Three men have been arrested."

21603   10    5245    1076   43      118     10195       5     1
summarize  :   Three   men   have   been   arrested     .   </s>
```

Neuf identifiants pour sept mots et deux ponctuations. Sur cet exemple chaque mot tient en un sous-mot, ce qui n'est pas garanti : un mot rare est découpé en plusieurs morceaux, et c'est ce qui permet au vocabulaire de rester fini. Le `1` final est le token de fin de séquence. Le vocabulaire compte 32 100 entrées, et le corpus en fait apparaître 24 480.

Le préfixe `"summarize: "` vient de la convention T5, qui multiplexe les tâches par un préfixe textuel. Le Transformer écrit à la main le reçoit aussi, pour que l'entrée soit strictement identique.

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

**Sur ce corpus, le sampler rapporte peu.** 85 % des articles atteignent le plafond de 512 tokens, donc dès la taille de batch 4 tous les batchs sont à 512 : le padding dynamique s'est déjà effondré en padding fixe, mais ce qu'il laisse à récupérer ne fait que 3,8 %. Le groupement en reprend 2,8 points. Il est conservé parce qu'il coûte un tri par fenêtre et ne peut pas nuire, pas parce qu'il serait ici l'optimisation décisive ; celle-ci serait d'allonger la troncature, et elle est bornée par le budget GPU.

Le groupement conserve un aléa entre les mega-batchs, pour que l'ordre change à chaque époque, et place le batch le plus large en tête : le pic mémoire de l'époque est payé au premier pas, donc une configuration qui ne tient pas dans les 8 Go échoue tout de suite plutôt qu'après vingt minutes.

---

## 3. Les Transformers, de bout en bout

Trois familles, deux implémentations. Les fichiers `scratch_*` construisent le Transformer de [src/models/scratch/](src/models/scratch/). Les fichiers `random_t5_*` et `pretrained_*` construisent la même classe `T5ForConditionalGeneration`, depuis la même configuration `t5-small` épinglée ; `T5Summarizer.load_random_model` la bâtit sans charger ses poids, `load_model` charge les poids pré-entraînés de la même révision, et c'est la seule différence entre ces deux branches.

| Famille | Implémentation | Paramètres |
| --- | --- | --- |
| `scratch_*` | [src/models/scratch/](src/models/scratch/), un concept par fichier | 60 575 744 |
| `random_t5_*` | `T5ForConditionalGeneration`, poids tirés au hasard | 60 506 624 |
| `pretrained_*` | `T5ForConditionalGeneration`, poids du hub | 60 506 624 |

### 3.1 Le Transformer écrit à la main

**Répertoire :** [src/models/scratch/](src/models/scratch/), un concept par fichier.

Configuration de référence : `d_model=512`, `num_heads=8`, donc `d_head=64`, 6 couches d'encodeur, 6 de décodeur, `d_ff=2048`. Vocabulaire T5 : 32 100 entrées. Total : 60,6 M paramètres, dont 27,1 % dans la seule table d'embedding. La forme est celle de `t5-small`, pour que le budget de paramètres soit le même de part et d'autre.

Le tableau des formes, pour un batch de 8, un article de `S` tokens et un résumé de `T` tokens :

| Étape | Forme en sortie |
| --- | --- |
| `input_ids` | `(8, S)` |
| Embedding | `(8, S, 512)` |
| + encodage positionnel | `(8, S, 512)` |
| Projections Q, K, V | `(8, S, 512)` |
| Découpage en têtes | `(8, 8, S, 64)` |
| Scores d'attention | `(8, 8, S, S)` |
| Contexte | `(8, 8, S, 64)` |
| Recomposition des têtes | `(8, S, 512)` |
| Sortie de l'encodeur | `(8, S, 512)` |
| Sortie du décodeur | `(8, T, 512)` |
| Logits | `(8, T, 32100)` |

**Pourquoi la table est mise à l'échelle par `sqrt(d_model)`.** La table est initialisée avec un écart-type de `d_model ** -0.5`, soit environ 0,044 ici. L'encodage positionnel qu'on va lui ajouter vit dans `[-1, 1]`. Sans le facteur `sqrt(512) ≈ 22,6`, le signal de position écraserait le signal de token.

**Le décalage à droite.** [`shift_target_right`](src/models/scratch/transformer.py) construit l'entrée du décodeur en décalant la cible d'une position et en préfixant le token de départ. La position `t` du décodeur contient donc le token `t - 1` de la cible : prédire le token `t` ne demande que ce que le masque causal autorise à voir. Le tokenizer T5 n'a pas de token de début de séquence, donc l'identifiant de padding joue ce rôle, exactement comme T5 lui-même.

**Poids liés.** Quand `tie_embeddings` est vrai, la projection réutilise la transposée de la table d'embedding. Cela économise `vocab_size * d_model` paramètres, soit 16,4 M ici, et régularise un modèle entraîné sur peu de données en forçant la vue d'entrée et la vue de sortie d'un token à s'accorder.

**La génération est une boucle.** À l'entraînement, tout le résumé est fourni d'un coup. À la génération, rien n'est fourni : le modèle produit un token, se le redonne, et recommence.

| Fonction | Ce qu'elle fait |
| --- | --- |
| `greedy_search` | Prend le token le plus probable à chaque pas |
| `beam_search` | Garde les `num_beams` hypothèses les plus probables, 4 ici |

**À la génération, une seule ligne de logits est calculée.** Le décodeur est rejoué sur tout le préfixe à chaque pas (il n'y a pas de cache de clés-valeurs), mais la boucle ne lit que la dernière position. Projeter les autres construit le seul tenseur du modèle dont la dernière dimension est le vocabulaire : à 8 documents, 4 faisceaux et 128 tokens générés, `(32, 128, 32100)` pèse 526 Mio pour en utiliser 4. Le préfixe grandissant d'un token par pas, l'allocateur de torch finit par détenir un bloc de chaque taille intermédiaire. Mesuré lors de la campagne d'août 2026, sur le checkpoint `scratch_50` d'alors et 32 documents : 105,9 s et 11,90 Gio réservés en projetant tout, 4,4 s et 0,30 Gio en ne projetant que la dernière position, pour des résumés identiques au mot près. Sur une carte de 8 Gio la première version ne tient pas, et le pilote Windows la fait déborder en mémoire système plutôt que d'échouer : c'est ainsi qu'une évaluation de deux minutes en a pris quatre-vingt-quinze. `Decoder.forward` prend donc `last_position_only`, que les trois boucles de [generation.py](src/models/scratch/generation.py) passent.

### 3.2 Les deux `t5-small`

**Répertoire :** [src/models/pretrained/](src/models/pretrained/) : `base.py` porte tout ce qui ne dépend pas de l'architecture, `t5.py` ce que T5 impose. Le carnet [03_transformer_walkthrough.ipynb](notebooks/03_transformer_walkthrough.ipynb) fait traverser un vrai batch à ce modèle, forme par forme, et vérifie au passage qu'un calcul refait à la main redonne exactement ce que le modèle produit.

Configuration de `t5-small` : `d_model=512`, `num_heads=8`, donc `d_kv=64`, 6 couches d'encodeur, 6 de décodeur, `d_ff=2048`, table de 32 128 entrées.

Le tableau des formes, pour un batch de 8, un article de `S` tokens et un résumé de `T` tokens :

| Étape | Forme en sortie |
| --- | --- |
| `input_ids` | `(8, S)` |
| Embedding | `(8, S, 512)` |
| `T5LayerNorm` | `(8, S, 512)` |
| Projections Q, K, V | `(8, S, 512)` |
| Découpage en têtes | `(8, 8, S, 64)` |
| Scores d'attention | `(8, 8, S, S)` |
| Biais de position relative ajouté aux scores | `(1, 8, S, S)` |
| Contexte | `(8, 8, S, 64)` |
| Recomposition des têtes | `(8, S, 512)` |
| Sortie de l'encodeur | `(8, S, 512)` |
| Sortie du décodeur | `(8, T, 512)` |
| Logits | `(8, T, 32128)` |

`(8, 8, S, S)` est le seul tenseur dont la taille croît comme le **carré** de la longueur, dans les deux implémentations : c'est lui qui fixe le prix de la troncature à 512 tokens.

**La position n'est pas dans le vecteur de token.** T5 n'ajoute aucun encodage positionnel à l'embedding, et ne met pas non plus la table à l'échelle par `sqrt(d_model)`. La position entre plus loin, comme un biais appris par tête et par seau de distance relative, ajouté aux scores d'attention. Un seul bloc de chaque pile porte ce biais et les autres réutilisent le sien : la notion de position est apprise une fois pour toute la pile. Conséquence pratique, la longueur maximale n'est pas une propriété de l'architecture, seulement du budget mémoire.

**Les scores ne sont pas divisés par `sqrt(d_k)`.** T5 absorbe ce facteur dans l'initialisation de ses projections. Une réimplémentation qui le remettrait obtiendrait des poids d'attention différents de ceux du modèle du hub, alors que rien dans les formes ne le signalerait.

**Le décalage à droite.** `prepare_decoder_input_ids_from_labels` construit l'entrée du décodeur en décalant la cible d'une position et en préfixant le token de départ ; les `-100` des labels y redeviennent du padding, que l'entrée du décodeur ne sait pas ignorer. La position `t` du décodeur contient donc le token `t - 1` de la cible : prédire le token `t` ne demande que ce que le masque causal autorise à voir. Le tokenizer T5 n'a pas de token de début de séquence, donc l'identifiant de padding joue ce rôle. Passer `labels` au modèle suffit, il fait ce décalage lui-même.

**Poids liés.** `t5-small` déclare `tie_word_embeddings` : la table de l'encodeur, celle du décodeur et la projection de sortie sont un seul tenseur en mémoire. L'économie vaut `vocab_size * d_model`, soit 16,45 M de paramètres, et force la vue d'entrée et la vue de sortie d'un token à s'accorder. La projection multiplie l'état par `d_model ** -0.5` avant de l'appliquer, pour compenser l'échelle de la table qu'elle réutilise.

**La génération est une boucle.** À l'entraînement, tout le résumé est fourni d'un coup. À la génération, rien n'est fourni : le modèle produit un token, se le redonne, et recommence. [`PretrainedSummarizer.generate_ids`](src/models/pretrained/base.py) délègue cette boucle à `generate` de transformers, qui tient le cache de clés-valeurs et ne projette que la dernière position, puis retire le token de départ que tout encodeur-décodeur rend en position zéro.

| Réglage de décodage | Ce qu'il fait |
| --- | --- |
| `num_beams=1` | Prend le token le plus probable à chaque pas |
| `num_beams=4` | Garde les quatre hypothèses les plus probables, réglage des expériences |

**Le seul point où le décodage peut fausser la comparaison est la configuration.** Les trois familles lisent la même [`GenerationConfig`](src/models/generation.py), écrite dans le fichier d'expérience, et [`src/experiments/ablation.py`](src/experiments/ablation.py) refuse de construire un tableau à partir de runs dont les réglages diffèrent.

---

## 4. L'entraînement

**Répertoire :** [src/training/](src/training/) · **Commandes :** `make train-scratch`, `make train-random`, `make train-pretrained`

**Le lissage de labels** retire 10 % de la masse au token de référence et l'étale sur le vocabulaire. Le modèle est ainsi pénalisé s'il devient trop confiant, ce qui réduit le surapprentissage sur un petit corpus.

**La loss est pondérée par les tokens, pas par les batchs.** Un batch contient un nombre variable de tokens cibles : une moyenne simple sur les batchs pondérerait un résumé court comme un résumé long, et la courbe d'entraînement ne serait plus comparable à celle de validation.

**Le taux d'apprentissage monte puis descend.** Ici : taux maximal `3e-4`, `warmup_ratio=0.06`, soit 6 % des pas totaux consacrés à la montée, puis une décroissance linéaire.

**Une époque se termine par une validation**, qui calcule la loss de validation avec la même mesure pondérée par tokens, décide si le checkpoint est le meilleur, et alimente l'arrêt anticipé.

**Les poids évalués sont relus depuis le meilleur checkpoint.** Quand l'arrêt anticipé retient une époque antérieure, l'objet en fin d'entraînement n'est pas celui qui a le meilleur score. Le lanceur reconstruit le modèle depuis le checkpoint avant de le mesurer : le score publié appartient aux poids que le run a sélectionnés, et un checkpoint illisible échoue là plutôt que silencieusement.

**La mémoire du GPU est rendue avant l'évaluation.** Relire le checkpoint construit un second modèle, qui arrive sur la carte pendant que celui de l'entraînement, son optimiseur et les blocs que l'allocateur de torch garde en cache y sont encore. [`release_accelerator`](src/utils/device.py) tourne entre les deux, comme il tourne déjà entre deux expériences d'une campagne. Il collecte avant de vider : un modèle, son optimiseur et son scheduler se référencent mutuellement, et sans passage du ramasse-miettes le cache rendrait des blocs encore détenus. Ce n'est pas ce qui remplissait la carte pendant les évaluations de la campagne (la cause était la projection du décodeur, section 3), mais deux modèles résidents à la fois sur 8 Gio restent deux de trop.

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

**Chaque score porte un intervalle de confiance.** Le calcul rééchantillonne 1 000 fois les documents notés et rend l'intervalle de percentiles à 95 %. Un score seul ne dit pas s'il diffère de son voisin, et deux intervalles qui se recouvrent ne permettent pas de conclure.

L'évaluation est faite sur les mêmes 1 000 documents de test pour toutes les expériences, sans quoi les scores ne se compareraient pas.

---

## 6. MLflow : les runs et les modèles

**Répertoire :** [src/tracking/](src/tracking/) · **Interface :** `make mlflow-ui`, sur <http://localhost:5000>

MLflow répond à une question : quel run a produit ce score, avec quelle configuration, sur quel corpus, depuis quel commit, et où sont ses poids.

### Le magasin

Les métadonnées vont dans une base **PostgreSQL** dédiée. L'URI porte un mot de passe, donc elle ne vit pas dans le dépôt :

| Fichier | Rôle |
| --- | --- |
| `.env.example` | La forme de l'URI, versionnée, sans valeur |
| `.env` | La valeur, ignorée par git |
| [store.py](src/tracking/store.py) | Résout l'URI, l'environnement d'abord, le fichier ensuite |

Le lanceur affiche le magasin obtenu avant la première expérience, mot de passe masqué. Une campagne qui aurait tracé dans un fichier SQLite local au lieu de la base partagée le dit à la première seconde.

Sans configuration, la résolution rend `None` et MLflow retombe sur un fichier SQLite local. C'est ce qui permet à `make reproduce` de tourner sur un clone frais sans base de données.

La distinction compte : ce repli répond à une **absence de configuration**, pas à un serveur injoignable. Un `.env` en place désigne la base PostgreSQL quoi qu'il arrive, et `make mlflow-ui` échoue sur un timeout si la machine qui l'héberge est éteinte. Un run, lui, survit à ce cas (voir *Un échec de tracking ne fait jamais échouer un run* plus bas), et `python -m src.tracking.log --all` renvoie après coup ce qui n'a pas pu partir.

**Vider le magasin avant de rejouer.** Une campagne rejouée écrase ses enregistrements sur le disque, mais elle s'ajoute dans le magasin : deux réponses par expérience, et rien dans l'interface ne dit laquelle le rapport cite. `python -m src.tracking.purge --all` compte ce qu'il y a, `--yes` le supprime ; sans lui la commande ne fait que lister, parce qu'une fois `reports/results/` effacé le magasin est le seul endroit où une mesure existe encore. Deux étapes se cachent derrière la suppression : `delete_run` marque le run et le sort de l'interface sans toucher ni à ses lignes ni à ses artefacts, `mlflow gc` les enlève. [purge.py](src/tracking/purge.py) enchaîne les deux, signale un `gc` qui a échoué plutôt que de le couvrir d'un code de sortie nul, et laisse tranquille ce qui est déjà marqué : c'est exactement ce qu'une collecte échouée demande de relancer.

### Ce qui est enregistré, quand

| Moment | Ce qui part vers MLflow | Fichier |
| --- | --- | --- |
| Avant l'entraînement | Ouverture du run, sous le nom de l'expérience | [live.py](src/tracking/live.py) |
| Pendant | `step_loss`, `step_learning_rate`, `step_grad_norm`, puis `epoch_train_loss` et `epoch_validation_loss` | [live.py](src/tracking/live.py) |
| Après l'évaluation | Le modèle mesuré, et son entrée au registre | [model.py](src/tracking/model.py) |
| À la fermeture | Paramètres, métriques finales, tags, artefacts | [payload.py](src/tracking/payload.py) |

**Paramètres** : ce que l'expérience a déclaré et qu'un rejeu devrait répéter, à savoir graine, proportion de corpus, architecture, hyperparamètres d'entraînement et de décodage.

**Métriques** : ROUGE-1, ROUGE-2, ROUGE-L, durée d'entraînement, meilleure loss de validation.

**Tags** : où le run a tourné, à savoir GPU, version de torch, commit git, propreté de l'arbre de travail.

**Artefacts** : `run.json`, `metrics.json`, `history.json`, `qualitative.json`, et le modèle.

### Les modèles

Un modèle par run complet, dans la saveur qui lui correspond : `mlflow.transformers` pour les deux branches `t5-small`, qui embarque les poids, le tokenizer et la configuration de génération ; `mlflow.pytorch` pour le Transformer écrit à la main, qui n'a pas de saveur à lui et voyage avec `src` par `code_paths`.

Chacun est enregistré sous `syntra-<expérience>` et se recharge par ce nom :

```python
import mlflow
model = mlflow.pytorch.load_model("models:/syntra-scratch_100/1")
```

Ce que [model.py](src/tracking/model.py) impose :

**La base ne porte aucun poids.** PostgreSQL stocke les métadonnées et un pointeur ; les fichiers vont sous `MLFLOW_ARTIFACT_ROOT`. Compter environ 1,5 Go pour une campagne complète.

**Ce sont les poids évalués qui sont déposés**, relus depuis le meilleur checkpoint, pas l'objet en fin d'entraînement. Quand l'arrêt anticipé a retenu une époque antérieure, les deux diffèrent.

**Seul un run `OK` entre au registre.** Un run `PARTIAL` a vu deux pas d'optimisation et huit documents de test : ses poids existent et ne veulent rien dire.

### Invariants du traçage

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

**Répertoire :** [src/experiments/](src/experiments/)

Il porte la machinerie de campagne : lecture des fichiers d'expérience, boucle sur les douze runs, ablations, tableaux, figures.

Les commandes qui s'en servent :

| Commande | Effet |
| --- | --- |
| `make reproduce MODE=quick` | Joue toute la chaîne en une minute, plafonnée. Ne produit aucun résultat, et le dit |
| `make reproduce MODE=full` | Rejoue la campagne réelle, plusieurs heures |
| `make ablation` | Rejoue l'ablation de taille du corpus et écrit ses tableaux |
| `make report-sync` | Régénère les tableaux du rapport depuis les enregistrements |

Un enregistrement porte un statut, qui décide de son entrée dans les tableaux :

| Statut | Sens |
| --- | --- |
| `OK` | Mesure complète, entre dans les tableaux et au registre de modèles |
| `PARTIAL` | Run plafonné ou évalué sur un sous-ensemble. N'entre dans aucun tableau |
| `FAILED` | L'expérience a levé. L'erreur est enregistrée |
| `NOT_RUN` | Déclarée, jamais exécutée |
| `STALE_CONFIG` | Un répertoire de run subsiste, mais il a été écrit par une autre architecture que celle déclarée aujourd'hui. Ni tableau ni registre |

L'agrégation ne lit que les enregistrements `OK` porteurs d'une évaluation, condition écrite dans [`record.py`](src/experiments/record.py).

**L'ordre de la campagne n'est pas celui des noms de fichiers.** [`campaign_order`](src/experiments/config.py) fait passer `t5-small` initialisé aléatoirement d'abord, des plus petites proportions de corpus aux plus grandes, puis `t5-small` zero-shot, puis ses fine-tunes. Les trois entrées qui lancent des runs suivent cet ordre : `make reproduce`, `python -m src.experiments.run --all` et le carnet `02_training`.

---

## 8. Par où commencer

Dans cet ordre :

0. `make kernel` : enregistrer le kernel du dépôt, sans quoi les carnets tournent sur l'interpréteur du PATH. Sous PowerShell, `.\make.ps1 kernel`.
1. `notebooks/00_environment_check.ipynb` : ce poste peut-il exécuter la chaîne, et sur quoi.
2. `make data` : construire le corpus, une fois.
3. [notebooks/01_eda_cnn_dailymail.ipynb](notebooks/01_eda_cnn_dailymail.ipynb) : les mesures qui fixent les plafonds et le budget de décodage.
4. [notebooks/03_transformer_walkthrough.ipynb](notebooks/03_transformer_walkthrough.ipynb) : voir un batch réel traverser le modèle, forme par forme.
5. `notebooks/02_training.ipynb` en `MODE = "quick"` : rejouer les douze expériences en une minute et voir la mécanique de bout en bout.
